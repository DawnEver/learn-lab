"""The gradient seam between torch and a classical solver: a numpy forward and its numpy VJP as one torch op.

A classical solver (numpy, or Rust behind numpy) stays where it is; learn-lab trains THROUGH it. The solver
hands two functions:

- ``forward(x) -> y``: arrays in, arrays out (any shapes);
- ``vjp(x, y, cotangent) -> dL/dx``: ``cotangent = dL/dy``, shaped like ``y``; the result shaped like ``x``.
  ``y`` is the forward's own output, so an adjoint (``J^T lambda = cotangent`` at the converged state) never
  re-solves.

:func:`classical_op` wraps them as a ``torch.autograd.Function``: the tensor leaves its device as float64 numpy,
the solver runs, the result returns on the caller's device and dtype. The op is first-order only -- a second
derivative through a hand-written VJP would be a guess. :func:`check_vjp` measures the VJP against central
finite differences BEFORE a fit trusts it: a wrong adjoint trains silently towards the wrong optimum.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from learn_lab.deep.runtime import numpy
from learn_lab.errors import refuse, require_finite

__all__ = ['check_vjp', 'classical_op']

Forward = Callable[[np.ndarray], np.ndarray]
Vjp = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _array(value: np.ndarray, what: str) -> np.ndarray:
    return require_finite(np.asarray(value, dtype=float), what)


def classical_op(forward: Forward, vjp: Vjp) -> Callable[[torch.Tensor], torch.Tensor]:
    """A torch-callable ``y = forward(x)`` whose backward is ``vjp``; float64 numpy inside, the caller's device out."""

    class ClassicalOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
            xn = numpy(x)
            yn = _array(forward(xn), 'the classical forward')
            ctx.arrays = (xn, yn)
            return torch.as_tensor(yn, dtype=x.dtype, device=x.device)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: torch.autograd.function.FunctionCtx, cotangent: torch.Tensor) -> torch.Tensor:
            xn, yn = ctx.arrays
            gx = _array(vjp(xn, yn, numpy(cotangent)), 'the classical vjp')
            if gx.shape != xn.shape:
                refuse(f'a vjp returns the shape of x {xn.shape}, got {gx.shape}')
            return torch.as_tensor(gx, dtype=cotangent.dtype, device=cotangent.device)

    return ClassicalOp.apply


def check_vjp(
    forward: Forward,
    vjp: Vjp,
    x: np.ndarray,
    *,
    step: float = 1e-6,
    tolerance: float = 1e-6,
    floor: float = 1e-12,
    seed: int = 0,
) -> float:
    """The relative error of ``vjp`` against central differences of ``<c, forward(x)>``; refused above *tolerance*.

    ``c`` is a random cotangent (seeded), so every output component is weighed. The error is
    ``|g_vjp - g_fd| / max(|g_fd|, floor)`` in the 2-norm; *step* is in x's unit, *floor* in the gradient's
    unit (``L``-unit per x-unit) -- it keeps a vanishing gradient from turning noise into a relative error.
    The default *tolerance* fits a smooth float64 forward at the default *step*: the central difference
    carries ``O(step^2)`` truncation and ``O(eps / step)`` rounding, both below 1e-8 relative.
    """
    x = _array(x, 'the check point')
    y = _array(forward(x), 'the classical forward')
    cotangent = np.random.default_rng(seed).standard_normal(y.shape)
    got = _array(vjp(x, y, cotangent), 'the classical vjp')
    if got.shape != x.shape:
        refuse(f'a vjp returns the shape of x {x.shape}, got {got.shape}')
    flat, fd = x.ravel(), np.empty(x.size)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = step
        up = np.vdot(cotangent, forward((flat + e).reshape(x.shape)))
        down = np.vdot(cotangent, forward((flat - e).reshape(x.shape)))
        fd[i] = (up - down) / (2 * step)
    error = float(np.linalg.norm(got.ravel() - fd) / max(float(np.linalg.norm(fd)), floor))
    if not error <= tolerance:
        refuse(f'the vjp disagrees with central differences: relative error {error:.3e} > {tolerance:.1e}')
    return error
