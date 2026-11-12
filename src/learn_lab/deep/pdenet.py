"""PDE-Net: learned finite-difference stencils under moment constraints, and a sparse symbolic response.

A 1-D periodic field ``u (n, N)`` on spacing ``dx`` is filtered by ``order + 1`` stencils of ``kernel`` points.
Stencil ``j`` is parametrised by its moments ``M_m = sum_s K_s s^m / m!`` (``s`` the offset in cells) and scaled
by ``dx^-j``. With ``constrained=True`` the moments ``m < j + accuracy`` are FIXED at ``delta_mj``: the stencil is
a ``j``-th derivative to order ``accuracy`` by construction, and only its higher moments are learned. The response
is ``u_t = sum_i c_i term_i`` over the PDE-FIND library ``u^p * d^k u`` (:func:`learn_lab.equation.pde_library`,
same names), so the identified law is read off its coefficients (:meth:`FittedPDENet.terms`).

``fit(before, after, dt=, dx=)`` trains on snapshot pairs ``after = u(t + dt)``, advancing ``before`` by one RK4
step of the learned rate: phase one penalises ``sparsity * |c|_1``, then every coefficient below ``threshold`` (in
the coefficient's unit) is masked to zero and the survivors are refitted without the penalty for ``refit_steps``.
``constrained=False`` frees every moment: the planted control, where a coefficient and its stencil's scale trade
off and the PDE's coefficients are not identified. The envelope is the box of ``u`` values the fit saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.equation import pde_library
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedPDENet', 'PDENet', 'StencilResponse']


def _moment_matrix(kernel: int) -> np.ndarray:
    """``(kernel, kernel)``: row ``m`` maps a stencil's weights to its ``m``-th moment ``sum_s K_s s^m / m!``."""
    offsets = np.arange(kernel) - kernel // 2
    return np.stack([offsets.astype(float) ** m / math.factorial(m) for m in range(kernel)])


class StencilResponse(nn.Module):
    """The moment-parametrised stencils and the library response: ``u (n, N) -> u_t (n, N)``, periodic."""

    def __init__(
        self, *, order: int, kernel: int, accuracy: int, degree: int, dx: float, constrained: bool, seed: int
    ) -> None:
        super().__init__()
        self.arguments = {
            'order': order,
            'kernel': kernel,
            'accuracy': accuracy,
            'degree': degree,
            'dx': dx,
            'constrained': constrained,
            'seed': seed,
        }
        library = pde_library(order=order, degree=degree)
        self.names = library.names
        self.factors = [term.factors for term in library.terms]
        fixed = np.zeros((order + 1, kernel), bool)
        target = np.zeros((order + 1, kernel))
        for j in range(order + 1):
            target[j, j] = 1.0
            if constrained:
                fixed[j, : j + accuracy] = True
        fixed[0] = True  # the order-0 stencil is the identity: u itself is never a learned filter
        free = torch.randn((order + 1, kernel), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        self.register_buffer('fixed', torch.as_tensor(fixed))
        self.register_buffer('target', torch.as_tensor(target))
        self.register_buffer('inverse', torch.as_tensor(np.linalg.inv(_moment_matrix(kernel))))
        self.register_buffer('scale', torch.as_tensor(float(dx)) ** -torch.arange(order + 1, dtype=torch.float64))
        self.register_buffer('mask', torch.ones(len(self.names), dtype=torch.float64))
        self.moments = nn.Parameter(
            torch.where(torch.as_tensor(fixed), torch.as_tensor(target), torch.as_tensor(target) + 0.1 * free)
        )
        self.coefficients = nn.Parameter(torch.zeros(len(self.names), dtype=torch.float64))

    def stencils(self) -> torch.Tensor:
        """``(order + 1, kernel)`` weights in physical units (``dx^-j`` applied); fixed moments held at target."""
        moments = torch.where(self.fixed, self.target.to(self.moments.dtype), self.moments)
        return (moments @ self.inverse.to(moments.dtype).T) * self.scale.to(moments.dtype)[:, None]

    def features(self, u: torch.Tensor) -> torch.Tensor:
        """``(n, terms, N)``: every library term at every point."""
        half = self.arguments['kernel'] // 2
        padded = torch.cat([u[:, -half:], u, u[:, :half]], dim=1)[:, None]
        derivatives = nn.functional.conv1d(padded, self.stencils()[:, None])  # (n, order + 1, N)
        ones = torch.ones_like(u)
        return torch.stack([math.prod((derivatives[:, i] for i in f), start=ones) for f in self.factors], dim=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.einsum('t,ntx->nx', self.coefficients * self.mask.to(u.dtype), self.features(u))


def _rk4(net: StencilResponse, u: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = net(u)
    k2 = net(u + dt / 2 * k1)
    k3 = net(u + dt / 2 * k2)
    k4 = net(u + dt * k3)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass(frozen=True, slots=True)
class PDENet:
    order: int = 2
    kernel: int = 5
    accuracy: int = 2
    degree: int = 2
    constrained: bool = True
    sparsity: float = 1e-4
    threshold: float = 0.05
    refit_steps: int = 500
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=2000, learning_rate=1e-2))
    runtime: Runtime = field(default_factory=Runtime)

    def __post_init__(self) -> None:
        if self.order < 1 or self.degree < 0 or self.accuracy < 1:
            refuse(
                f'a PDE-Net has order >= 1, degree >= 0, accuracy >= 1; got {self.order}, {self.degree}, '
                f'{self.accuracy}'
            )
        if self.kernel % 2 == 0 or self.kernel < self.order + self.accuracy:
            refuse(
                f'an order-{self.order} stencil of accuracy {self.accuracy} needs an odd kernel of >= '
                f'{self.order + self.accuracy} points, got {self.kernel}'
            )
        if self.sparsity < 0 or self.threshold < 0 or self.refit_steps < 0:
            refuse(
                f'sparsity, threshold and refit steps are >= 0, got {self.sparsity}, {self.threshold}, '
                f'{self.refit_steps}'
            )

    def fit(
        self, before: np.ndarray, after: np.ndarray, *, dt: float, dx: float, checkpoint: Checkpoint | None = None
    ) -> FittedPDENet:
        """Snapshot pairs ``before, after (n, N)`` on a periodic grid of spacing *dx*, ``after = u(t + dt)``."""
        u0 = require_finite(np.asarray(before, float), 'the snapshots before')
        u1 = require_finite(np.asarray(after, float), 'the snapshots after')
        if u0.ndim != 2 or u0.shape != u1.shape or not len(u0) or u0.shape[1] < self.kernel:
            refuse(f'a PDE-Net is fitted on pairs (n, N >= {self.kernel}) of one shape, got {u0.shape}, {u1.shape}')
        if not (dt > 0 and dx > 0):
            refuse(f'a PDE grid has dt > 0 and dx > 0, got {dt}, {dx}')
        rt = self.runtime
        rt.seeded()
        net = StencilResponse(
            order=self.order,
            kernel=self.kernel,
            accuracy=self.accuracy,
            degree=self.degree,
            dx=float(dx),
            constrained=self.constrained,
            seed=rt.seed,
        ).to(rt.torch_device, rt.torch_dtype)
        x0, x1 = rt.tensor(u0), rt.tensor(u1)
        # The residual is a rate: the one-step change over dt, so its scale is the data's, not dt's.
        rate = (x1 - x0) / dt

        def misfit() -> torch.Tensor:
            return (((_rk4(net, x0, dt) - x0) / dt - rate) ** 2).mean() / (rate**2).mean()

        def penalised() -> torch.Tensor:
            return misfit() + self.sparsity * net.coefficients.abs().sum()

        history = minimise(
            net.parameters(),
            penalised,
            self.schedule,
            checkpoint=checkpoint and checkpoint.member(0),
            config=repr(self),
            data=[x0, x1],
        )
        with torch.no_grad():
            net.mask.copy_((net.coefficients.abs() >= self.threshold).to(net.mask.dtype))
            net.coefficients.mul_(net.mask)
        if self.refit_steps:
            refit = replace(self.schedule, adam_steps=self.refit_steps, lbfgs_steps=0)
            history += minimise(
                net.parameters(),
                misfit,
                refit,
                checkpoint=checkpoint and checkpoint.member(1),
                config=repr(self),
                data=[x0, x1],
            )
        return FittedPDENet(net=net.eval(), runtime=rt, envelope=BoxEnvelope.of(u0.reshape(-1, 1)), history=history)


@dataclass(frozen=True, slots=True, eq=False)
class FittedPDENet:
    net: StencilResponse
    runtime: Runtime
    envelope: BoxEnvelope
    history: list[float]

    def terms(self) -> dict[str, float]:
        """The identified law's nonzero terms by name: ``u_t = sum terms[name] * name``."""
        c = numpy(self.net.coefficients * self.net.mask)
        return {name: float(v) for name, v in zip(self.net.names, c, strict=True) if v != 0.0}

    def stencils(self) -> np.ndarray:
        """``(order + 1, kernel)`` learned stencil weights in physical units (row ``j`` approximates ``d^j/dx^j``)."""
        with torch.no_grad():
            return numpy(self.net.stencils())

    def _require(self, u: np.ndarray) -> torch.Tensor:
        u = np.asarray(u, float)
        if u.ndim != 2 or u.shape[1] < self.net.arguments['kernel']:
            refuse(f'a PDE-Net query is (n, N >= {self.net.arguments["kernel"]}), got {u.shape}')
        self.envelope.require(u.reshape(-1, 1), 'a PDE-Net state')
        return self.runtime.tensor(u)

    def rate(self, u: np.ndarray) -> np.ndarray:
        """``u_t (n, N)`` at in-domain states ``u (n, N)``."""
        with torch.no_grad():
            return numpy(self.net(self._require(u)))

    def advance(self, u: np.ndarray, dt: float) -> np.ndarray:
        """``u (n, N)`` advanced by one RK4 step of *dt* under the learned law."""
        with torch.no_grad():
            return numpy(_rk4(self.net, self._require(u), dt))
