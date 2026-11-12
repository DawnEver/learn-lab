"""The torch-backed model families. Needs the ``[torch]`` extra; the only package that imports torch.

Every family keeps the numpy boundary: arrays in, arrays out. Tensors never cross it.

- :mod:`~learn_lab.deep.surrogate` -- offline response surrogate (deep ensemble: mean, variance, gradient);
- :mod:`~learn_lab.deep.physics` -- the residual-provider seam; PINN and inverse problems are one fit;
- :mod:`~learn_lab.deep.constitutive` -- a monotone learned law with its derivative, the hybrid seam's callable;
- :mod:`~learn_lab.deep.field` -- grid -> grid (U-Net), enveloped by channel values and grid shape;
- :mod:`~learn_lab.deep.diffusion` -- conditional field generation (DDPM over a U-Net): samples, their mean and
  variance, enveloped by the condition box;
- :mod:`~learn_lab.deep.operator` -- function -> function: DeepONet, 1-D/2-D FNO, transformer (attention) operator on
  irregular points, graph neural operator on mesh nodes (discretisation-invariant);
- :mod:`~learn_lab.deep.graph` -- node fields on numpy graph batches by message passing;
- :mod:`~learn_lab.deep.preconditioner` -- a learned SPD preconditioner ``M(r) -> z`` (a correction to Jacobi) for
  any Krylov solver;
- :mod:`~learn_lab.deep.hysteresis` -- a play-bank hysteresis, a pure ``(state, x) -> (state, y)`` step, passive
  by construction in its DECLARED causality (effort- or displacement-driven);
- :mod:`~learn_lab.deep.bridge` -- a classical numpy solve and its VJP as one torch op, checked by differences;
- :mod:`~learn_lab.deep.dynamics` -- a neural ODE ``dx/dt = f(x)`` (RK4); sparse identification (ODE and PDE) is
  numpy, in :mod:`learn_lab.equation`;
- :mod:`~learn_lab.deep.pdenet` -- PDE-Net: moment-constrained learned stencils and a sparse response over the
  PDE-FIND library, identifying a 1-D periodic PDE from snapshot pairs.
"""

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - the extra is the one remedy
    msg = 'learn_lab.deep needs the torch extra: install learn-lab[torch]'
    raise ModuleNotFoundError(msg) from exc
