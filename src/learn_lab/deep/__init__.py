"""The torch-backed model families. Needs the ``[torch]`` extra; the only package that imports torch.

Every family keeps the numpy boundary: arrays in, arrays out. Tensors never cross it.

- :mod:`~learn_lab.deep.surrogate` -- offline response surrogate (deep ensemble: mean, variance, gradient);
- :mod:`~learn_lab.deep.physics` -- the residual-provider seam; PINN and inverse problems are one fit;
- :mod:`~learn_lab.deep.constitutive` -- a monotone learned law with its derivative, the hybrid seam's callable.
"""

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - the extra is the one remedy
    msg = 'learn_lab.deep needs the torch extra: install learn-lab[torch]'
    raise ModuleNotFoundError(msg) from exc
