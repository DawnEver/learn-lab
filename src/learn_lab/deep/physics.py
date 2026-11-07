"""Physics-informed fitting through the RESIDUAL-PROVIDER seam: learn-lab owns the training, never the physics.

A provider (the domain library's -- e.g. motronics hamilton for the magnetic vector potential) declares:

- ``domain``: the :class:`~learn_lab.validity.BoxEnvelope` of the coordinates the field is defined on;
- ``outputs``: how many field components;
- ``parameters``: its physical parameters by name, at their nominal values (ONE source of those values);
- ``collocation(runtime)``: the points per region (``'interior'``, ``'boundary'``, ...) as tensors;
- ``residuals(field, points, parameters)``: one :class:`Term` per condition. ``field`` maps ``(n, dim)``
  coordinates to ``(n, outputs)`` and is differentiable: derive it with :func:`derivative`.

A :class:`Term` carries its ``scale`` IN THE RESIDUAL'S UNIT, so ``residual / scale`` is dimensionless and the
loss sums like with like -- a weight that is a bare number would hide the unit (TOLERANCE-CARRIES-A-UNIT).

A forward PINN and an inverse problem are the SAME fit: the inverse names which parameters to ``identify``
and supplies ``observations``; those parameters become trainable alongside the network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import mlp
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = ['FittedField', 'PhysicsInformedField', 'ResidualProvider', 'Term', 'derivative']

Field = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class Term:
    """``residual`` at the points of one condition, and the ``scale`` (same unit) that makes it dimensionless."""

    residual: torch.Tensor
    scale: float

    def __post_init__(self) -> None:
        if not self.scale > 0:
            refuse(f'a residual scale is > 0 in the residual unit, got {self.scale}')

    def loss(self) -> torch.Tensor:
        return ((self.residual / self.scale) ** 2).mean()


class ResidualProvider(Protocol):
    @property
    def domain(self) -> BoxEnvelope: ...

    @property
    def outputs(self) -> int: ...

    @property
    def parameters(self) -> Mapping[str, float]: ...

    def collocation(self, runtime: Runtime) -> Mapping[str, torch.Tensor]: ...

    def residuals(
        self, field: Field, points: Mapping[str, torch.Tensor], parameters: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Term]: ...


def derivative(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``d values / d points`` row-wise, ``(n,) -> (n, dim)``, differentiable again (second derivatives)."""
    return torch.autograd.grad(values.sum(), points, create_graph=True)[0]


@dataclass(frozen=True, slots=True)
class PhysicsInformedField:
    hidden: Sequence[int] = (32, 32, 32)
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=500))
    runtime: Runtime = field(default_factory=Runtime)

    def fit(
        self,
        provider: ResidualProvider,
        *,
        identify: Sequence[str] = (),
        observations: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> FittedField:
        """Fit the field to the provider's residuals; ``identify`` + ``observations`` make it an inverse problem."""
        unknown = set(identify) - set(provider.parameters)
        if unknown:
            refuse(f'identify names {sorted(unknown)}; the provider declares {sorted(provider.parameters)}')
        if identify and observations is None:
            refuse('identifying parameters needs observations of the field; none were given')
        rt = self.runtime
        rt.seeded()
        dim = provider.domain.width
        net = mlp([dim, *self.hidden, provider.outputs]).to(rt.torch_device, rt.torch_dtype)
        params = {
            name: torch.tensor(
                float(value), dtype=rt.torch_dtype, device=rt.torch_device, requires_grad=name in identify
            )
            for name, value in provider.parameters.items()
        }
        lo, hi = rt.tensor(provider.domain.lower), rt.tensor(provider.domain.upper)

        def field_at(x: torch.Tensor) -> torch.Tensor:
            return net(2.0 * (x - lo) / (hi - lo) - 1.0)

        points = {name: p.detach().requires_grad_(True) for name, p in provider.collocation(rt).items()}
        data = None
        if observations is not None:
            xo, uo = (
                provider.domain.require(observations[0], 'an observation point'),
                np.asarray(observations[1], float),
            )
            if uo.shape != (xo.shape[0], provider.outputs):
                refuse(f'observations are ({xo.shape[0]}, {provider.outputs}) values, got {uo.shape}')
            data = (rt.tensor(xo), rt.tensor(uo), float(np.std(uo)) or 1.0)

        def loss() -> torch.Tensor:
            total = sum(term.loss() for term in provider.residuals(field_at, points, params).values())
            if data is not None:
                total = total + Term(field_at(data[0]) - data[1], data[2]).loss()
            return total

        trainable = [*net.parameters(), *(params[name] for name in identify)]
        history = minimise(trainable, loss, self.schedule)
        net.eval()
        return FittedField(
            field_at=field_at,
            runtime=rt,
            domain=provider.domain,
            parameters={name: float(value.detach()) for name, value in params.items()},
            history=history,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedField:
    """The fitted field of ONE problem instance and the parameters it was fitted with (identified ones updated)."""

    field_at: Field
    runtime: Runtime
    domain: BoxEnvelope
    parameters: Mapping[str, float]
    history: list[float]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self.domain.require(x, 'a field query')
        with torch.no_grad():
            return numpy(self.field_at(self.runtime.tensor(x)))
