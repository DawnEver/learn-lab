"""Physics-informed fitting through the RESIDUAL-PROVIDER seam: learn-lab owns the training, never the physics.

A provider (the domain library's -- e.g. motronics hamilton for the magnetic vector potential) declares:

- ``domain``: the :class:`~learn_lab.validity.BoxEnvelope` of the coordinates the field is defined on;
- ``outputs``: how many field components;
- ``parameters``: its physical parameters by name, at their nominal values (ONE source of those values);
- ``observables``: what can be measured, by name -> its component count ``k``; the field itself is one, declared
  explicitly like any other (e.g. ``{'u': 2, 'du': 2}``, or ``{'A': 2, 'B': 4}``);
- ``observe(name, field, points)``: the observable at ``(n, dim)`` coordinates, ``(n, k)``, differentiable -- a
  derived quantity (a gradient, a curl) is computed from ``field`` with :func:`derivative`;
- ``owns(name, points)``: ``(n, k)`` booleans, which components of that observable ARE it at each point. A
  single-field provider owns every component everywhere; a provider with one column per region owns only the
  region's components, so an observation pins what it belongs to and never the neighbour's (ONE source of the
  region map: the provider that defines the regions);
- ``collocation(runtime)``: the points per region (``'interior'``, ``'boundary'``, ...) as tensors;
- ``residuals(field, points, parameters)``: one :class:`Term` per condition. ``field`` maps ``(n, dim)``
  coordinates to ``(n, outputs)`` and is differentiable: derive it with :func:`derivative`.

A :class:`Term` carries its ``scale`` IN THE RESIDUAL'S UNIT, so ``residual / scale`` is dimensionless and the
loss sums like with like -- a weight that is a bare number would hide the unit (TOLERANCE-CARRIES-A-UNIT).

A forward PINN and an inverse problem are the SAME fit: the inverse names which parameters to ``identify``
and supplies ``observations`` -- a mapping observable name -> ``(x, values)``, ``values`` of shape ``(n, k)``; those
parameters become trainable alongside the network. Only the entries ``provider.owns(name, x)`` marks are fitted; the
others are not the observable there and are ignored (NaN is the honest value to put in them). Each observable is one
data term scaled by the root-mean-square of its owned values, in its own unit: a derived observable (a slope, a flux
density) can carry a coefficient the field itself hides below the fit's accuracy.

The scale is the UNIT carrier; it cannot fix training dynamics. With several regions and interface conditions the
dimensionless terms' gradients can differ by orders of magnitude, and the weak ones (an interface, a boundary) stay
under-fitted. A :class:`Balance` multiplies each dimensionless term by a dimensionless weight: ``fixed`` (the
default, every weight 1) or ``gradient_norm`` -- learning-rate annealing (Wang, Teng & Perdikaris 2021): every
``every`` Adam steps ``w_i <- (1 - alpha) w_i + alpha max|grad L_ref| / mean|grad L_i|`` over the network weights,
the ``reference`` condition (the PDE interior) held at 1. Data terms join the balance as ``observed:<name>``: an
observation is a condition like any other, and under-fitting it is the same defect. The weights move during Adam
only; L-BFGS polishes the objective they settled on (a moving objective would void its curvature memory). The
final weights and each update are recorded on the :class:`FittedField`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = ['Balance', 'FittedField', 'PhysicsInformedField', 'ResidualProvider', 'Term', 'derivative']

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


@dataclass(frozen=True, slots=True)
class Balance:
    """How the dimensionless terms are weighted: ``fixed`` (all 1) or ``gradient_norm`` against ``reference``."""

    rule: str = 'fixed'
    reference: str = ''
    alpha: float = 0.1
    every: int = 10

    def __post_init__(self) -> None:
        if self.rule not in ('fixed', 'gradient_norm'):
            refuse(f"a balance rule is 'fixed' or 'gradient_norm', got {self.rule!r}")
        if self.rule == 'gradient_norm' and not self.reference:
            refuse('a gradient_norm balance names its reference condition (the PDE interior)')
        if not 0.0 < self.alpha <= 1.0:
            refuse(f'a balance alpha is in (0, 1], got {self.alpha}')
        if self.every < 1:
            refuse(f'a balance updates every >= 1 steps, got {self.every}')


class ResidualProvider(Protocol):
    @property
    def domain(self) -> BoxEnvelope: ...

    @property
    def outputs(self) -> int: ...

    @property
    def parameters(self) -> Mapping[str, float]: ...

    @property
    def observables(self) -> Mapping[str, int]: ...

    def observe(self, name: str, field: Field, points: torch.Tensor) -> torch.Tensor: ...

    def owns(self, name: str, points: np.ndarray) -> np.ndarray: ...

    def collocation(self, runtime: Runtime) -> Mapping[str, torch.Tensor]: ...

    def residuals(
        self, field: Field, points: Mapping[str, torch.Tensor], parameters: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Term]: ...


def derivative(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``d values / d points`` row-wise, ``(n,) -> (n, dim)``, differentiable again (second derivatives)."""
    return torch.autograd.grad(values.sum(), points, create_graph=True)[0]


def _mapped(net: MLP, runtime: Runtime, domain: BoxEnvelope, x: torch.Tensor) -> torch.Tensor:
    """The net at coordinates *x* mapped from the *domain* box onto ``[-1, 1]``."""
    lo, hi = runtime.tensor(domain.lower), runtime.tensor(domain.upper)
    return net(2.0 * (x - lo) / (hi - lo) - 1.0)


@dataclass(frozen=True, slots=True)
class PhysicsInformedField:
    hidden: Sequence[int] = (32, 32, 32)
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=500))
    runtime: Runtime = field(default_factory=Runtime)
    balance: Balance = field(default_factory=Balance)

    def fit(
        self,
        provider: ResidualProvider,
        *,
        identify: Sequence[str] = (),
        observations: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> FittedField:
        """Fit the field to the provider's residuals; ``identify`` + ``observations`` make it an inverse problem."""
        unknown = set(identify) - set(provider.parameters)
        if unknown:
            refuse(f'identify names {sorted(unknown)}; the provider declares {sorted(provider.parameters)}')
        if identify and observations is None:
            refuse('identifying parameters needs observations; none were given')
        rt = self.runtime
        rt.seeded()
        dim = provider.domain.width
        net = MLP([dim, *self.hidden, provider.outputs]).to(rt.torch_device, rt.torch_dtype)
        params = {
            name: torch.tensor(
                float(value), dtype=rt.torch_dtype, device=rt.torch_device, requires_grad=name in identify
            )
            for name, value in provider.parameters.items()
        }

        def field_at(x: torch.Tensor) -> torch.Tensor:
            return _mapped(net, rt, provider.domain, x)

        points = {name: p.detach().requires_grad_(True) for name, p in provider.collocation(rt).items()}
        data = [] if observations is None else [_observed(provider, rt, n, o) for n, o in observations.items()]
        if observations is not None and not data:
            refuse('observations name no observable; give at least one')

        balancing = self.balance.rule != 'fixed'
        if balancing and checkpoint is not None:
            refuse('a balanced fit cannot be checkpointed: its weights are training state the checkpoint omits')
        weights: dict[str, float] = {}
        updates: list[tuple[int, dict[str, float]]] = []
        calls = [0]

        def loss() -> torch.Tensor:
            terms = {name: term.loss() for name, term in provider.residuals(field_at, points, params).items()}
            for name, xt, ut, mask, scale in data:
                terms[f'observed:{name}'] = Term((provider.observe(name, field_at, xt) - ut)[mask], scale).loss()
            if not weights:
                if balancing and self.balance.reference not in terms:
                    refuse(f'the balance reference {self.balance.reference!r} is not a condition: {sorted(terms)}')
                weights.update(dict.fromkeys(terms, 1.0))
            step, calls[0] = calls[0], calls[0] + 1
            if balancing and step < self.schedule.adam_steps and step % self.balance.every == 0:
                self._rebalance(terms, list(net.parameters()), weights)
                updates.append((step, dict(weights)))
            return sum(weights[name] * value for name, value in terms.items())

        trainable = [*net.parameters(), *(params[name] for name in identify)]
        history = minimise(
            trainable,
            loss,
            self.schedule,
            checkpoint=checkpoint,
            config=repr(self),
            data=_trained_on(points, data, params),
        )
        net.eval()
        return FittedField(
            net=net,
            runtime=rt,
            domain=provider.domain,
            parameters={name: float(value.detach()) for name, value in params.items()},
            history=history,
            weights=dict(weights),
            weight_history=updates,
        )

    def _rebalance(self, terms: Mapping[str, torch.Tensor], theta: list[torch.Tensor], weights: dict) -> None:
        """One learning-rate-annealing update of *weights* from the terms' gradients over the network *theta*."""

        def grads(value: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(value, theta, retain_graph=True, allow_unused=True)
            return torch.cat([
                (torch.zeros_like(p) if d is None else d).abs().flatten() for p, d in zip(theta, g, strict=True)
            ])

        b = self.balance
        peak = float(grads(terms[b.reference]).max())
        for name, value in terms.items():
            mean = float(grads(value).mean()) if name != b.reference else 0.0
            if mean > 0.0:
                weights[name] = (1.0 - b.alpha) * weights[name] + b.alpha * peak / mean


def _observed(provider: ResidualProvider, rt: Runtime, name: str, observation: tuple[np.ndarray, np.ndarray]):
    """One observable's data term: points, values (unowned zeroed), ownership mask, RMS scale in its unit."""
    if name not in provider.observables:
        refuse(f'observable {name!r} is not declared; the provider declares {sorted(provider.observables)}')
    k = provider.observables[name]
    xo = provider.domain.require(observation[0], 'an observation point')
    uo = np.asarray(observation[1], float)
    if uo.shape != (xo.shape[0], k):
        refuse(f'observations of {name!r} are ({xo.shape[0]}, {k}) values, got {uo.shape}')
    owned = np.asarray(provider.owns(name, xo), bool)
    if owned.shape != uo.shape:
        refuse(f'provider.owns({name!r}) returns {owned.shape}, the observations are {uo.shape}')
    if not owned.any():
        refuse(f'no observed value of {name!r} lies in a component the provider owns at its point')
    values = uo[owned]
    if not np.isfinite(values).all():
        refuse(f'an observed value of {name!r} in an owned component is not finite')
    scale = float(np.sqrt(np.mean(values**2)))
    if not scale > 0:
        refuse(f'every owned observation of {name!r} is zero: it carries no scale in its unit')
    points = rt.tensor(xo).requires_grad_(True)
    return name, points, rt.tensor(np.where(owned, uo, 0.0)), torch.as_tensor(owned, device=rt.torch_device), scale


@dataclass(frozen=True, slots=True, eq=False)
class FittedField:
    """The fitted field of ONE problem instance and the parameters it was fitted with (identified ones updated)."""

    net: MLP
    runtime: Runtime
    domain: BoxEnvelope
    parameters: Mapping[str, float]
    history: list[float]
    weights: Mapping[str, float]  # each condition's final balance weight (all 1 under a fixed balance)
    weight_history: list[tuple[int, dict[str, float]]]  # (Adam step, weights after it) per balance update

    def field_at(self, x: torch.Tensor) -> torch.Tensor:
        """The field at ``(n, dim)`` coordinates, differentiable."""
        return _mapped(self.net, self.runtime, self.domain, x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self.domain.require(x, 'a field query')
        with torch.no_grad():
            return numpy(self.field_at(self.runtime.tensor(x)))


def _trained_on(points, data, params) -> list:
    """The arrays a physics fit trains on: the collocation points, the observations and the starting parameters."""
    observed = [array for _, xt, ut, mask, _ in data for array in (xt, ut, mask)]
    return [*(points[name] for name in sorted(points)), *observed, *(params[name] for name in sorted(params))]
