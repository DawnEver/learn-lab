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

Where ``gradient_norm`` fails (measured, motronics EM cells, alpha 0.1, every 10, reference ``pde:steel``): it
made both WORSE than fixed weights -- a saturating steel bar max|dB| 4.9e-2 T -> 5.13e-1 T, an identified
saturation scale (truth 1.15) 1.1362 -> 1.2094 -- with side weights at 1.5e5-6.7e6 against the PDE's 1. The rule
assumes the side terms are under-fitted; when the unit scaling already satisfies one, its mean gradient is tiny
and ``max / mean`` has no upper bound (the max of millions of entries is an extreme statistic besides), the
weights only grow since nothing renormalises them, and the PDE term's share of the loss falls towards zero. Use it
only where a side condition is demonstrably under-fitted at fixed weights (the planted stiff Poisson).

DEEP RITZ (E & Yu 2018) is the same fit with another objective: given :class:`Ritz`, the field MINIMISES the
provider's ``energy(field, points, parameters)`` instead of squaring its residuals. The provider states the physics
-- per region an :class:`Energy` density with the region's ``measure`` (its length/area/volume in coordinate units)
and the energy's ``scale`` -- plus, as :class:`Term` s, the ESSENTIAL conditions the energy does not carry (a
Dirichlet side); natural conditions and interface fluxes are the energy's own stationarity and are never stated.
The quadrature is the provider's choice, carried by each :class:`Energy`. MONTE CARLO (a ``measure``):
``measure * mean(density)`` over the region's points, redrawn from the provider's own sampler under a fresh seed every
``every`` Adam steps so the minimiser cannot fit one point set; an L-BFGS polish runs on the last draw and fits ITS
quadrature error (measured, planted Poisson, 1024 points: 7.5e-2 polished against 1.1e-2 Adam only), so a Monte Carlo
Ritz schedule is Adam only. DETERMINISTIC (``weights``, e.g. :func:`gauss_legendre` per box region or
:func:`simplex_rule` per polygon region, a material interface a cell boundary): ``sum(weights * density)``, exact for polynomials to the rule's order, so the objective is the true
energy to that order and an L-BFGS polish is sound (measured on the planted two-slab Poisson with a 1:4 jump in
``tests/unit/deep/test_physics_ritz_quadrature.py``, whose bars quote the numbers). An essential term enters as ``penalty * term.loss()``.

A penalised essential condition biases the minimiser by O(1 / penalty) and stiffens it as the penalty grows, so no
penalty is right. A HARD ANSATZ removes both: with ``ansatz=True`` the network's ``latent`` raw columns are not the
field; the provider's ``ansatz(x, raw)`` maps them to the ``(n, outputs)`` field so that the essential conditions
hold by construction -- ``g(x) + d(x) * N(x)``, ``d`` vanishing on the Dirichlet set and ``g`` lifting its data; for
one column per region a SHARED TRACE (one raw column common to both regions plus per-region corrections times a
distance to the interface) makes interface continuity exact. The provider owns the geometry, so it supplies ``d`` and
``g``, and stops returning the terms its ansatz enforces; any :class:`Term` it still returns is penalised as before.
The ansatz is provider code, so a hard-ansatz :class:`FittedField` evaluates in process and is not an artifact.
Measured in ``tests/unit/deep/test_physics_ritz_hard.py``, whose bars quote the numbers.

INPUT FEATURES. A provider with ``features(x) -> (n, m)`` gives the net that as its input instead of ``x`` mapped
onto ``[-1, 1]``: periodic features (``cos``/``sin`` of a coordinate over its period) make every raw column periodic
by construction, what an exactly periodic ansatz needs. Like the ansatz it is provider code: such a field evaluates in
process.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import torch

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = [
    'Balance',
    'Continuation',
    'Energy',
    'FittedField',
    'PhysicsInformedField',
    'Resample',
    'ResidualProvider',
    'Ritz',
    'Term',
    'derivative',
    'gauss_legendre',
    'simplex_rule',
]

Field = Callable[[torch.Tensor], torch.Tensor]
Ansatz = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, raw net columns) -> field
Features = Callable[[torch.Tensor], torch.Tensor]  # (n, dim) coordinates -> (n, m) net input


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
class Energy:
    """An energy ``density`` at the points of one region and the energy ``scale`` (its unit) that makes the region's
    integral dimensionless. The integral is EITHER Monte Carlo -- ``measure * mean(density)``, ``measure`` the
    region's length/area/volume in coordinate units -- OR a deterministic quadrature -- ``sum(weights * density)``,
    one weight per point (e.g. :func:`gauss_legendre`, whose weights sum to the measure). An energy may be negative.
    """

    density: torch.Tensor
    scale: float
    measure: float | None = None
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.measure is None) == (self.weights is None):
            refuse('an energy integrates by a measure or quadrature weights: give exactly one')
        if self.measure is not None and not self.measure > 0:
            refuse(f'a region measure is > 0 in coordinate units, got {self.measure}')
        if self.weights is not None and self.weights.shape != self.density.shape:
            shapes = f'{tuple(self.weights.shape)} for {tuple(self.density.shape)}'
            refuse(f'a quadrature carries one weight per point: {shapes}')
        if not self.scale > 0:
            refuse(f'an energy scale is > 0 in the energy unit, got {self.scale}')

    def loss(self) -> torch.Tensor:
        if self.weights is None:
            return self.measure * self.density.mean() / self.scale
        return (self.weights * self.density).sum() / self.scale


def gauss_legendre(
    lower: Sequence[float], upper: Sequence[float], order: int, runtime: Runtime, *, cells: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """A composite tensor Gauss-Legendre rule on the box ``[lower, upper]``: ``cells`` equal cells per coordinate,
    ``order`` points per cell per coordinate, exact for polynomials of degree ``2 order - 1`` per coordinate on each
    cell. Returns points ``(n, dim)`` and weights ``(n,)`` summing to the box's measure; give each material region
    its own box so an interface is a cell boundary, never inside one.
    """
    if order < 1 or cells < 1:
        refuse(f'a Gauss-Legendre rule takes order >= 1 and cells >= 1, got order={order} cells={cells}')
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or lo.ndim != 1 or not (hi > lo).all():
        refuse(f'a quadrature box has lower < upper per coordinate, got {lo} and {hi}')
    node, weight = np.polynomial.legendre.leggauss(order)
    axes, factors = [], []
    for a, b in zip(lo, hi, strict=True):
        edges = np.linspace(a, b, cells + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        axes.append(((edges[:-1] + half)[:, None] + half[:, None] * node).ravel())
        factors.append((half[:, None] * weight).ravel())
    points = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], axis=1)
    weights = np.prod(np.stack([g.ravel() for g in np.meshgrid(*factors, indexing='ij')], axis=1), axis=1)
    return runtime.tensor(points), runtime.tensor(weights)


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice the signed area of the triangle ``a b c``: > 0 counter-clockwise."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _ear_clip(v: np.ndarray) -> np.ndarray:
    """The simple polygon ``v (k, 2)`` as ``(k - 2, 3, 2)`` counter-clockwise triangles."""
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        refuse(f'a polygon is (k >= 3, 2) vertices, got shape {v.shape}')
    x, y = v[:, 0], v[:, 1]
    area = 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    if area == 0.0:
        refuse(f'a polygon to integrate over has area, got {v.tolist()}')
    pts = v if area > 0 else v[::-1]
    tol = 1e-12 * float(np.max(np.ptp(pts, axis=0))) ** 2  # a flat corner is not an ear
    idx = list(range(len(pts)))
    out: list[np.ndarray] = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, m = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i], pts[j], pts[m]
            if _cross2(a, b, c) <= tol:
                continue  # reflex or flat
            others = (pts[n] for n in idx if n not in (i, j, m))
            if any(_cross2(a, b, p) >= -tol and _cross2(b, c, p) >= -tol and _cross2(c, a, p) >= -tol for p in others):
                continue  # another vertex inside or on it
            out.append(np.stack([a, b, c]))
            del idx[k]
            break
        else:
            refuse(f'polygon {v.tolist()} is not simple: no ear to clip')
    last = pts[idx]
    if _cross2(*last) <= tol:
        refuse(f'polygon {v.tolist()} is not simple: its last triangle is not counter-clockwise')
    out.append(last)
    return np.stack(out)


def simplex_rule(
    vertices: np.ndarray, order: int, runtime: Runtime, *, cells: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """A quadrature on the simple polygon ``vertices (k, 2)`` (either orientation), exact for polynomials of TOTAL
    degree ``2 order - 1``. The polygon is ear-clipped into triangles, each split into ``cells^2`` similar ones, each
    carrying the collapsed (Duffy) Gauss-Legendre rule of ``order x (order + 1)`` points. Returns points ``(n, 2)``
    strictly inside and positive weights ``(n,)`` summing to the area; a polygon region is to :func:`gauss_legendre`
    what a box region is.
    """
    if order < 1 or cells < 1:
        refuse(f'a simplex rule takes order >= 1 and cells >= 1, got order={order} cells={cells}')
    triangles = _ear_clip(np.asarray(vertices, dtype=float))
    gu, wu = np.polynomial.legendre.leggauss(order)
    gv, wv = np.polynomial.legendre.leggauss(order + 1)
    u, t1 = 0.5 * (gu + 1.0), 0.5 * (gv + 1.0)
    uu, tt = np.meshgrid(u, t1, indexing='ij')
    s, t = (uu * (1.0 - tt)).ravel(), tt.ravel()  # on the unit triangle
    w_ref = (0.25 * wu[:, None] * wv[None, :] * (1.0 - t1)[None, :]).ravel()  # sums to 1/2
    subs = []
    for i in range(cells):
        for j in range(cells - i):
            subs.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j < cells - 1:
                subs.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    points, weights = [], []
    for a, b, c in triangles:
        e1, e2 = (b - a) / cells, (c - a) / cells
        for corners in subs:
            p0, p1, p2 = (a + i * e1 + j * e2 for i, j in corners)
            points.append(p0 + s[:, None] * (p1 - p0) + t[:, None] * (p2 - p0))
            weights.append(_cross2(p0, p1, p2) * w_ref)
    return runtime.tensor(np.concatenate(points)), runtime.tensor(np.concatenate(weights))


@dataclass(frozen=True, slots=True)
class Ritz:
    """Train on the provider's energy (deep Ritz): essential :class:`Term` s weighted by ``penalty``, the quadrature
    points redrawn every ``every`` Adam steps (0: never).
    """

    penalty: float = 100.0
    every: int = 0

    def __post_init__(self) -> None:
        if not self.penalty > 0:
            refuse(f'a Ritz penalty is a dimensionless weight > 0, got {self.penalty}')
        if self.every < 0:
            refuse(f'a Ritz redraw period is >= 0 Adam steps, got {self.every}')


#: The balance rules: ``fixed``; ``gradient_norm`` (max/mean, unbounded); ``normalised_gradient`` (mean/mean
#: targets scaled to sum to the term count, so every weight stays in [0, N] -- but a condition the field already meets
#: has a tiny gradient and a huge target, and the rescale then starves the REFERENCE: measured, motronics steel bar,
#: pde 9.3e-5 against neumann 2.86); ``anchored_gradient`` (mean/mean, the reference held at 1, every other weight
#: clipped to [1 / cap, cap], so neither the reference nor any term can be starved or run away).
RULES = ('fixed', 'gradient_norm', 'normalised_gradient', 'anchored_gradient')


@dataclass(frozen=True, slots=True)
class Balance:
    """How the dimensionless terms are weighted: ``fixed`` (all 1) or a gradient rule against ``reference``."""

    rule: str = 'fixed'
    reference: str = ''
    alpha: float = 0.1
    every: int = 10
    cap: float = 100.0  # anchored_gradient: the largest ratio of any weight to the reference's 1

    def __post_init__(self) -> None:
        if not self.cap >= 1.0:
            refuse(f'a balance cap is >= 1 (a ratio to the reference weight), got {self.cap}')
        if self.rule not in RULES:
            refuse(f'a balance rule is one of {RULES}, got {self.rule!r}')
        if self.rule != 'fixed' and not self.reference:
            refuse(f'a {self.rule} balance names its reference condition (the PDE interior)')
        if not 0.0 < self.alpha <= 1.0:
            refuse(f'a balance alpha is in (0, 1], got {self.alpha}')
        if self.every < 1:
            refuse(f'a balance updates every >= 1 steps, got {self.every}')


@dataclass(frozen=True, slots=True)
class Continuation:
    """Move the provider parameter ``parameter`` from ``start`` to its nominal value over the first ``steps`` Adam
    steps, linearly, then hold it there: a homotopy from an easier problem (e.g. a saturation scale stretching a
    nonlinear law towards its linear start) into the one asked. A continued parameter is never identified.
    """

    parameter: str
    start: float
    steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            refuse(f'a continuation moves its parameter over >= 1 steps, got {self.steps}')


@dataclass(frozen=True, slots=True)
class Resample:
    """Residual-based adaptive refinement (RAR, Lu et al. 2021): every ``every`` Adam steps, draw a fresh set of the
    provider's collocation points (its own sampler under the next seed) and ADD, to each condition named in
    ``conditions``, the ``add`` candidates where that condition's dimensionless residual is largest. A condition
    names a point set whose residual term carries the same name (e.g. ``'pde:steel'``).
    """

    conditions: tuple[str, ...]
    every: int = 500
    add: int = 100

    def __post_init__(self) -> None:
        if not self.conditions:
            refuse('a resample names the conditions it refines')
        if self.every < 1 or self.add < 1:
            refuse(f'a resample adds >= 1 points every >= 1 steps, got add={self.add} every={self.every}')


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


def _field(
    net: MLP, runtime: Runtime, domain: BoxEnvelope, ansatz: Ansatz | None, features: Features | None, x: torch.Tensor
) -> torch.Tensor:
    """The field at coordinates *x*: the net on the provider's *features* of *x*, else on *x* mapped from the
    *domain* box onto ``[-1, 1]``, through the provider's hard *ansatz* when the fit has one.
    """
    if features is None:
        lo, hi = runtime.tensor(domain.lower), runtime.tensor(domain.upper)
        raw = net(2.0 * (x - lo) / (hi - lo) - 1.0)
    else:
        raw = net(features(x))
    return raw if ansatz is None else ansatz(x, raw)


@dataclass(frozen=True, slots=True)
class PhysicsInformedField:
    hidden: Sequence[int] = (32, 32, 32)
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=1e-3, lbfgs_steps=500))
    runtime: Runtime = field(default_factory=Runtime)
    balance: Balance = field(default_factory=Balance)
    continuation: Sequence[Continuation] = ()
    resample: Resample | None = None
    input_scale: bool = False  # a learned per-coordinate input factor (networks.InputScale) before the first layer
    ritz: Ritz | None = None  # minimise the provider's energy (deep Ritz) instead of its squared residuals
    ansatz: bool = False  # the field is provider.ansatz(x, net(x)): essential conditions hard, not penalised

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
        ansatz = None
        if self.ansatz:
            latent = getattr(provider, 'latent', None)
            if not callable(getattr(provider, 'ansatz', None)) or not isinstance(latent, int) or latent < 1:
                refuse(f'a hard-ansatz fit needs ansatz(x, raw) and latent >= 1; {type(provider).__name__} lacks them')
            ansatz = provider.ansatz
        features = getattr(provider, 'features', None)
        if features is not None:
            dim = int(features(rt.tensor(provider.domain.lower[None, :])).shape[1])
        net = MLP(
            [dim, *self.hidden, provider.latent if self.ansatz else provider.outputs], input_scale=self.input_scale
        ).to(rt.torch_device, rt.torch_dtype)
        params = {
            name: torch.tensor(
                float(value), dtype=rt.torch_dtype, device=rt.torch_device, requires_grad=name in identify
            )
            for name, value in provider.parameters.items()
        }

        def field_at(x: torch.Tensor) -> torch.Tensor:
            return _field(net, rt, provider.domain, ansatz, features, x)

        points = {name: p.detach().requires_grad_(True) for name, p in provider.collocation(rt).items()}
        data = [] if observations is None else [_observed(provider, rt, n, o) for n, o in observations.items()]
        if observations is not None and not data:
            refuse('observations name no observable; give at least one')

        balancing = self.balance.rule != 'fixed'
        if self.ritz is not None:
            if not callable(getattr(provider, 'energy', None)):
                refuse(f'a Ritz fit minimises an energy; {type(provider).__name__} declares no energy')
            if balancing or self.resample is not None:
                refuse('a Ritz fit takes a fixed balance and no residual refinement: an energy is not a residual')
            if self.ritz.every and checkpoint is not None:
                refuse('a redrawn Ritz fit cannot be checkpointed: its quadrature draw is training state it omits')
        if balancing and checkpoint is not None:
            refuse('a balanced fit cannot be checkpointed: its weights are training state the checkpoint omits')
        if (self.continuation or self.resample is not None) and checkpoint is not None:
            refuse('a continued or resampled fit cannot be checkpointed: its schedule is training state it omits')
        for c in self.continuation:
            if c.parameter not in params:
                refuse(f'continuation names {c.parameter!r}; the provider declares {sorted(params)}')
            if c.parameter in identify:
                refuse(f'{c.parameter!r} is identified; a continued parameter follows the schedule, not the fit')
        nominal = {c.parameter: float(params[c.parameter]) for c in self.continuation}
        if self.resample is not None:
            missing = set(self.resample.conditions) - set(points)
            if missing:
                refuse(f'resample names {sorted(missing)}; the provider samples {sorted(points)}')
        weights: dict[str, float] = {}
        updates: list[tuple[int, dict[str, float]]] = []
        calls = [0]

        def loss() -> torch.Tensor:
            step, calls[0] = calls[0], calls[0] + 1
            if step < self.schedule.adam_steps:
                self._continue(step, params, nominal)
                if self.resample is not None and step > 0 and step % self.resample.every == 0:
                    self._refine(provider, field_at, points, params, step // self.resample.every)
            terms = self._terms(provider, field_at, points, params, step)
            for name, xt, ut, mask, scale in data:
                terms[f'observed:{name}'] = Term((provider.observe(name, field_at, xt) - ut)[mask], scale).loss()
            if not weights:
                if balancing and self.balance.reference not in terms:
                    refuse(f'the balance reference {self.balance.reference!r} is not a condition: {sorted(terms)}')
                weights.update(dict.fromkeys(terms, 1.0))
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
            ansatz=ansatz,
            features=features,
        )

    def _terms(self, provider, field_at: Field, points: dict, params: Mapping, step: int) -> dict[str, torch.Tensor]:
        """Each condition's dimensionless loss: the squared residuals, or under :class:`Ritz` the energies and the
        penalised essential terms on the current quadrature draw.
        """
        if self.ritz is None:
            return {name: term.loss() for name, term in provider.residuals(field_at, points, params).items()}
        every = self.ritz.every
        if every and 0 < step < self.schedule.adam_steps and step % every == 0:
            fresh = provider.collocation(replace(self.runtime, seed=self.runtime.seed + step // every))
            points.update({name: p.detach().requires_grad_(True) for name, p in fresh.items()})
        return {
            name: part.loss() if isinstance(part, Energy) else self.ritz.penalty * part.loss()
            for name, part in provider.energy(field_at, points, params).items()
        }

    def _continue(self, step: int, params: dict[str, torch.Tensor], nominal: Mapping[str, float]) -> None:
        """Set every continued parameter to its value at Adam *step*: ``start`` -> nominal, linear, then held."""
        with torch.no_grad():
            for c in self.continuation:
                t = min(1.0, step / c.steps)
                params[c.parameter].fill_((1.0 - t) * c.start + t * nominal[c.parameter])

    def _refine(self, provider: ResidualProvider, field_at: Field, points: dict, params: Mapping, round_: int) -> None:
        """Add to each resampled condition the fresh candidates where its residual is largest (round *round_*)."""
        rt = self.runtime
        fresh = provider.collocation(replace(rt, seed=rt.seed + round_))
        candidates = {name: fresh[name].detach().requires_grad_(True) for name in self.resample.conditions}
        terms = provider.residuals(field_at, {**points, **candidates}, params)
        for name in self.resample.conditions:
            if name not in terms:
                refuse(f'resample names {name!r}, which no residual term carries: {sorted(terms)}')
            worst = torch.topk((terms[name].residual / terms[name].scale).detach().abs(), self.resample.add).indices
            points[name] = torch.cat([points[name].detach(), candidates[name].detach()[worst]]).requires_grad_(True)

    def _rebalance(self, terms: Mapping[str, torch.Tensor], theta: list[torch.Tensor], weights: dict) -> None:
        """One learning-rate-annealing update of *weights* from the terms' gradients over the network *theta*."""

        def grads(value: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(value, theta, retain_graph=True, allow_unused=True)
            return torch.cat([
                (torch.zeros_like(p) if d is None else d).abs().flatten() for p, d in zip(theta, g, strict=True)
            ])

        b = self.balance
        if b.rule == 'normalised_gradient':
            means = {name: float(grads(value).mean()) for name, value in terms.items()}
            ref = means[b.reference]
            targets = {n: (ref / m if m > 0.0 else weights[n]) for n, m in means.items()}
            scale = len(targets) / sum(targets.values())
            for name, target in targets.items():
                weights[name] = (1.0 - b.alpha) * weights[name] + b.alpha * target * scale
            return
        if b.rule == 'anchored_gradient':
            ref = float(grads(terms[b.reference]).mean())
            for name, value in terms.items():
                mean = float(grads(value).mean()) if name != b.reference else 0.0
                if mean > 0.0:
                    target = min(b.cap, max(1.0 / b.cap, ref / mean))
                    weights[name] = (1.0 - b.alpha) * weights[name] + b.alpha * target
            return
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
    ansatz: Ansatz | None = None  # the provider's hard ansatz the net's raw columns pass through, if any
    features: Features | None = None  # the provider's net input in place of the normalised coordinates, if any

    def field_at(self, x: torch.Tensor) -> torch.Tensor:
        """The field at ``(n, dim)`` coordinates, differentiable."""
        return _field(self.net, self.runtime, self.domain, self.ansatz, self.features, x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self.domain.require(x, 'a field query')
        with torch.no_grad():
            return numpy(self.field_at(self.runtime.tensor(x)))


def _trained_on(points, data, params) -> list:
    """The arrays a physics fit trains on: the collocation points, the observations and the starting parameters."""
    observed = [array for _, xt, ut, mask, _ in data for array in (xt, ut, mask)]
    return [*(points[name] for name in sorted(points)), *observed, *(params[name] for name in sorted(params))]
