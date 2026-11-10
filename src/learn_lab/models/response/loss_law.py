"""A loss scaling law as the mean and a learned residual on top: ``P_k = law_k(x) + residual_k(x)``.

Each output column is one loss TERM with the law ``P = exp(c) f^a D^b``: the iron Bertotti sub-terms
(``hysteresis`` a=1 b=2, ``classical`` a=2 b=2, ``excess`` a=1.5 b=1.5; D a flux-linkage amplitude) and the
``magnet_eddy`` loss (a=2 b=2; D a current amplitude). The drivers are read off the input columns by NAME:

* frequency -- a ``frequency`` column, or a ``speed`` column (times a ``pole_pairs`` column when declared);
* flux proxy -- a ``flux`` column, or ``hypot(psi_d, psi_q)``;
* current -- a ``current`` column, or ``hypot(i_d, i_q)``.

Any other column name is a plain feature the residual sees and the law ignores. A driver's unit constant
(rpm -> Hz, the pole-pair factor) is absorbed in the level ``c``, since the law is a ratio to its anchor.

The anchor is EVERY training row: ``c`` is the least-squares level in log space, ``mean(log y - a log f -
b log D)`` -- the anchor-row ratio law when the data lie on it, and no single row's noise when they do not.
``fit_exponents`` frees ``a, b`` to the same log-space least squares. A training loss negligible against its
column (below a relative floor of 1e-9) cannot anchor a ratio and is refused, never answered.

The *base* surrogate (a Gaussian process by default) is fitted on ``y - law`` with its inputs mapped onto the
unit box of the fit's envelope, so no driver's unit (rpm against Wb) sets its length scales; its variance is the
prediction's, and the gradient is the law's exact one plus the residual's (chained through that map).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numerics_lab.linear.dense import least_squares

from learn_lab.errors import refuse
from learn_lab.models.response.gaussian_process import GaussianProcess
from learn_lab.protocol import Give, Prediction, Surrogate, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['TEXTBOOK_EXPONENTS', 'FittedLossLawResidual', 'LossLawResidual']

#: ``(frequency exponent, driver exponent)`` per loss term.
TEXTBOOK_EXPONENTS: Final = {
    'hysteresis': (1.0, 2.0),
    'classical': (2.0, 2.0),
    'excess': (1.5, 1.5),
    'magnet_eddy': (2.0, 2.0),
}
#: The amplitude each term scales with: flux linkage for iron, current for the magnet.
_DRIVER: Final = {'hysteresis': 'flux', 'classical': 'flux', 'excess': 'flux', 'magnet_eddy': 'current'}
#: Each driver: its direct column, else its components (a product for frequency, a hypot otherwise).
_SOURCES: Final = {
    'frequency': ('frequency', ('speed',), ('pole_pairs',)),
    'flux': ('flux', ('psi_d', 'psi_q'), ()),
    'current': ('current', ('i_d', 'i_q'), ()),
}
#: A training loss below this fraction of its column's largest is negligible and refused as an anchor.
NEGLIGIBLE: Final = 1e-9


def _columns(inputs: tuple[str, ...], driver: str) -> tuple[str, tuple[int, ...]]:
    """How *driver* is read: ``('direct'|'product'|'hypot', column indices)``, or a refusal naming the gap."""
    direct, parts, optional = _SOURCES[driver]
    present = [p for p in parts if p in inputs]
    if direct in inputs and present:
        refuse(f'the {driver} driver is declared once: {direct!r} or {list(parts)}, got both')
    if direct in inputs:
        return 'direct', (inputs.index(direct),)
    if len(present) != len(parts):
        missing = [p for p in parts if p not in inputs]
        refuse(f'the {driver} driver needs a {direct!r} column or {list(parts)}; missing {missing}')
    found = [*parts, *(o for o in optional if o in inputs)]
    return ('product' if driver == 'frequency' else 'hypot'), tuple(inputs.index(p) for p in found)


def _driver(x: np.ndarray, how: str, cols: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The driver ``(n,)`` and its log-gradient ``d log D / d x`` as ``(n, d)``."""
    slope = np.zeros_like(x)
    if how == 'hypot':
        value = np.hypot(*(x[:, c] for c in cols))
        square = value**2
        for c in cols:
            slope[:, c] = np.divide(x[:, c], square, out=np.zeros_like(value), where=square > 0)
        return value, slope
    value = np.prod(x[:, list(cols)], axis=1)
    for c in cols:
        slope[:, c] = np.divide(1.0, x[:, c], out=np.zeros_like(value), where=x[:, c] != 0)
    return value, slope


def _span(envelope: BoxEnvelope) -> np.ndarray:
    """The envelope's width per input; a constant input (zero width) keeps unit scale."""
    span = envelope.upper - envelope.lower
    return np.where(span > 0, span, 1.0)


def _unit(x: np.ndarray, envelope: BoxEnvelope) -> np.ndarray:
    """*x* mapped onto the unit box of *envelope*: the residual's inputs, so no driver's unit sets its length scale."""
    return (x - envelope.lower) / _span(envelope)


@dataclass(frozen=True, slots=True)
class LossLawResidual:
    """Loss *terms* (one per output) on named *inputs*; *exponents* default to textbook; *base* fits the residual."""

    inputs: tuple[str, ...] = ('frequency', 'flux')
    terms: tuple[str, ...] = ('classical',)
    exponents: tuple[tuple[float, float], ...] | None = None
    fit_exponents: bool = False
    base: Surrogate = field(default_factory=GaussianProcess)

    def __post_init__(self) -> None:
        inputs, terms = tuple(self.inputs), tuple(self.terms)
        if len(set(inputs)) != len(inputs) or not all(isinstance(n, str) and n for n in inputs):
            refuse(f'each input column is named once by a non-empty string, got {inputs}')
        if not terms or any(t not in TEXTBOOK_EXPONENTS for t in terms):
            refuse(f'a loss term is one of {list(TEXTBOOK_EXPONENTS)}, got {terms}')
        for driver in {'frequency', *(_DRIVER[t] for t in terms)}:
            _columns(inputs, driver)
        exponents = tuple(TEXTBOOK_EXPONENTS[t] for t in terms) if self.exponents is None else self.exponents
        exponents = tuple((float(a), float(b)) for a, b in exponents)
        if len(exponents) != len(terms) or not np.all(np.isfinite(exponents)):
            refuse(f'exponents are one finite (a, b) per term {terms}, got {self.exponents!r}')
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'terms', terms)
        object.__setattr__(self, 'exponents', exponents)

    @property
    def gives(self) -> frozenset[Give]:
        return self.base.gives

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> FittedLossLawResidual:
        x, y, envelope = training_data(x, y, domain)
        n, d = x.shape
        if d != len(self.inputs) or y.shape[1] != len(self.terms):
            refuse(
                f'the law reads ({len(self.inputs)}) named inputs into {len(self.terms)} terms; got {x.shape}, {y.shape}'
            )
        floor = NEGLIGIBLE * np.max(np.abs(y), axis=0)
        bad = np.argwhere(~(y > floor))
        if bad.size:
            row, k = (int(i) for i in bad[0])
            refuse(f'a {self.terms[k]} loss of {y[row, k]!r} W at row {row} is negligible: it cannot anchor a ratio')
        frequency, _ = _driver(x, *_columns(self.inputs, 'frequency'))
        exponents, level = np.empty((len(self.terms), 2)), np.empty(len(self.terms))
        for k, term in enumerate(self.terms):
            amplitude, _ = _driver(x, *_columns(self.inputs, _DRIVER[term]))
            if np.any(frequency <= 0) or np.any(amplitude <= 0):
                refuse(f'the {term} law is anchored on strictly positive frequency and {_DRIVER[term]} drivers')
            logs = np.column_stack([np.log(frequency), np.log(amplitude)])
            if self.fit_exponents:
                solution = least_squares(np.column_stack([np.ones(n), logs]), np.log(y[:, k : k + 1]))[:, 0]
                level[k], exponents[k] = solution[0], solution[1:]
            else:
                exponents[k] = self.exponents[k]
                level[k] = np.mean(np.log(y[:, k]) - logs @ exponents[k])
        law = FittedLossLawResidual(self.inputs, self.terms, exponents, level, None, envelope)
        unit = BoxEnvelope(np.zeros(d), np.ones(d))
        residual = self.base.fit(_unit(x, envelope), y - law.law(x), domain=unit)
        return FittedLossLawResidual(self.inputs, self.terms, exponents, level, residual, envelope)


@dataclass(frozen=True, slots=True, eq=False)
class FittedLossLawResidual:
    """``law(x) + residual(x)``; ``exponents`` is ``(terms, 2)`` = (frequency, driver), ``level`` is ``(terms,)``."""

    inputs: tuple[str, ...]
    terms: tuple[str, ...]
    exponents: np.ndarray
    level: np.ndarray
    residual: object
    envelope: BoxEnvelope

    @property
    def gives(self) -> frozenset[Give]:
        return self.residual.gives

    def _law(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frequency, df = _driver(x, *_columns(self.inputs, 'frequency'))
        means, slopes = [], []
        for k, term in enumerate(self.terms):
            amplitude, da = _driver(x, *_columns(self.inputs, _DRIVER[term]))
            a, b = self.exponents[k]
            mean = np.exp(self.level[k]) * frequency**a * amplitude**b
            means.append(mean)
            slopes.append(mean[:, None] * (a * df + b * da))
        return np.column_stack(means), np.stack(slopes, axis=1)

    def law(self, x) -> np.ndarray:
        """The law alone ``(m, terms)`` in watts: the mean without its learned residual."""
        return self._law(query_points(x, self.envelope))[0]

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a loss law with a residual')
        x = query_points(x, self.envelope)
        mean, slope = self._law(x)
        residual = self.residual.predict(_unit(x, self.envelope), variance=variance, gradient=gradient)
        return Prediction(
            mean + residual.mean,
            residual.variance,
            None if residual.gradient is None else slope + residual.gradient / _span(self.envelope),
        )
