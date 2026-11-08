"""The deep families by name, each beside the :class:`~learn_lab.space.Space` it tunes by default.

Every row is shaped (:class:`~learn_lab.space.Family`): its build takes ``shapes=(x_shape, y_shape)`` -- the
per-sample shapes of sample-indexed arrays -- and an optional :class:`~learn_lab.deep.fit.Checkpoint`, and
returns a surrogate on those arrays flattened to rows, so :func:`~learn_lab.validation.cross_validate` folds
it along the samples. The n-th fit of one build checkpoints under ``member-<n>``. Common settings: ``steps``
(Adam) and ``learning_rate``; a network of uniform layers is ``width`` x ``depth``. Data a family needs
beyond ``(x, y)`` is a fixed setting: DeepONet's shared query ``points``, a graph's ``edges`` (one topology
for every sample), a neural ODE's sample times ``t`` (``x`` and ``y`` both the trajectories; a trajectory is
predicted from its start). The physics-informed field fits a residual, not data, so it has no data objective.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from learn_lab.deep.constitutive import MonotoneLaw
from learn_lab.deep.dynamics import NeuralODE
from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Checkpoint, Schedule
from learn_lab.deep.graph import GraphModel, Graphs
from learn_lab.deep.hysteresis import PlayHysteresis
from learn_lab.deep.operator import FNO, DeepONet
from learn_lab.deep.surrogate import DeepEnsemble
from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction
from learn_lab.space import Family, Integer, Real, Space
from learn_lab.validity import BoxEnvelope

__all__ = ['DEEP']

Shapes = tuple[tuple[int, ...], tuple[int, ...]]
Predict = Callable[[np.ndarray], np.ndarray]
#: ``(x, y, checkpoint, domain) -> predict``: a fit on sample-indexed arrays, and its prediction on such arrays;
#: *domain* is the rows' envelope, for the one family that is fitted in a box (the deep ensemble).
Fit = Callable[[np.ndarray, np.ndarray, Checkpoint | None, BoxEnvelope], Predict]


@dataclass(frozen=True, slots=True)
class _Predictor:
    predict_samples: Predict
    x_shape: tuple[int, ...]
    envelope: BoxEnvelope
    gives: frozenset[Give] = frozenset()

    def predict(self, x: np.ndarray, *, variance: bool = False, gradient: bool = False) -> Prediction:
        if variance or gradient:
            refuse('a tuned deep fit gives the mean only')
        out = np.asarray(self.predict_samples(np.asarray(x, float).reshape(len(x), *self.x_shape)), float)
        return Prediction(out.reshape(len(x), -1))


@dataclass(frozen=True, slots=True)
class _Rows:
    """A deep fit as a surrogate on flattened rows; the n-th fit checkpoints under ``member-<n>``."""

    fit_samples: Fit
    shapes: Shapes
    checkpoint: Checkpoint | None
    _fits: list[int] = field(default_factory=list)
    gives: frozenset[Give] = frozenset()

    def fit(self, x: np.ndarray, y: np.ndarray, *, domain: BoxEnvelope | None = None) -> _Predictor:
        x_shape, y_shape = self.shapes
        checkpoint = None if self.checkpoint is None else self.checkpoint.member(len(self._fits))
        self._fits.append(len(x))
        domain = domain if domain is not None else BoxEnvelope.of(x)
        predict = self.fit_samples(x.reshape(len(x), *x_shape), y.reshape(len(y), *y_shape), checkpoint, domain)
        return _Predictor(predict, x_shape, domain)


_STEPS: Final = Integer(1, 20_000)
_RATE: Final = Real(1e-5, 1.0, log=True)
_WIDTH: Final = Integer(2, 512)
_DEPTH: Final = Integer(1, 8)


def _schedule(steps: int, learning_rate: float, polish: int = 0) -> Schedule:
    return Schedule(adam_steps=steps, learning_rate=learning_rate, lbfgs_steps=polish)


def _deep_ensemble(*, shapes, checkpoint, width=64, depth=2, steps=3000, learning_rate=3e-3, **declared):
    model = DeepEnsemble(hidden=(width,) * depth, schedule=_schedule(steps, learning_rate), **declared)

    def fit(x, y, c, domain):
        fitted = model.fit(x, y, domain=domain, checkpoint=c)
        return lambda q: fitted.predict(q).mean

    return _Rows(fit, shapes, checkpoint)


def _unet(*, shapes, checkpoint, width=16, levels=2, steps=3000, learning_rate=2e-3, **declared):
    model = FieldModel(width=width, levels=levels, schedule=_schedule(steps, learning_rate), **declared)
    return _Rows(lambda x, y, c, _domain: model.fit(x, y, checkpoint=c).predict, shapes, checkpoint)


def _deeponet(*, shapes, checkpoint, points, width=64, depth=2, basis=32, steps=3000, learning_rate=2e-3, **declared):
    model = DeepONet(hidden=(width,) * depth, basis=basis, schedule=_schedule(steps, learning_rate), **declared)

    def fit(x, y, c, _domain):
        fitted = model.fit(x, points, y, checkpoint=c)
        return lambda sensors: fitted.predict(sensors, points)

    return _Rows(fit, shapes, checkpoint)


def _fno(*, shapes, checkpoint, modes=12, width=32, layers=4, steps=2000, learning_rate=3e-3, **declared):
    grid = len(shapes[0]) - 1  # (channels, *grid)
    schedule = _schedule(steps, learning_rate)
    model = FNO(modes=(modes,) * grid, width=width, layers=layers, schedule=schedule, **declared)
    return _Rows(lambda x, y, c, _domain: model.fit(x, y, checkpoint=c).predict, shapes, checkpoint)


def _gnn(
    *, shapes, checkpoint, edges, edge_features=None, hidden=64, rounds=6, steps=3000, learning_rate=2e-3, **declared
):
    model = GraphModel(hidden=hidden, rounds=rounds, schedule=_schedule(steps, learning_rate), **declared)

    def batch(nodes: np.ndarray) -> Graphs:
        return Graphs.batch([(n, edges, edge_features) for n in nodes])

    def fit(x, y, c, _domain):
        fitted = model.fit(batch(x), y.reshape(-1, y.shape[-1]), checkpoint=c)
        return lambda nodes: fitted.predict(batch(nodes)).reshape(len(nodes), -1)

    return _Rows(fit, shapes, checkpoint)


def _neural_ode(*, shapes, checkpoint, t, width=64, depth=2, horizon=10, steps=2000, learning_rate=3e-3, **declared):
    model = NeuralODE(hidden=(width,) * depth, horizon=horizon, schedule=_schedule(steps, learning_rate), **declared)

    def fit(_paths, y, c, _domain):
        fitted = model.fit(t, y, checkpoint=c)
        return lambda paths: fitted.integrate(paths[:, 0], t)

    return _Rows(fit, shapes, checkpoint)


def _monotone_law(*, shapes, checkpoint, hidden=16, steps=3000, learning_rate=1e-2, polish=200, **declared):
    model = MonotoneLaw(hidden=hidden, schedule=_schedule(steps, learning_rate, polish), **declared)
    return _Rows(lambda x, y, c, _domain: model.fit(x, y, checkpoint=c).value, shapes, checkpoint)


def _play_hysteresis(*, shapes, checkpoint, plays=16, hidden=16, steps=1500, learning_rate=1e-2, **declared):
    model = PlayHysteresis(plays=plays, hidden=hidden, schedule=_schedule(steps, learning_rate), **declared)

    def fit(x, y, c, _domain):
        fitted = model.fit(list(x), list(y), checkpoint=c)
        return lambda histories: np.stack([fitted.respond(h) for h in histories])

    return _Rows(fit, shapes, checkpoint)


def _space(**dimensions: Integer | Real) -> Space:
    return Space({**dimensions, 'steps': _STEPS, 'learning_rate': _RATE})


#: Every deep family: its shaped build and the settings it tunes.
DEEP: Final = {
    'deep_ensemble': Family(_deep_ensemble, _space(width=_WIDTH, depth=_DEPTH, members=Integer(1, 16)), shaped=True),
    'unet': Family(_unet, _space(width=Integer(2, 128), levels=Integer(1, 5)), shaped=True),
    'deeponet': Family(_deeponet, _space(width=_WIDTH, depth=_DEPTH, basis=Integer(2, 256)), shaped=True),
    'fno': Family(_fno, _space(modes=Integer(1, 64), width=Integer(2, 256), layers=_DEPTH), shaped=True),
    'gnn': Family(_gnn, _space(hidden=_WIDTH, rounds=Integer(1, 16)), shaped=True),
    'neural_ode': Family(_neural_ode, _space(width=_WIDTH, depth=_DEPTH, horizon=Integer(1, 100)), shaped=True),
    'monotone_law': Family(_monotone_law, _space(hidden=Integer(1, 256), polish=Integer(0, 2000)), shaped=True),
    'play_hysteresis': Family(_play_hysteresis, _space(plays=Integer(1, 128), hidden=Integer(1, 256)), shaped=True),
}
