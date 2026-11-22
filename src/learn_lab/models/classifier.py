"""Classifiers ``x (n, d) -> label``: fit on labels, predict the class and its probabilities.

A classifier is a frozen declaration of its settings; ``fit(x, labels, domain=)`` returns a fitted classifier
whose ``predict(x)`` returns a :class:`ClassPrediction` -- the probability of every class ``(m, c)`` (each row
a distribution) and the most probable label per point. Labels are any non-object numpy dtype (integers,
strings); the classes are their sorted distinct values and keep that dtype.

* :class:`NearestNeighboursClassifier` -- the (distance-)weighted vote of the k nearest labels; its
  probabilities ARE the response :class:`~learn_lab.models.response.NearestNeighbours` mean on one-hot labels;
* :class:`ClassificationTree` -- a CART tree on the Gini impurity; on one-hot labels the summed squared error a
  regression tree minimises is ``n * Gini``, so it IS the response :class:`~learn_lab.models.response.RegressionTree`
  grown on one-hot labels, and a leaf's mean is its class frequencies;
* :class:`LogisticRegression` -- multinomial (softmax) logistic regression with an l2 penalty *alpha*, on inputs
  scaled to the unit box of the fit's envelope, trained by numerics-lab's ``minimize_lbfgs``.

Every fit carries the ENVELOPE of :func:`learn_lab.protocol.training_data`: a query outside it is refused.
scikit-learn is a TEST oracle only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numerics_lab.minimize import minimize_lbfgs
from scipy.special import log_softmax, softmax

from learn_lab.errors import refuse, require_finite
from learn_lab.models.response.neighbours import NEIGHBOUR_WEIGHTS, FittedNeighbours, NearestNeighbours
from learn_lab.models.response.tree import SPLITTERS, FittedTree, RegressionTree
from learn_lab.protocol import query_points, training_data
from learn_lab.validity import BoxEnvelope

__all__ = [
    'NEIGHBOUR_WEIGHTS',
    'SPLITTERS',
    'ClassPrediction',
    'ClassificationTree',
    'FittedClassificationTree',
    'FittedLogistic',
    'FittedNeighboursClassifier',
    'LogisticRegression',
    'NearestNeighboursClassifier',
    'training_labels',
]

#: A probability row may miss one by this much from rounding (dimensionless).
_SUMS_TO_ONE: Final = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class ClassPrediction:
    """``probabilities (m, c)`` over ``classes (c,)``, each row a distribution; ``labels (m,)`` its argmax."""

    classes: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.classes):
            refuse(f'probabilities are (points, {len(self.classes)} classes), got shape {p.shape}')
        if np.any(p < 0.0) or not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=_SUMS_TO_ONE):
            refuse('each row of class probabilities is a distribution: non-negative, summing to one')
        p.setflags(write=False)
        object.__setattr__(self, 'probabilities', p)

    @property
    def labels(self) -> np.ndarray:
        """The most probable class of each point (the first, on a tie)."""
        return self.classes[np.argmax(self.probabilities, axis=1)]


def training_labels(
    x: object, labels: object, domain: BoxEnvelope | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BoxEnvelope]:
    """``(x (n, d), index (n,), classes (c,), envelope)``: the points, each label's class index, the classes.

    At least two classes, one label per point, and no object dtype (an artifact holds no pickled values).
    """
    labels = np.asarray(labels)
    if labels.dtype.hasobject:
        refuse('labels are integers or strings; an object array cannot be held or compared honestly')
    if labels.ndim != 1:
        refuse(f'labels are one per point, a vector; got shape {labels.shape}')
    if np.shape(x)[:1] != labels.shape:
        refuse(f'a classifier takes one label per point: {labels.shape[0]} label(s) for {np.shape(x)[0]} point(s)')
    classes, index = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        refuse(f'a classifier needs at least two classes to tell apart, got {classes.tolist()}')
    x, _, envelope = training_data(x, np.zeros((len(labels), 1)), domain)
    return x, index, classes, envelope


def _one_hot(index: np.ndarray, c: int) -> np.ndarray:
    return np.eye(c)[index]


@dataclass(frozen=True, slots=True)
class NearestNeighboursClassifier:
    """The vote of the *k* nearest labels, weighted per :data:`NEIGHBOUR_WEIGHTS`."""

    k: int = 5
    weights: str = 'uniform'

    def __post_init__(self) -> None:
        NearestNeighbours(self.k, self.weights)

    def fit(self, x, labels, *, domain: BoxEnvelope | None = None) -> FittedNeighboursClassifier:
        x, index, classes, envelope = training_labels(x, labels, domain)
        votes = NearestNeighbours(self.k, self.weights).fit(x, _one_hot(index, len(classes)), domain=envelope)
        return FittedNeighboursClassifier(classes, votes)


@dataclass(frozen=True, slots=True, eq=False)
class FittedNeighboursClassifier:
    """The classes and the neighbours whose one-hot labels are averaged into probabilities."""

    classes: np.ndarray
    votes: FittedNeighbours

    @property
    def envelope(self) -> BoxEnvelope:
        return self.votes.envelope

    def predict(self, x) -> ClassPrediction:
        return ClassPrediction(self.classes, self.votes.predict(x).mean)


@dataclass(frozen=True, slots=True)
class ClassificationTree:
    """One CART tree on the Gini impurity (see the module docstring); *seed* matters only when draws are made."""

    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float = 1.0
    splitter: str = 'best'
    seed: int = 0

    def __post_init__(self) -> None:
        self._regression()

    def _regression(self) -> RegressionTree:
        return RegressionTree(
            self.max_depth, self.min_samples_split, self.min_samples_leaf, self.max_features, self.splitter, self.seed
        )

    def fit(self, x, labels, *, domain: BoxEnvelope | None = None) -> FittedClassificationTree:
        x, index, classes, envelope = training_labels(x, labels, domain)
        return FittedClassificationTree(
            classes, self._regression().fit(x, _one_hot(index, len(classes)), domain=envelope)
        )


@dataclass(frozen=True, slots=True, eq=False)
class FittedClassificationTree:
    """The classes and the tree whose leaves hold their class frequencies."""

    classes: np.ndarray
    tree: FittedTree

    @property
    def envelope(self) -> BoxEnvelope:
        return self.tree.envelope

    def predict(self, x) -> ClassPrediction:
        return ClassPrediction(self.classes, self.tree.predict(x).mean)


@dataclass(frozen=True, slots=True)
class LogisticRegression:
    """Softmax regression: l2 penalty *alpha* on the weights (not the bias), L-BFGS *max_iterations*."""

    alpha: float = 1e-4
    max_iterations: int = 2000

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0:
            refuse(f'alpha is a finite number >= 0, got {self.alpha!r}')
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            refuse(f'max_iterations is an integer >= 1, got {self.max_iterations!r}')

    def fit(self, x, labels, *, domain: BoxEnvelope | None = None) -> FittedLogistic:
        x, index, classes, envelope = training_labels(x, labels, domain)
        z = _unit(x, envelope)
        n, d = z.shape
        c = len(classes)
        target = _one_hot(index, c)

        def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = flat[: d * c].reshape(d, c), flat[d * c :]
            log_p = log_softmax(z @ w + b, axis=1)
            residual = (np.exp(log_p) - target) / n
            value = -np.sum(target * log_p) / n + 0.5 * self.alpha * np.sum(w**2)
            return float(value), np.concatenate([(z.T @ residual + self.alpha * w).ravel(), residual.sum(axis=0)])

        trained = minimize_lbfgs(loss, np.zeros(d * c + c), rtol=0.0, atol=1e-10, max_iter=self.max_iterations)
        flat = require_finite(np.asarray(trained.x), 'the trained logistic weights')
        return FittedLogistic(classes, envelope, flat[: d * c].reshape(d, c), flat[d * c :])


def _unit(x: np.ndarray, envelope: BoxEnvelope) -> np.ndarray:
    """*x* in the unit box of *envelope*; a flat dimension stays at zero."""
    span = envelope.upper - envelope.lower
    return (x - envelope.lower) / np.where(span > 0, span, 1.0)


@dataclass(frozen=True, slots=True, eq=False)
class FittedLogistic:
    """The classes, the envelope and the softmax weights ``(d, c)`` and bias ``(c,)`` on unit-box inputs."""

    classes: np.ndarray
    envelope: BoxEnvelope
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, x) -> ClassPrediction:
        z = _unit(query_points(x, self.envelope), self.envelope)
        return ClassPrediction(self.classes, softmax(z @ self.weights + self.bias, axis=1))
