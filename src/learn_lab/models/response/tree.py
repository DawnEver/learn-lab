"""D53 -- regression trees (CART) and their ensembles: random forest, extra trees, bagging.

A tree splits a node on the feature and threshold that most reduce the summed squared error of all
outputs, and predicts a leaf's mean. A node stays a leaf when it is at ``max_depth``, has fewer than
``min_samples_split`` points, cannot leave ``min_samples_leaf`` on each side, or is already pure.

* ``splitter='best'`` -- every distinct midpoint of every candidate feature (CART);
* ``splitter='random'`` -- one uniform threshold per candidate feature (extremely randomised trees).

``max_features`` is the fraction of the features drawn (without replacement) as candidates at each
node. :class:`Forest` averages ``n_trees`` trees, each on a bootstrap resample when ``bootstrap``:
a random forest is ``Forest()``, bagging ``Forest(max_features=1.0)`` (the same thing for
regression), extra trees ``Forest(bootstrap=False, splitter='random')``. All randomness flows from
``seed``. A piecewise-constant surface gives no gradient and no variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data

__all__ = ['SPLITTERS', 'FittedForest', 'FittedTree', 'Forest', 'RegressionTree']

#: The declared split rules.
SPLITTERS: Final = ('best', 'random')
#: Two feature values closer than this are one value (no split between them), in input units.
_SAME: Final = 1e-7
#: A node whose mean squared deviation is at or below this is pure (values' squared units).
_PURE: Final = float(np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
class _Rules:
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    max_features: float
    splitter: str

    def __post_init__(self) -> None:
        if self.max_depth is not None and self.max_depth < 1:
            refuse(f'max_depth is None or >= 1, got {self.max_depth!r}')
        if self.min_samples_split < 2 or self.min_samples_leaf < 1:
            refuse(
                f'min_samples_split >= 2 and min_samples_leaf >= 1, got {self.min_samples_split}, {self.min_samples_leaf}'
            )
        if not 0 < self.max_features <= 1:
            refuse(f'max_features is a fraction of the features in (0, 1], got {self.max_features!r}')
        if self.splitter not in SPLITTERS:
            refuse(f'a splitter is one of {list(SPLITTERS)}, got {self.splitter!r}')


def _proxy(left_sum: np.ndarray, n_left: np.ndarray, total: np.ndarray, n: int) -> np.ndarray:
    """``sum_k S_left^2 / n_left + S_right^2 / n_right``: larger is a smaller summed squared error."""
    return np.sum(left_sum**2, axis=-1) / n_left + np.sum((total - left_sum) ** 2, axis=-1) / (n - n_left)


def _best(xs: np.ndarray, ys: np.ndarray, leaf: int) -> tuple[float, float] | None:
    order = np.argsort(xs, kind='stable')
    xs, ys = xs[order], ys[order]
    n = len(xs)
    sizes = np.arange(1, n)
    valid = (xs[1:] > xs[:-1] + _SAME) & (sizes >= leaf) & (n - sizes >= leaf)
    if not valid.any():
        return None
    sums = np.cumsum(ys, axis=0)[:-1]
    score = np.where(valid, _proxy(sums, sizes, sums[-1] + ys[-1], n), -np.inf)
    i = int(np.argmax(score))
    threshold = xs[i] / 2 + xs[i + 1] / 2
    return float(score[i]), float(xs[i] if threshold == xs[i + 1] else threshold)


def _random(xs: np.ndarray, ys: np.ndarray, leaf: int, rng: np.random.Generator) -> tuple[float, float] | None:
    lo, hi = xs.min(), xs.max()
    if hi <= lo + _SAME:
        return None
    threshold = float(rng.uniform(lo, hi))
    left = xs <= threshold
    n_left = int(left.sum())
    if n_left < leaf or len(xs) - n_left < leaf:
        return None
    return float(_proxy(ys[left].sum(axis=0), n_left, ys.sum(axis=0), len(xs))), threshold


def _grow(x: np.ndarray, y: np.ndarray, rows: np.ndarray, rules: _Rules, rng: np.random.Generator) -> FittedTree:
    feature, threshold, left, right, value = [], [], [], [], []

    def node(members: np.ndarray) -> int:
        feature.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(y[members].mean(axis=0))
        return len(value) - 1

    d = x.shape[1]
    candidates = max(1, int(rules.max_features * d))
    stack = [(node(rows), rows, 0)]
    while stack:
        at, members, depth = stack.pop()
        n = len(members)
        if (rules.max_depth is not None and depth >= rules.max_depth) or n < rules.min_samples_split:
            continue
        if n < 2 * rules.min_samples_leaf or np.mean(np.var(y[members], axis=0)) <= _PURE:
            continue
        features = np.arange(d) if candidates == d and rules.splitter == 'best' else rng.permutation(d)[:candidates]
        best = None
        for f in features:
            xs, ys = x[members, f], y[members]
            found = (
                _best(xs, ys, rules.min_samples_leaf)
                if rules.splitter == 'best'
                else _random(xs, ys, rules.min_samples_leaf, rng)
            )
            if found is not None and (best is None or found[0] > best[0]):
                best = (found[0], int(f), found[1])
        if best is None:
            continue
        _, f, t = best
        goes_left = x[members, f] <= t
        feature[at], threshold[at] = f, t
        left[at], right[at] = node(members[goes_left]), node(members[~goes_left])
        stack += [(left[at], members[goes_left], depth + 1), (right[at], members[~goes_left], depth + 1)]
    return FittedTree(np.array(feature), np.array(threshold), np.array(left), np.array(right), np.array(value), d)


@dataclass(frozen=True, slots=True, eq=False)
class FittedTree:
    """The nodes as parallel arrays; a leaf has ``feature == -1`` and predicts ``value``."""

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    width: int

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def leaves(self, q: np.ndarray) -> np.ndarray:
        """The leaf each query point falls in."""
        at = np.zeros(len(q), dtype=int)
        inner = self.feature[at] >= 0
        while inner.any():
            idx = np.flatnonzero(inner)
            f = self.feature[at[idx]]
            at[idx] = np.where(q[idx, f] <= self.threshold[at[idx]], self.left[at[idx]], self.right[at[idx]])
            inner = self.feature[at] >= 0
        return at

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a regression tree')
        return Prediction(self.value[self.leaves(query_points(x, self.width))])


@dataclass(frozen=True, slots=True)
class RegressionTree:
    """One CART tree (see the module docstring); *seed* matters only when features or thresholds are drawn."""

    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float = 1.0
    splitter: str = 'best'
    seed: int = 0

    def __post_init__(self) -> None:
        _Rules(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.max_features, self.splitter)

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def fit(self, x, y) -> FittedTree:
        x, y = training_data(x, y)
        rules = _Rules(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.max_features, self.splitter)
        return _grow(x, y, np.arange(len(x)), rules, np.random.default_rng(self.seed))


@dataclass(frozen=True, slots=True)
class Forest:
    """The mean of *n_trees* trees, each grown on a bootstrap resample when *bootstrap* (see the module docstring)."""

    n_trees: int = 100
    bootstrap: bool = True
    max_features: float = 1.0
    splitter: str = 'best'
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trees < 1:
            refuse(f'a forest has n_trees >= 1, got {self.n_trees!r}')
        _Rules(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.max_features, self.splitter)

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def fit(self, x, y) -> FittedForest:
        x, y = training_data(x, y)
        rules = _Rules(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.max_features, self.splitter)
        rng = np.random.default_rng(self.seed)
        n = len(x)
        trees = tuple(
            _grow(x, y, rng.integers(0, n, n) if self.bootstrap else np.arange(n), rules, rng)
            for _ in range(self.n_trees)
        )
        return FittedForest(trees)


@dataclass(frozen=True, slots=True, eq=False)
class FittedForest:
    """The fitted trees, averaged."""

    trees: tuple[FittedTree, ...]

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset()

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a forest')
        return Prediction(np.mean([tree.predict(x).mean for tree in self.trees], axis=0))
