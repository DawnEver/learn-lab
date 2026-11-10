"""D52 nearest neighbours, D53 CART / forest / extra trees / bagging, against scikit-learn (tests only).

A deterministic model must agree EXACTLY (absolute, target units, to rounding). A randomised
ensemble draws its own stream, so it is compared by held-out error, as a RELATIVE band on sklearn's
RMSE at equal settings; and the deterministic limit of each ensemble (no resampling, every feature)
must collapse to one tree exactly -- the planted structural control.
"""

import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from learn_lab.errors import Refused
from learn_lab.models.response import Forest, NearestNeighbours, RegressionTree
from learn_lab.validity import BoxEnvelope

#: Absolute, target units: the same arithmetic on the same points.
EXACT = 1e-12
#: Relative to sklearn's held-out RMSE at the same settings (dimensionless).
ENSEMBLE_BAND = 0.25


def _cube_of(x) -> BoxEnvelope:
    """The unit cube in the width of *x*: the declared domain of data drawn on [0, 1)."""
    return BoxEnvelope(np.zeros(np.shape(x)[1]), np.ones(np.shape(x)[1]))


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(9)
    x = rng.random((200, 3))
    y = np.column_stack([np.sin(4 * x[:, 0]) + x[:, 1] ** 2 + 0.05 * rng.standard_normal(200), x[:, 2] - x[:, 0]])
    q = rng.random((300, 3))
    truth = np.column_stack([np.sin(4 * q[:, 0]) + q[:, 1] ** 2, q[:, 2] - q[:, 0]])
    return x, y, q, truth


@pytest.mark.parametrize('weights', ['uniform', 'distance'])
@pytest.mark.parametrize('k', [1, 5])
def test_nearest_neighbours_equal_sklearns(data, k, weights) -> None:
    x, y, q, _ = data
    ours = NearestNeighbours(k=k, weights=weights).fit(x, y, domain=_cube_of(x)).predict(q).mean
    theirs = KNeighborsRegressor(n_neighbors=k, weights=weights).fit(x, y).predict(q)
    assert np.max(np.abs(ours - theirs)) <= EXACT


def test_distance_weighting_returns_the_measured_value_at_a_measured_point(data) -> None:
    x, y, _, _ = data
    assert (
        np.max(np.abs(NearestNeighbours(k=5, weights='distance').fit(x, y, domain=_cube_of(x)).predict(x).mean - y))
        <= EXACT
    )


@pytest.mark.parametrize(('depth', 'leaf'), [(3, 1), (6, 4)])
def test_a_regression_tree_equals_sklearns(data, depth, leaf) -> None:
    x, y, q, _ = data
    ours = RegressionTree(max_depth=depth, min_samples_leaf=leaf).fit(x, y, domain=_cube_of(x)).predict(q).mean
    theirs = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(x, y).predict(q)
    assert np.max(np.abs(ours - theirs)) <= EXACT


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_a_fully_grown_tree_partitions_the_data_as_sklearns_does(data, seed) -> None:
    """Grown out, small nodes TIE: two features give the same partition with equal improvement, and
    sklearn breaks the tie by its random feature order (its own trees for seeds 0..4 disagree at 26-40
    of 300 query points). The partition of the measured points is what is determined, and must agree.
    """
    x, y, _, _ = data
    ours = RegressionTree(min_samples_leaf=2).fit(x, y, domain=_cube_of(x))
    theirs = DecisionTreeRegressor(min_samples_leaf=2, random_state=seed).fit(x, y)
    assert len(ours.feature) == theirs.tree_.node_count
    assert np.max(np.abs(ours.predict(x).mean - theirs.predict(x))) <= EXACT


def test_an_ensemble_without_resampling_is_one_tree(data) -> None:
    x, y, q, _ = data
    tree = RegressionTree(max_depth=5).fit(x, y, domain=_cube_of(x)).predict(q).mean
    forest = (
        Forest(n_trees=7, bootstrap=False, max_features=1.0, max_depth=5).fit(x, y, domain=_cube_of(x)).predict(q).mean
    )
    assert np.max(np.abs(forest - tree)) <= EXACT


def _rmse(prediction, truth):
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


@pytest.mark.parametrize(
    ('ours', 'theirs'),
    [
        (Forest(n_trees=100, seed=0), RandomForestRegressor(n_estimators=100, random_state=0)),
        (
            Forest(n_trees=100, bootstrap=False, splitter='random', seed=0),
            ExtraTreesRegressor(n_estimators=100, random_state=0),
        ),
        (Forest(n_trees=100, max_features=1.0, seed=0), BaggingRegressor(n_estimators=100, random_state=0)),
        (
            Forest(n_trees=100, max_features=0.5, seed=0),
            RandomForestRegressor(n_estimators=100, max_features=0.5, random_state=0),
        ),
    ],
    ids=['random-forest', 'extra-trees', 'bagging', 'feature-subsampled-forest'],
)
def test_each_randomised_ensemble_is_as_accurate_as_sklearns(data, ours, theirs) -> None:
    x, y, q, truth = data
    mine = _rmse(ours.fit(x, y, domain=_cube_of(x)).predict(q).mean, truth)
    reference = _rmse(theirs.fit(x, y).predict(q), truth)
    assert mine <= (1 + ENSEMBLE_BAND) * reference


def test_the_same_seed_is_the_same_forest_and_another_is_not(data) -> None:
    x, y, q, _ = data
    a = Forest(n_trees=10, seed=1).fit(x, y, domain=_cube_of(x)).predict(q).mean
    b = Forest(n_trees=10, seed=1).fit(x, y, domain=_cube_of(x)).predict(q).mean
    c = Forest(n_trees=10, seed=2).fit(x, y, domain=_cube_of(x)).predict(q).mean
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bad_settings_are_refused() -> None:
    with pytest.raises(Refused):
        NearestNeighbours(k=0)
    with pytest.raises(Refused, match='distance'):
        NearestNeighbours(weights='cosine')
    with pytest.raises(Refused):
        RegressionTree(min_samples_leaf=0)
    with pytest.raises(Refused, match='random'):
        Forest(splitter='greedy')
    with pytest.raises(Refused):
        Forest(max_features=1.5)
    with pytest.raises(Refused, match='neighbours'):
        NearestNeighbours(k=10).fit(np.zeros((3, 1)), np.zeros((3, 1)), domain=_cube_of(np.zeros((3, 1))))


def test_uniform_weighting_does_not_return_the_measured_value(data) -> None:
    """Planted control for the distance rule: without it the k nearest are averaged, measured point or not."""
    x, y, _, _ = data
    assert np.max(np.abs(NearestNeighbours(k=5).fit(x, y, domain=_cube_of(x)).predict(x).mean - y)) > 1e-3
