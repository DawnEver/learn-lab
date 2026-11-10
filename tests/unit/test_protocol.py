"""D42 -- the L6 protocol: ``fit -> Fitted``, ``predict -> Prediction(mean, variance | None, gradient | None)``.

Every surrogate declares what it ``gives`` beyond a mean; asking for anything else is refused naming
what the model does give (UNSUPPORTED-RAISES), never answered with a placeholder.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.models.response import NearestNeighbours, Polynomial
from learn_lab.protocol import Give, Prediction, query_points, training_data
from learn_lab.registry import SURROGATES
from learn_lab.validity import BoxEnvelope


def test_a_prediction_holds_one_shape_per_part() -> None:
    p = Prediction(np.zeros((3, 2)), np.ones((3, 2)), np.zeros((3, 2, 4)))
    assert p.mean.shape == (3, 2)
    assert p.variance.shape == (3, 2)
    assert p.gradient.shape == (3, 2, 4)
    assert not p.mean.flags.writeable


@pytest.mark.parametrize(
    ('variance', 'gradient'),
    [(np.ones((3, 1)), None), (None, np.zeros((3, 1, 4))), (-np.ones((3, 2)), None)],
    ids=['variance-columns', 'gradient-outputs', 'negative-variance'],
)
def test_a_prediction_with_a_mismatched_part_is_refused(variance, gradient) -> None:
    with pytest.raises(Refused):
        Prediction(np.zeros((3, 2)), variance, gradient)


def test_training_data_is_two_matrices_with_one_row_per_point() -> None:
    x, y, envelope = training_data(np.zeros((4, 2)), np.zeros((4, 1)))
    assert x.shape == (4, 2)
    assert y.shape == (4, 1)
    assert envelope.width == 2
    for bad_x, bad_y in [
        (np.zeros(4), np.zeros((4, 1))),
        (np.zeros((4, 2)), np.zeros(4)),
        (np.zeros((4, 2)), np.zeros((3, 1))),
    ]:
        with pytest.raises(Refused):
            training_data(bad_x, bad_y)
    with pytest.raises(Refused, match='finite'):
        training_data(np.full((2, 1), np.nan), np.zeros((2, 1)))


def test_query_points_must_have_the_training_width() -> None:
    envelope = BoxEnvelope(np.zeros(3), np.ones(3))
    assert query_points(np.zeros((5, 3)), envelope).shape == (5, 3)
    with pytest.raises(Refused, match='3'):
        query_points(np.zeros((5, 2)), envelope)


def test_the_envelope_is_the_training_span_unless_a_domain_is_declared() -> None:
    x = np.array([[0.2, 0.4], [0.6, 0.9]])
    _, _, span = training_data(x, np.zeros((2, 1)))
    np.testing.assert_array_equal(span.lower, [0.2, 0.4])
    np.testing.assert_array_equal(span.upper, [0.6, 0.9])
    cube = BoxEnvelope(np.zeros(2), np.ones(2))
    assert training_data(x, np.zeros((2, 1)), cube)[2] is cube


def test_a_declared_domain_must_hold_every_training_point() -> None:
    """PLANTED CONTROL: a domain that misses a training row is a declaration that lies."""
    with pytest.raises(Refused, match='out_of_domain'):
        training_data(np.array([[0.5], [1.5]]), np.zeros((2, 1)), BoxEnvelope(np.zeros(1), np.ones(1)))
    with pytest.raises(Refused, match='BoxEnvelope'):
        training_data(np.zeros((2, 1)), np.zeros((2, 1)), (np.zeros(1), np.ones(1)))


def test_a_query_outside_the_envelope_or_not_finite_is_refused() -> None:
    envelope = BoxEnvelope(np.zeros(2), np.ones(2))
    np.testing.assert_array_equal(query_points([[0.0, 1.0]], envelope), [[0.0, 1.0]])
    with pytest.raises(Refused, match='out_of_domain'):
        query_points([[0.5, 1.0 + 1e-12]], envelope)
    with pytest.raises(Refused, match='finite'):
        query_points([[np.nan, 0.5]], envelope)


def test_asking_a_model_for_what_it_does_not_give_is_refused_naming_what_it_gives() -> None:
    x = np.random.default_rng(0).random((12, 2))
    fitted = NearestNeighbours(k=3).fit(x, x[:, :1])
    assert NearestNeighbours(k=3).gives == frozenset()
    with pytest.raises(Refused, match='variance'):
        fitted.predict(x, variance=True)
    with pytest.raises(Refused, match='gradient'):
        fitted.predict(x, gradient=True)
    fitted = Polynomial(degree=2).fit(x, x[:, :1])
    assert fitted.predict(x, gradient=True).gradient.shape == (12, 1, 2)


def test_every_registered_surrogate_declares_what_it_gives() -> None:
    """The registry is the declared set of L6 models; every row names a ``gives`` that is a set of ``Give``."""
    assert {'polynomial', 'radial_basis', 'gaussian_process', 'nearest_neighbours', 'regression_tree', 'forest'} <= set(
        SURROGATES
    )
    for name, row in SURROGATES.items():
        gives = row.build().gives
        assert gives <= set(Give), name
