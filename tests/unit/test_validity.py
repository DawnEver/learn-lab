import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.validity import BoxEnvelope, HullEnvelope, Validity


def test_the_envelope_classifies_inside_outside_and_invalid():
    env = BoxEnvelope.of(np.array([[0.0, 0.0], [1.0, 2.0]]))
    verdict = env.classify(np.array([[0.5, 1.0], [1.5, 1.0], [np.nan, 1.0]]))
    assert list(verdict) == [Validity.IN_DOMAIN, Validity.OUT_OF_DOMAIN, Validity.INVALID]


def test_a_query_outside_the_envelope_is_refused_and_one_inside_is_not():
    env = BoxEnvelope.of(np.array([[0.0], [1.0]]))
    assert env.require(np.array([[0.25]]), 'q').shape == (1, 1)
    with pytest.raises(Refused, match='out_of_domain'):
        env.require(np.array([[1.01]]), 'q')


def test_a_query_of_the_wrong_width_is_invalid():
    env = BoxEnvelope.of(np.zeros((3, 2)))
    with pytest.raises(Refused, match='invalid'):
        env.require(np.zeros((1, 3)), 'q')


#: A triangle's vertices plus an interior point; its hull is the triangle, its bounding box the unit square.
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.2]])


def test_the_hull_holds_its_own_vertices_and_refuses_the_box_corner_it_does_not_cover():
    env = HullEnvelope.of(TRIANGLE)
    assert env.width == 2
    assert env.rank == 2
    assert list(env.classify(TRIANGLE)) == [Validity.IN_DOMAIN] * len(TRIANGLE)
    assert BoxEnvelope.of(TRIANGLE).classify(np.array([[0.9, 0.9]]))[0] == Validity.IN_DOMAIN
    assert env.classify(np.array([[0.9, 0.9]]))[0] == Validity.OUT_OF_DOMAIN


def test_a_point_just_outside_a_facet_is_refused_and_one_just_inside_is_not():
    """PLANTED CONTROL: one micro-unit across the hypotenuse x + y = 1 flips the verdict."""
    env = HullEnvelope.of(TRIANGLE)
    assert env.require(np.array([[0.5 - 1e-6, 0.5 - 1e-6]]), 'q').shape == (1, 2)
    with pytest.raises(Refused, match='out_of_domain'):
        env.require(np.array([[0.5 + 1e-6, 0.5 + 1e-6]]), 'q')


def test_the_hull_classifies_a_non_finite_or_wrong_width_point_invalid():
    env = HullEnvelope.of(TRIANGLE)
    assert env.classify(np.array([[np.nan, 0.1]]))[0] == Validity.INVALID
    with pytest.raises(Refused, match='invalid'):
        env.require(np.zeros((1, 3)), 'q')


def test_a_degenerate_set_spans_its_affine_hull_and_refuses_off_it():
    """Collinear points in 3-D: the hull is the segment; a point off its line, or past its end, is refused."""
    line = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.0], [0.5, 1.0, 1.0]])
    env = HullEnvelope.of(line)
    assert env.rank == 1
    assert env.classify(np.array([[0.25, 0.5, 1.0]]))[0] == Validity.IN_DOMAIN
    assert env.classify(np.array([[0.25, 0.5, 1.0 + 1e-6]]))[0] == Validity.OUT_OF_DOMAIN
    assert env.classify(np.array([[1.0 + 1e-6, 2.0 + 2e-6, 1.0]]))[0] == Validity.OUT_OF_DOMAIN


def test_a_single_point_is_its_own_hull():
    env = HullEnvelope.of(np.array([[3.0, 4.0]]))
    assert env.rank == 0
    assert env.classify(np.array([[3.0, 4.0], [3.0, 4.0 + 1e-6]])).tolist() == [
        Validity.IN_DOMAIN,
        Validity.OUT_OF_DOMAIN,
    ]
