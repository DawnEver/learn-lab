import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.validity import BoxEnvelope, Validity


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
