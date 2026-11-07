import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.protocol import Give, Prediction, require_gives, training_data


def test_a_prediction_refuses_a_negative_variance_and_a_misshapen_gradient():
    with pytest.raises(Refused, match='variance is >= 0'):
        Prediction(np.zeros((2, 1)), variance=-np.ones((2, 1)))
    with pytest.raises(Refused, match='gradient'):
        Prediction(np.zeros((2, 1)), gradient=np.zeros((2, 2)))


def test_asking_for_what_a_model_does_not_give_is_refused_naming_what_it_gives():
    with pytest.raises(Refused, match='gives no variance'):
        require_gives(frozenset({Give.GRADIENT}), variance=True, gradient=False, what='m')
    require_gives(frozenset({Give.GRADIENT}), variance=False, gradient=True, what='m')


def test_training_data_needs_one_value_row_per_point():
    with pytest.raises(Refused, match='one row per point'):
        training_data(np.zeros((3, 2)), np.zeros((2, 1)))
