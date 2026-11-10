import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.errors import Refused

from ._poisson import Poisson
from ._two_slab import EveryColumnEverywhere, TwoSlab, slope, truth

FAST = Schedule(adam_steps=1500, learning_rate=2e-3, lbfgs_steps=300)


def test_a_pinn_solves_the_planted_poisson_problem():
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(Poisson())
    x = np.linspace(0, 1, 101)[:, None]
    assert np.max(np.abs(fitted.evaluate(x)[:, 0] - np.sin(np.pi * x[:, 0]))) < 2e-3  # unit of u


def test_the_inverse_fit_recovers_the_coefficient_the_source_was_built_with():
    x = np.linspace(0.05, 0.95, 19)[:, None]
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(
        Poisson(k_nominal=1.0, k_source=2.0), identify=('k',), observations={'u': (x, np.sin(np.pi * x))}
    )
    assert fitted.parameters['k'] == pytest.approx(2.0, abs=2e-2)  # k is dimensionless here


def test_identifying_an_undeclared_parameter_or_without_observations_is_refused():
    pif = PhysicsInformedField(schedule=Schedule(adam_steps=1))
    with pytest.raises(Refused, match='declares'):
        pif.fit(Poisson(), identify=('mu',), observations={'u': (np.zeros((1, 1)), np.zeros((1, 1)))})
    with pytest.raises(Refused, match='needs observations'):
        pif.fit(Poisson(), identify=('k',))


def test_a_field_query_outside_the_domain_is_refused():
    fitted = PhysicsInformedField(schedule=Schedule(adam_steps=1)).fit(Poisson())
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.evaluate(np.array([[1.2]]))


#: observations on both slabs, two of them next to the interface at 1/2.
SLAB_X = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.48, 0.52, 0.55, 0.65, 0.75, 0.85, 0.95])[:, None]
K_RIGHT = 4.0
K_BAR = 0.05  # relative, k is dimensionless


def _identified_k_right(provider):
    u = truth(SLAB_X, 1.0, K_RIGHT)
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(
        provider, identify=('k_right',), observations={'u': (SLAB_X, np.stack([u, u], axis=1))}
    )
    return fitted.parameters['k_right']


def test_an_observation_pins_only_the_column_its_region_owns():
    assert _identified_k_right(TwoSlab()) == pytest.approx(K_RIGHT, rel=K_BAR, abs=0.0)


def test_planted_control_pinning_every_column_everywhere_misidentifies_the_coefficient():
    assert abs(_identified_k_right(EveryColumnEverywhere()) / K_RIGHT - 1.0) > K_BAR


def test_an_ownership_of_the_wrong_shape_or_a_nonfinite_owned_value_is_refused():
    pif = PhysicsInformedField(schedule=Schedule(adam_steps=1))
    u = np.array([[0.1, np.nan]])
    pif.fit(TwoSlab(), identify=('k_right',), observations={'u': (np.array([[0.1]]), u)})  # unowned NaN is fine
    with pytest.raises(Refused, match='not finite'):
        pif.fit(TwoSlab(), identify=('k_right',), observations={'u': (np.array([[0.1]]), u[:, ::-1])})
    with pytest.raises(Refused, match=r'returns \(1, 2\)'):
        pif.fit(_WrongOwnership(), observations={'u': (np.array([[0.1]]), np.array([[0.1]]))})


class _WrongOwnership(Poisson):
    def owns(self, name, points):
        return np.ones((points.shape[0], 2), bool)


def test_an_undeclared_observable_or_values_of_the_wrong_shape_are_refused():
    pif = PhysicsInformedField(schedule=Schedule(adam_steps=1))
    x = np.array([[0.1], [0.2]])
    with pytest.raises(Refused, match="'B' is not declared"):
        pif.fit(Poisson(), observations={'B': (x, np.ones((2, 1)))})
    with pytest.raises(Refused, match=r"'du' are \(2, 2\) values, got \(2, 1\)"):
        pif.fit(TwoSlab(), observations={'du': (x, np.ones((2, 1)))})
    with pytest.raises(Refused, match='no observable'):
        pif.fit(Poisson(), identify=('k',), observations={})
    with pytest.raises(Refused, match='is zero'):
        pif.fit(Poisson(), observations={'u': (x, np.zeros((2, 1)))})


#: A stiff right slab read by sensors with a 0.5 % gain error (an instrument's honest floor). At k_right = 20 the right
#: slab's u = 1 - b (1 - x), b = 2 / (k_right + 1), sits within 0.5 % of 1: the gain error alone moves the
#: least-squares k_right to 23.8 (+19 %, closed form). Its slope u' = b carries k_right at gain ~1: 19.90 (-0.5 %).
STIFF = 20.0
STIFF_X = np.array([0.55, 0.65, 0.75, 0.85, 0.95])[:, None]
SENSOR_GAIN = 1.005


def _identified_stiff_k_right(observable):
    values = SENSOR_GAIN * {'u': truth, 'du': slope}[observable](STIFF_X, 1.0, STIFF)
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(
        TwoSlab(), identify=('k_right',), observations={observable: (STIFF_X, np.stack([values, values], axis=1))}
    )
    return fitted.parameters['k_right']


def test_a_derived_observable_identifies_a_coefficient_the_field_hides():
    assert _identified_stiff_k_right('du') == pytest.approx(STIFF, rel=K_BAR, abs=0.0)


def test_planted_control_observing_the_field_alone_misses_the_hidden_coefficient():
    assert abs(_identified_stiff_k_right('u') / STIFF - 1.0) > K_BAR
