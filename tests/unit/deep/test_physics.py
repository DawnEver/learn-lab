import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.errors import Refused

from ._poisson import Poisson
from ._two_slab import EveryColumnEverywhere, TwoSlab, truth

FAST = Schedule(adam_steps=1500, learning_rate=2e-3, lbfgs_steps=300)


def test_a_pinn_solves_the_planted_poisson_problem():
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(Poisson())
    x = np.linspace(0, 1, 101)[:, None]
    assert np.max(np.abs(fitted.evaluate(x)[:, 0] - np.sin(np.pi * x[:, 0]))) < 2e-3  # unit of u


def test_the_inverse_fit_recovers_the_coefficient_the_source_was_built_with():
    x = np.linspace(0.05, 0.95, 19)[:, None]
    fitted = PhysicsInformedField(hidden=(24, 24), schedule=FAST).fit(
        Poisson(k_nominal=1.0, k_source=2.0), identify=('k',), observations=(x, np.sin(np.pi * x))
    )
    assert fitted.parameters['k'] == pytest.approx(2.0, abs=2e-2)  # k is dimensionless here


def test_identifying_an_undeclared_parameter_or_without_observations_is_refused():
    pif = PhysicsInformedField(schedule=Schedule(adam_steps=1))
    with pytest.raises(Refused, match='declares'):
        pif.fit(Poisson(), identify=('mu',), observations=(np.zeros((1, 1)), np.zeros((1, 1))))
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
        provider, identify=('k_right',), observations=(SLAB_X, np.stack([u, u], axis=1))
    )
    return fitted.parameters['k_right']


def test_an_observation_pins_only_the_column_its_region_owns():
    assert _identified_k_right(TwoSlab()) == pytest.approx(K_RIGHT, rel=K_BAR, abs=0.0)


def test_planted_control_pinning_every_column_everywhere_misidentifies_the_coefficient():
    assert abs(_identified_k_right(EveryColumnEverywhere()) / K_RIGHT - 1.0) > K_BAR


def test_an_ownership_of_the_wrong_shape_or_a_nonfinite_owned_value_is_refused():
    pif = PhysicsInformedField(schedule=Schedule(adam_steps=1))
    u = np.array([[0.1, np.nan]])
    pif.fit(TwoSlab(), identify=('k_right',), observations=(np.array([[0.1]]), u))  # unowned NaN is fine
    with pytest.raises(Refused, match='not finite'):
        pif.fit(TwoSlab(), identify=('k_right',), observations=(np.array([[0.1]]), u[:, ::-1]))
