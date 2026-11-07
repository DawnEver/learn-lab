import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.physics import PhysicsInformedField
from learn_lab.errors import Refused

from ._poisson import Poisson

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
