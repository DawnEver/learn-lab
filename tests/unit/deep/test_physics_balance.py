"""The opt-in term balance: a planted stiff problem the fixed weights under-fit and learning-rate annealing does not."""

from dataclasses import dataclass

import numpy as np
import pytest

from learn_lab.deep.fit import Checkpoint, Schedule
from learn_lab.deep.physics import Balance, PhysicsInformedField, Term
from learn_lab.errors import Refused

from ._poisson import Poisson

SHORT = Schedule(adam_steps=300, learning_rate=2e-3)
ANNEAL = Balance(rule='gradient_norm', reference='pde', alpha=0.1, every=10)
EDGES = np.array([[0.0], [1.0]])


@dataclass(frozen=True)
class Stiff(Poisson):
    """Poisson with its interior term over-weighted ``stiffness`` times: gradients ~ stiffness^2 above the boundary's."""

    stiffness: float = 100.0

    def residuals(self, field, points, parameters):
        terms = dict(super().residuals(field, points, parameters))
        pde = terms['pde']
        terms['pde'] = Term(pde.residual, scale=pde.scale / self.stiffness)
        return terms


def _boundary_error(fitted) -> float:
    """The weak condition's residual, in u's unit: |u| at the two ends where u = 0."""
    return float(np.abs(fitted.evaluate(EDGES)).max())


def test_the_gradient_norm_balance_fits_the_boundary_the_fixed_one_underfits():
    fixed = PhysicsInformedField(hidden=(16, 16), schedule=SHORT).fit(Stiff())
    balanced = PhysicsInformedField(hidden=(16, 16), schedule=SHORT, balance=ANNEAL).fit(Stiff())
    # Planted control: the same problem unbalanced leaves the boundary measurably worse, so this test can say no.
    errors = _boundary_error(fixed), _boundary_error(balanced)
    assert errors[1] < 0.3 * errors[0], (errors, balanced.weights)
    assert balanced.weights['boundary'] > 1e2 * balanced.weights['pde']


def test_the_weights_are_recorded_at_each_update_and_reproduce_under_a_seed():
    pif = PhysicsInformedField(hidden=(8, 8), schedule=Schedule(adam_steps=30, lbfgs_steps=2), balance=ANNEAL)
    a, b = pif.fit(Stiff()), pif.fit(Stiff())
    assert [step for step, _ in a.weight_history] == [0, 10, 20]
    assert a.weights == a.weight_history[-1][1] and a.weights['pde'] == 1.0
    assert a.weight_history == b.weight_history and a.history == b.history


def test_the_fixed_default_weights_every_term_one_and_records_no_update():
    fitted = PhysicsInformedField(hidden=(8, 8), schedule=Schedule(adam_steps=3)).fit(Poisson())
    assert fitted.weights == {'pde': 1.0, 'boundary': 1.0} and fitted.weight_history == []


def test_an_inverse_fit_balances_its_data_term_too():
    x = np.linspace(0.0, 1.0, 9)[:, None]
    pif = PhysicsInformedField(hidden=(8, 8), schedule=Schedule(adam_steps=11), balance=ANNEAL)
    fitted = pif.fit(Poisson(k_source=2.0), identify=('k',), observations={'u': (x, np.sin(np.pi * x))})
    assert set(fitted.weights) == {'pde', 'boundary', 'observed:u'}
    assert fitted.weights['observed:u'] != 1.0 and fitted.parameters['k'] != 1.0


def test_a_balance_out_of_range_an_unknown_reference_or_a_checkpoint_is_refused(tmp_path):
    for bad in (
        {'rule': 'softadapt'},
        {'rule': 'gradient_norm'},
        {'rule': 'gradient_norm', 'reference': 'pde', 'alpha': 0.0},
        {'rule': 'gradient_norm', 'reference': 'pde', 'every': 0},
    ):
        with pytest.raises(Refused):
            Balance(**bad)
    with pytest.raises(Refused, match='reference'):
        PhysicsInformedField(schedule=Schedule(adam_steps=1), balance=Balance('gradient_norm', 'bulk')).fit(Poisson())
    with pytest.raises(Refused, match='checkpoint'):
        PhysicsInformedField(schedule=Schedule(adam_steps=1), balance=ANNEAL).fit(
            Poisson(), checkpoint=Checkpoint(tmp_path, every=1)
        )


@dataclass(frozen=True)
class Satisfied(Poisson):
    """Poisson plus a condition the field meets almost for free: its gradient is ~1e-6 of the PDE's."""

    def residuals(self, field, points, parameters):
        terms = dict(super().residuals(field, points, parameters))
        terms['free'] = Term(1e-6 * field(points['boundary'])[:, 0], scale=1.0)
        return terms


def test_the_anchored_balance_holds_its_reference_at_one_where_the_normalised_one_starves_it():
    schedule = Schedule(adam_steps=60, learning_rate=2e-3)
    anchored = PhysicsInformedField(
        hidden=(8, 8), schedule=schedule, balance=Balance('anchored_gradient', 'pde', alpha=1.0, cap=50.0)
    ).fit(Satisfied())
    assert anchored.weights['pde'] == 1.0
    assert all(w <= 50.0 for w in anchored.weights.values()), anchored.weights
    # Planted control: the normalised rule pushes the satisfied term up and the reference toward zero.
    normalised = PhysicsInformedField(
        hidden=(8, 8), schedule=schedule, balance=Balance('normalised_gradient', 'pde', alpha=1.0)
    ).fit(Satisfied())
    assert normalised.weights['pde'] < 0.1, normalised.weights


def test_the_anchored_balance_still_fits_the_under_fitted_boundary():
    fixed = PhysicsInformedField(hidden=(16, 16), schedule=SHORT).fit(Stiff())
    anchored = PhysicsInformedField(
        hidden=(16, 16), schedule=SHORT, balance=Balance('anchored_gradient', 'pde', cap=1e4)
    ).fit(Stiff())
    assert _boundary_error(anchored) < 0.3 * _boundary_error(fixed), anchored.weights


def test_an_anchored_cap_below_one_is_refused():
    with pytest.raises(Refused, match='cap'):
        Balance('anchored_gradient', 'pde', cap=0.5)
