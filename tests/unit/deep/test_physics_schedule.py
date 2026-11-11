"""Opt-in training schedules of a physics fit: a parameter continuation and residual-based adaptive refinement."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from learn_lab.deep.fit import Checkpoint, Schedule
from learn_lab.deep.physics import Continuation, PhysicsInformedField, Resample, Term
from learn_lab.errors import Refused

from ._poisson import Poisson

SHORT = Schedule(adam_steps=40, learning_rate=2e-3, lbfgs_steps=2)


@dataclass(frozen=True, eq=False)
class Recording(Poisson):
    """Poisson that records, per residual call, the ``k`` it was handed and its interior points (named ``pde``, as
    its term is); its PDE residual is over-weighted 100x on ``x > 0.9``, so it is largest there whatever the field.
    """

    seen: list = field(default_factory=list)

    def collocation(self, runtime):
        points = dict(super().collocation(runtime))
        return {'pde': points.pop('interior'), **points}

    def residuals(self, field, points, parameters):
        points = {'interior': points['pde'], 'boundary': points['boundary']}
        self.seen.append((float(parameters['k']), points['interior'].detach().cpu().numpy()[:, 0].copy()))
        terms = dict(super().residuals(field, points, parameters))
        x = points['interior'][:, 0]
        terms['pde'] = Term(terms['pde'].residual * (1.0 + 99.0 * (x > 0.9)), terms['pde'].scale)
        return terms


def test_a_continued_parameter_moves_from_its_start_to_its_nominal_and_holds():
    provider = Recording(k_nominal=1.0)
    fitted = PhysicsInformedField(
        hidden=(8,), schedule=SHORT, continuation=(Continuation('k', start=3.0, steps=20),)
    ).fit(provider)
    k = [s[0] for s in provider.seen]
    assert k[0] == pytest.approx(3.0, abs=1e-12)  # dimensionless k: the start
    assert k[10] == pytest.approx(2.0, abs=1e-12)  # halfway
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in k[20:])  # held at nominal, polish included
    assert fitted.parameters['k'] == pytest.approx(1.0, abs=1e-12)
    # Planted control: without a continuation the fit is handed the nominal k throughout.
    plain = Recording(k_nominal=1.0)
    PhysicsInformedField(hidden=(8,), schedule=SHORT).fit(plain)
    assert {s[0] for s in plain.seen} == {1.0}


def test_resampling_adds_the_fresh_points_where_the_residual_is_largest():
    provider = Recording(points=64)
    PhysicsInformedField(hidden=(8,), schedule=SHORT, resample=Resample(conditions=('pde',), every=10, add=4)).fit(
        provider
    )
    first, last = provider.seen[0][1], provider.seen[-1][1]
    assert len(first) == 64
    assert len(last) == 64 + 4 * 3  # rounds at Adam steps 10, 20, 30
    added = last[64:]
    assert (added > 0.9).all(), added
    # Planted control: the sampler alone puts most points below 0.9, so the selection is what moved them.
    assert (first > 0.9).mean() < 0.3


def test_a_schedule_naming_an_unknown_or_identified_parameter_or_point_set_or_a_checkpoint_is_refused(tmp_path):
    with pytest.raises(Refused, match='continuation names'):
        PhysicsInformedField(hidden=(8,), schedule=SHORT, continuation=(Continuation('q', 1.0, 5),)).fit(Poisson())
    with pytest.raises(Refused, match='identified'):
        PhysicsInformedField(hidden=(8,), schedule=SHORT, continuation=(Continuation('k', 2.0, 5),)).fit(
            Poisson(), identify=('k',), observations={'u': (np.array([[0.5]]), np.array([[1.0]]))}
        )
    with pytest.raises(Refused, match='resample names'):
        PhysicsInformedField(hidden=(8,), schedule=SHORT, resample=Resample(conditions=('pde:x',))).fit(Poisson())
    with pytest.raises(Refused, match='checkpointed'):
        PhysicsInformedField(hidden=(8,), schedule=SHORT, resample=Resample(conditions=('pde',))).fit(
            Recording(), checkpoint=Checkpoint(tmp_path, every=5)
        )
    with pytest.raises(Refused):
        Continuation('k', 1.0, 0)
    with pytest.raises(Refused):
        Resample(conditions=())
