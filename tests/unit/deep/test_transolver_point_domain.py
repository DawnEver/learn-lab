"""A coordinate box is declared independently of the points that sample a function inside it."""

import numpy as np
import pytest

from learn_lab import artifact
from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import Transolver
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused
from learn_lab.validity import BoxEnvelope, Validity


def _data():
    coords = np.array([[0.2, 0.4], [0.8, 0.4], [0.2, 1.6], [0.8, 1.6]])
    x = np.broadcast_to(np.array([1.0, 2.0])[:, None, None], (2, 1, len(coords))).copy()
    return x, coords, x * coords.sum(axis=1)[None, None]


def _model(**options):
    return Transolver(
        width=8,
        heads=2,
        layers=1,
        slices=4,
        schedule=Schedule(adam_steps=2),
        runtime=Runtime(device='cpu'),
        **options,
    )


def _domain():
    return BoxEnvelope(np.array([0.0, 0.0]), np.array([1.0, 2.0]))


@pytest.mark.parametrize('reference', [None, 2])
def test_the_declared_box_drives_the_actual_normalizer_and_the_prediction_guard(reference):
    x, coords, y = _data()
    fitted = _model(reference=reference).fit(x, coords, y, point_domain=_domain())
    query = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]])
    seen = []
    hook = fitted.net.lift.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        answer = fitted.predict(np.full((1, 1, len(query)), 1.5), query)
    finally:
        hook.remove()
    assert answer.shape == (1, 1, len(query))
    assert np.isfinite(answer).all()
    np.testing.assert_array_equal(fitted.points.lower, [0.0, 0.0])
    np.testing.assert_array_equal(fitted.points.upper, [1.0, 2.0])
    np.testing.assert_allclose(seen[0][0, :, 1:3], [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], rtol=0, atol=1e-15)
    # The coordinate box is not a nonconvex-solid or hole-membership test.
    assert fitted.points.classify(query)[2] == Validity.IN_DOMAIN
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.ones((1, 1, 1)), np.array([[1.0 + 1e-10, 1.0]]))


def test_an_omitted_domain_keeps_the_observed_box_and_its_margin():
    x, coords, y = _data()
    fitted = _model(margin=0.1).fit(x, coords, y)
    expected = BoxEnvelope.of(coords, margin=0.1)
    np.testing.assert_array_equal(fitted.points.lower, expected.lower)
    np.testing.assert_array_equal(fitted.points.upper, expected.upper)
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.ones((1, 1, 1)), np.zeros((1, 2)))


def test_an_explicit_domain_is_exact_while_the_channel_margin_remains_selectable():
    x, coords, y = _data()
    fitted = _model(margin=0.1).fit(x, coords, y, point_domain=_domain())
    np.testing.assert_array_equal(fitted.points.lower, [0.0, 0.0])
    np.testing.assert_array_equal(fitted.points.upper, [1.0, 2.0])
    assert np.isfinite(fitted.predict(np.full((1, 1, 1), 2.05), np.zeros((1, 2)))).all()
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.full((1, 1, 1), 2.2), np.zeros((1, 2)))
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.ones((1, 1, 1)), np.array([[-0.01, 0.0]]))


@pytest.mark.parametrize(
    'domain',
    [
        ([0.0, 0.0], [1.0, 2.0]),
        BoxEnvelope(np.array([0.0]), np.array([1.0])),
        BoxEnvelope(np.array([[0.0, 0.0]]), np.array([[1.0, 2.0]])),
        BoxEnvelope(np.array([0.0, np.nan]), np.array([1.0, 2.0])),
        BoxEnvelope(np.zeros(2), np.array([1.0, np.inf])),
        BoxEnvelope(np.array([1.0, 0.0]), np.array([0.0, 2.0])),
        BoxEnvelope(np.array([0.3, 0.0]), np.array([1.0, 2.0])),
    ],
)
def test_a_malformed_or_training_excluding_domain_is_refused_by_name(domain):
    with pytest.raises(Refused, match='domain|training point'):
        _model().fit(*_data(), point_domain=domain)


def test_per_sample_points_and_a_flat_coordinate_dimension_keep_the_same_domain():
    x, coords, y = _data()
    own = np.stack([coords, coords])
    own = np.concatenate([own, np.zeros((*own.shape[:-1], 1))], axis=-1)
    domain = BoxEnvelope(np.zeros(3), np.array([1.0, 2.0, 0.0]))
    fitted = _model().fit(x, own, y, point_domain=domain)
    query = np.array([[0.1, 0.2, 0.0], [0.9, 1.8, 0.0]])
    assert fitted.predict(np.ones((1, 1, 2)), query).shape == (1, 1, 2)
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.ones((1, 1, 1)), np.array([[0.5, 1.0, 1e-12]]))


def test_the_domain_is_owned_by_the_fit_and_survives_the_artifact_round_trip(tmp_path):
    domain = _domain()
    model = _model()
    fitted = model.fit(*_data(), point_domain=domain)
    domain.lower[:] = -100.0
    domain.upper[:] = 100.0
    np.testing.assert_array_equal(fitted.points.lower, [0.0, 0.0])
    np.testing.assert_array_equal(fitted.net.lower.detach().cpu().numpy(), [0.0, 0.0])
    path = artifact.save(fitted, tmp_path / 'fit', declaration=model, data_identity='declared-coordinate-box')
    restored = artifact.load(path)
    query = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]])
    values = np.full((1, 1, len(query)), 1.5)
    np.testing.assert_array_equal(restored.predict(values, query), fitted.predict(values, query))
    np.testing.assert_array_equal(restored.points.upper, [1.0, 2.0])
    np.testing.assert_array_equal(restored.net.span.detach().cpu().numpy(), [1.0, 2.0])
    with pytest.raises(Refused, match='out_of_domain'):
        restored.predict(np.ones((1, 1, 1)), np.array([[1.01, 1.0]]))
