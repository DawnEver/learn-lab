"""The artifact contract on the numpy families: a saved fit reloads bit for bit, and a changed byte is refused.

Every comparison is EXACT (``np.array_equal``): the same arrays through the same arithmetic on the same machine
give the same bits, so a tolerance would only hide a lost field. ``test_the_comparison_sees_one_ulp`` is the
planted control that the comparison can say no.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from learn_lab import artifact
from learn_lab.equation import Library, SparseRegression, polynomial_library
from learn_lab.errors import Refused
from learn_lab.models.response import (
    Bagging,
    ChaosExpansion,
    CoKriging,
    Ensemble,
    Forest,
    GaussianProcess,
    GradientEnhancedKriging,
    Lasso,
    LossLawResidual,
    MultilayerPerceptron,
    MultiOutputKriging,
    NearestNeighbours,
    Polynomial,
    PowerLaw,
    RadialBasis,
    RegressionTree,
)
from learn_lab.registry import SURROGATES

RNG = np.random.default_rng(7)
X = RNG.uniform(0, 1, (24, 2))
Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1] ** 2, X[:, 0] * X[:, 1]])
SLOPES = np.stack([np.column_stack([3 * np.cos(3 * X[:, 0]), 2 * X[:, 1]])], axis=1)
QUERY = np.random.default_rng(8).uniform(0.05, 0.95, (9, 2))
IDENTITY = 'sha256:planted-training-data'

#: name -> (declaration, its fit). Every response family of the registry, plus the ensemble.
CASES = {
    'polynomial': (Polynomial(degree=3, penalty=Lasso(1e-4)), lambda m: m.fit(X, Y)),
    'power_law': (PowerLaw(fixed=((None, 1.0), (1.0, None))), lambda m: m.fit(X, Y)),
    'loss_law_residual': (
        LossLawResidual(inputs=('frequency', 'flux'), terms=('hysteresis', 'excess'), base=GaussianProcess(restarts=1)),
        lambda m: m.fit(X, np.exp(X @ [[1.0, 1.5], [2.0, 1.5]])),
    ),
    'radial_basis': (RadialBasis(), lambda m: m.fit(X, Y)),
    'gaussian_process': (GaussianProcess(restarts=1), lambda m: m.fit(X, Y)),
    'gradient_enhanced_kriging': (GradientEnhancedKriging(restarts=1), lambda m: m.fit(X, Y[:, :1], SLOPES)),
    'multi_output_kriging': (MultiOutputKriging(restarts=1), lambda m: m.fit(X, Y)),
    'co_kriging': (CoKriging(restarts=1), lambda m: m.fit(X, Y, np.arange(len(X)) % 2)),
    'nearest_neighbours': (NearestNeighbours(k=3, weights='distance'), lambda m: m.fit(X, Y)),
    'regression_tree': (RegressionTree(max_depth=4), lambda m: m.fit(X, Y)),
    'forest': (Forest(n_trees=4, seed=3), lambda m: m.fit(X, Y)),
    'multilayer_perceptron': (MultilayerPerceptron(hidden=(6,), max_iterations=40), lambda m: m.fit(X, Y)),
    'chaos_expansion': (ChaosExpansion(degree=2), lambda m: m.fit(X, Y)),
    'ensemble': (Ensemble((Polynomial(degree=2), NearestNeighbours(k=3)), k=3), lambda m: m.fit(X, Y)),
    'bagging': (Bagging(GaussianProcess(restarts=0), n_members=3, seed=5), lambda m: m.fit(X, Y)),
}


def _same(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))


def _answers(fitted) -> list:
    out = []
    for variance in (False, True):
        for gradient in (False, True):
            if (not variance or 'variance' in fitted.gives) and (not gradient or 'gradient' in fitted.gives):
                p = fitted.predict(QUERY, variance=variance, gradient=gradient)
                out += [p.mean, p.variance, p.gradient]
    return out


def test_the_cases_cover_every_registered_surrogate() -> None:
    assert set(SURROGATES) <= set(CASES) <= set(artifact.FAMILIES)


def test_the_comparison_sees_one_ulp() -> None:
    """PLANTED CONTROL: one unit in the last place is a different answer."""
    a = np.array([[0.3, 1.7]])
    assert _same(a, a.copy())
    assert not _same(a, a + np.spacing(a) * np.array([[0, 1]]))
    assert not _same(a, None)


@pytest.mark.parametrize('family', list(CASES))
def test_a_reloaded_response_model_answers_bit_for_bit(family, tmp_path) -> None:
    declaration, fit = CASES[family]
    fitted = fit(declaration)
    path = artifact.save(fitted, tmp_path / family, declaration=declaration, data_identity=IDENTITY)
    loaded = artifact.load(path)
    assert type(loaded) is type(fitted)
    assert loaded.gives == fitted.gives
    before, after = _answers(fitted), _answers(loaded)
    assert len(before) >= 3
    assert all(_same(a, b) for a, b in zip(before, after, strict=True))
    record = artifact.manifest(path)
    assert record['family'] == family
    envelope = {'lower': fitted.envelope.lower.tolist(), 'upper': fitted.envelope.upper.tolist()}
    assert record['schema']['inputs'] == {'kind': 'vector', 'width': 2, 'envelope': envelope}
    assert _same(loaded.envelope.lower, fitted.envelope.lower)
    assert _same(loaded.envelope.upper, fitted.envelope.upper)
    with pytest.raises(Refused, match='out_of_domain'):
        loaded.predict(fitted.envelope.upper[None, :] + 1.0)
    assert record['schema']['outputs']['width'] == fitted.predict(QUERY).mean.shape[1]


def test_the_manifest_records_what_an_audit_needs(tmp_path) -> None:
    declaration = Forest(n_trees=3, seed=11)
    path = artifact.save(
        declaration.fit(X, Y), tmp_path / 'f', declaration=declaration, data_identity=IDENTITY, metrics={'rmse': 0.25}
    )
    record = artifact.manifest(path)
    assert (record['format'], record['version']) == (artifact.FORMAT, artifact.VERSION)
    assert record['learn_lab']['version'] == '0.1.0'
    assert len(record['learn_lab']['git']['sha']) == 40
    assert record['seed'] == 11
    assert record['data_identity'] == IDENTITY
    assert record['metrics'] == {'rmse': 0.25}
    assert record['runtime'] is None
    assert record['declaration']['dataclass'] == 'learn_lab.models.response.tree:Forest'
    assert record['declaration']['fields']['n_trees'] == 3
    assert record['content_hash'].startswith('sha256:')
    assert set(record['payload']) == {artifact.ARRAYS}


def _rehash(path: Path, edit) -> None:
    """Edit the manifest and re-seal it: the hostile artifact a hash alone cannot refuse."""
    record = json.loads((path / artifact.MANIFEST).read_text(encoding='utf-8'))
    record.pop('content_hash')
    edit(record)
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    record['content_hash'] = 'sha256:' + hashlib.sha256(canonical).hexdigest()
    (path / artifact.MANIFEST).write_text(json.dumps(record), encoding='utf-8')


@pytest.fixture
def saved(tmp_path) -> Path:
    declaration = Polynomial(degree=2)
    return artifact.save(declaration.fit(X, Y), tmp_path / 'p', declaration=declaration, data_identity=IDENTITY)


def test_the_resealed_control_loads(saved) -> None:
    """PLANTED CONTROL: re-sealing an unchanged manifest is accepted, so each refusal below is the edit's."""
    _rehash(saved, lambda _record: None)
    assert _same(artifact.load(saved).predict(QUERY).mean, artifact.load(saved).predict(QUERY).mean)


def test_one_changed_payload_byte_is_refused(saved) -> None:
    data = bytearray((saved / artifact.ARRAYS).read_bytes())
    data[len(data) // 2] ^= 0x01
    (saved / artifact.ARRAYS).write_bytes(bytes(data))
    with pytest.raises(Refused, match='does not match its hash'):
        artifact.load(saved)


def test_a_changed_manifest_value_is_refused(saved) -> None:
    text = (saved / artifact.MANIFEST).read_text(encoding='utf-8')
    (saved / artifact.MANIFEST).write_text(text.replace(IDENTITY, 'sha256:other-data'), encoding='utf-8')
    with pytest.raises(Refused, match='does not match its content hash'):
        artifact.manifest(saved)


def test_an_unknown_format_version_is_refused(saved) -> None:
    _rehash(saved, lambda record: record.update(version=artifact.VERSION + 1))
    with pytest.raises(Refused, match=f'version {artifact.VERSION + 1}'):
        artifact.load(saved)


def test_a_class_outside_learn_lab_is_never_resolved(saved) -> None:
    def hostile(record) -> None:
        record['state']['dataclass'] = 'subprocess:Popen'

    _rehash(saved, hostile)
    with pytest.raises(Refused, match='only learn_lab classes'):
        artifact.load(saved)


def test_a_learn_lab_name_that_is_not_a_dataclass_is_never_called(saved) -> None:
    def hostile(record) -> None:
        record['state']['dataclass'] = 'learn_lab.errors:refuse'

    _rehash(saved, hostile)
    with pytest.raises(Refused, match='neither a dataclass nor an enum'):
        artifact.load(saved)


def test_a_pickled_array_is_never_unpickled(saved) -> None:
    """A consistent artifact whose npz holds an object array: numpy refuses it with ``allow_pickle=False``."""
    with zipfile.ZipFile(saved / artifact.ARRAYS, 'a') as npz, npz.open('evil.npy', 'w') as member:
        np.lib.format.write_array(member, np.array([print], dtype=object), allow_pickle=True)
    digest = 'sha256:' + hashlib.sha256((saved / artifact.ARRAYS).read_bytes()).hexdigest()
    _rehash(saved, lambda record: record['payload'].update({artifact.ARRAYS: digest}))
    with pytest.raises(ValueError, match='allow_pickle'):
        artifact.load(saved)


def test_save_refuses_what_it_cannot_record_honestly(tmp_path) -> None:
    declaration = Polynomial(degree=2)
    fitted = declaration.fit(X, Y)
    with pytest.raises(Refused, match='declared by'):
        artifact.save(fitted, tmp_path / 'a', declaration=RadialBasis(), data_identity=IDENTITY)
    with pytest.raises(Refused, match='identity of its training data'):
        artifact.save(fitted, tmp_path / 'b', declaration=declaration, data_identity=' ')
    with pytest.raises(Refused, match='finite number'):
        artifact.save(fitted, tmp_path / 'c', declaration=declaration, data_identity=IDENTITY, metrics={'r2': np.nan})
    with pytest.raises(Refused, match='not a fitted family'):
        artifact.save(declaration, tmp_path / 'd', declaration=declaration, data_identity=IDENTITY)
    artifact.save(fitted, tmp_path / 'e', declaration=declaration, data_identity=IDENTITY)
    with pytest.raises(Refused, match='never over another'):
        artifact.save(fitted, tmp_path / 'e', declaration=declaration, data_identity=IDENTITY)
    with pytest.raises(Refused, match='only a deep family'):
        artifact.load(tmp_path / 'e', device='cpu')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['e']


def _law_data():
    states = np.random.default_rng(3).uniform(-1, 1, (60, 2))
    rates = np.column_stack([-2 * states[:, 0] + states[:, 1], states[:, 0] * states[:, 1]])
    return states, rates


def test_an_identified_law_reloads_with_its_terms_and_envelope(tmp_path) -> None:
    states, rates = _law_data()
    declaration = SparseRegression(threshold=0.05)
    law = declaration.fit(polynomial_library(('x', 'y'), degree=2), states, rates)
    loaded = artifact.load(artifact.save(law, tmp_path / 'law', declaration=declaration, data_identity=IDENTITY))
    assert loaded.terms() == law.terms()
    assert _same(loaded.rate(states[:5]), law.rate(states[:5]))
    outside = np.array([[1.5, 0.0]])
    for model in (law, loaded):  # the original refuses it too: the point is genuinely outside
        with pytest.raises(Refused, match='out_of_domain'):
            model.rate(outside)


def test_a_law_on_a_callable_term_is_refused_naming_it(tmp_path) -> None:
    states, rates = _law_data()
    library = Library(('x', 'y'), ('x', 'sin(y)'), (lambda s: s[:, 0], lambda s: np.sin(s[:, 1])))
    declaration = SparseRegression(threshold=0.0)
    law = declaration.fit(library, states, rates)
    with pytest.raises(Refused, match=r'library\.terms\[0\] is a function'):
        artifact.save(law, tmp_path / 'law', declaration=declaration, data_identity=IDENTITY)
    assert not (tmp_path / 'law').exists()
