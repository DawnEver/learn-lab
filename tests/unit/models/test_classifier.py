"""The classifier family: fit on labels, predict the class and its probabilities.

Every member must beat chance on a separable synthetic set (held out), give probabilities that are a
distribution per point, and reload from an artifact answering bit for bit. A label-shuffled control
must NOT beat chance by the same margin -- the planted negative that proves the accuracy check can say no.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from learn_lab import artifact
from learn_lab.errors import Refused
from learn_lab.models.classifier import (
    ClassificationTree,
    ClassPrediction,
    LogisticRegression,
    NearestNeighborsClassifier,
)
from learn_lab.registry import CLASSIFIERS, classifier, family
from learn_lab.validity import BoxEnvelope

#: Probabilities sum to one per point (dimensionless, to rounding).
SUMS_TO_ONE = 1e-12
#: Held-out accuracy a member must reach on the separable set (fraction; chance is 1/3).
SEPARABLE_ACCURACY = 0.9


def _blobs(seed: int, n_per: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.85]])
    x = np.vstack([c + 0.05 * rng.standard_normal((n_per, 2)) for c in centers])
    labels = np.repeat(np.array(['healthy', 'short', 'open']), n_per)
    return np.clip(x, 0.0, 1.0), labels


X, LABELS = _blobs(1)
XQ, LQ = _blobs(2, 20)
#: The declared domain every fit answers in: the unit square the blobs are clipped to.
UNIT = BoxEnvelope(np.zeros(2), np.ones(2))

MEMBERS = {
    'nearest_neighbors_classifier': NearestNeighborsClassifier(k=5),
    'classification_tree': ClassificationTree(max_depth=4),
    'logistic_regression': LogisticRegression(alpha=1e-3),
}


def test_the_members_are_the_registered_classifiers() -> None:
    assert set(MEMBERS) == set(CLASSIFIERS)
    assert all(family(name) is CLASSIFIERS[name] for name in CLASSIFIERS)
    assert isinstance(classifier('classification_tree', max_depth=2), ClassificationTree)
    with pytest.raises(Refused, match='a classifier is one of'):
        classifier('polynomial')


@pytest.mark.parametrize('name', list(MEMBERS))
def test_each_member_beats_chance_on_a_separable_set(name) -> None:
    prediction = MEMBERS[name].fit(X, LABELS, domain=UNIT).predict(XQ)
    assert isinstance(prediction, ClassPrediction)
    assert np.mean(prediction.labels == LQ) >= SEPARABLE_ACCURACY


@pytest.mark.parametrize('name', list(MEMBERS))
def test_shuffled_labels_do_not_beat_chance_by_that_margin(name) -> None:
    """PLANTED CONTROL: with the labels decoupled from the points the accuracy check must say no."""
    shuffled = np.random.default_rng(3).permutation(LABELS)
    prediction = MEMBERS[name].fit(X, shuffled, domain=UNIT).predict(XQ)
    assert np.mean(prediction.labels == LQ) < SEPARABLE_ACCURACY


@pytest.mark.parametrize('name', list(MEMBERS))
def test_probabilities_are_a_distribution_over_the_classes(name) -> None:
    prediction = MEMBERS[name].fit(X, LABELS, domain=UNIT).predict(XQ)
    assert list(prediction.classes) == ['healthy', 'open', 'short']
    assert prediction.probabilities.shape == (len(XQ), 3)
    assert np.all(prediction.probabilities >= 0.0)
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=SUMS_TO_ONE)
    assert np.array_equal(prediction.labels, prediction.classes[np.argmax(prediction.probabilities, axis=1)])


@pytest.mark.parametrize('name', list(MEMBERS))
def test_a_reloaded_classifier_answers_bit_for_bit(name, tmp_path) -> None:
    declaration = MEMBERS[name]
    fitted = declaration.fit(X, LABELS, domain=UNIT)
    path = artifact.save(fitted, tmp_path / name, declaration=declaration, data_identity='sha256:planted')
    loaded = artifact.load(path)
    assert type(loaded) is type(fitted)
    before, after = fitted.predict(XQ), loaded.predict(XQ)
    assert np.array_equal(before.probabilities, after.probabilities)
    assert np.array_equal(before.labels, after.labels)
    record = artifact.manifest(path)
    assert record['family'] == name
    assert record['schema']['outputs'] == {'kind': 'classes', 'classes': ['healthy', 'open', 'short']}
    assert record['schema']['inputs']['width'] == 2


def test_integer_labels_keep_their_type() -> None:
    prediction = MEMBERS['classification_tree'].fit(X, np.repeat([0, 2, 1], 40), domain=UNIT).predict(XQ)
    assert prediction.labels.dtype.kind == 'i'


def test_knn_and_tree_match_sklearn_probabilities() -> None:
    """A kNN classifier is a vote of the k nearest; a Gini tree's leaves are class frequencies (scikit-learn, tests only)."""
    ours = MEMBERS['nearest_neighbors_classifier'].fit(X, LABELS, domain=UNIT).predict(XQ).probabilities
    theirs = KNeighborsClassifier(n_neighbors=5).fit(X, LABELS).predict_proba(XQ)
    assert np.allclose(ours, theirs, rtol=0.0, atol=SUMS_TO_ONE)
    ours = MEMBERS['classification_tree'].fit(X, LABELS, domain=UNIT).predict(XQ).labels
    theirs = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, LABELS).predict(XQ)
    assert np.mean(ours == theirs) >= 0.95


def test_logistic_agrees_with_sklearn_on_labels() -> None:
    ours = LogisticRegression(alpha=1e-3).fit(X, LABELS, domain=UNIT).predict(XQ).labels
    theirs = SkLogistic(C=1e3, max_iter=5000).fit(X, LABELS).predict(XQ)
    assert np.mean(ours == theirs) >= 0.95


def test_one_class_and_object_labels_are_refused() -> None:
    with pytest.raises(Refused, match='at least two classes'):
        LogisticRegression().fit(X, np.repeat('healthy', len(X)))
    with pytest.raises(Refused, match='one label per point'):
        LogisticRegression().fit(X, LABELS[:-1])
    with pytest.raises(Refused, match='object'):
        LogisticRegression().fit(X, np.array([object()] * len(X), dtype=object))


def test_a_query_outside_the_envelope_is_refused() -> None:
    fitted = MEMBERS['nearest_neighbors_classifier'].fit(X, LABELS, domain=UNIT)
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(np.array([[5.0, 5.0]]))
