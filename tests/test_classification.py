"""Prediction encoders, class ordering and cross-fitted classification."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from stagecoachml import PredictionEncoder, StagecoachClassifier


@pytest.mark.parametrize("n_classes", [2, 3])
def test_class_order_and_out_of_fold_probabilities(n_classes):
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_classes=n_classes,
        random_state=12,
    )
    labels = np.array(["zebra", "ant", "mouse"])[y]
    model = StagecoachClassifier(
        PredictionEncoder(LogisticRegression()),
        LogisticRegression(),
        upstream_features=[0, 1, 2],
        online_features=[3, 4, 5],
        cv=3,
    ).fit(X, labels)
    state = model.encode(X[:, :3])
    assert state.values.shape == (len(X), n_classes)
    np.testing.assert_array_equal(model.classes_, np.unique(labels))
    np.testing.assert_allclose(
        model.predict_proba(X), model.predict_proba_from_state(X[:, 3:], state)
    )
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[model.predict_proba(X).argmax(axis=1)]
    )


def test_decision_function_is_a_margin(data):
    X, y = data
    model = StagecoachClassifier(
        PredictionEncoder(LinearSVC(), method="decision_function"),
        LogisticRegression(),
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
        cv=3,
    ).fit(X, y > 0)
    state = model.encode(X[["u", "v"]])
    expected = model.encoder_.estimator_.decision_function(X[["u", "v"]])
    np.testing.assert_allclose(state.values[:, 0], expected)
    assert np.any((state.values < 0) | (state.values > 1))
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.parametrize(
    ("method", "estimator", "message"),
    [
        ("missing", LinearRegression(), "method must"),
        ("predict_proba", LinearRegression(), "does not support"),
    ],
)
def test_invalid_prediction_methods(data, method, estimator, message):
    X, y = data
    with pytest.raises(ValueError, match=message):
        PredictionEncoder(estimator, method=method).fit(X, y)


def test_missing_class_fold_is_rejected():
    X = np.arange(80).reshape(20, 4)
    y = np.repeat([0, 1], 10)
    model = StagecoachClassifier(
        PredictionEncoder(DummyClassifier()),
        LogisticRegression(),
        upstream_features=[0, 1],
        online_features=[2, 3],
        cv=KFold(2),
    )
    with pytest.raises(ValueError, match="same classes"):
        model.fit(X, y)


def test_online_classifier_requires_probabilities(data):
    X, y = data
    model = StagecoachClassifier(
        PredictionEncoder(LogisticRegression()),
        LinearSVC(),
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
    )
    with pytest.raises(ValueError, match="predict_proba"):
        model.fit(X, y > 0)


def test_prediction_encoder_failed_refit_is_invalidated(data):
    from sklearn.exceptions import NotFittedError

    X, y = data
    encoder = PredictionEncoder(LogisticRegression()).fit(X, y > 0)
    encoder.method = "missing"
    with pytest.raises(ValueError, match="method must"):
        encoder.fit(X, y > 0)
    with pytest.raises(NotFittedError):
        encoder.transform(X)


def test_prediction_encoder_refit_resets_classes(data):
    X, y = data
    encoder = PredictionEncoder(LogisticRegression()).fit(X, y > 0)
    encoder.estimator = LinearRegression()
    encoder.fit(X, y)
    assert not hasattr(encoder, "classes_")
    assert encoder.transform(X).shape == (len(X), 1)
