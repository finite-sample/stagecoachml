"""Cross-fitting, weight routing and residual regression."""

from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer

from stagecoachml import PredictionEncoder, StagecoachRegressor


class RecordingRegressor(RegressorMixin, BaseEstimator):
    calls: ClassVar[list] = []

    def fit(self, X, y, sample_weight=None):
        self.calls.append((np.asarray(X).copy(), np.asarray(y).copy(), sample_weight))
        self.mean_ = np.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def model_for(cv=None, residual=False):
    return StagecoachRegressor(
        PredictionEncoder(LinearRegression()),
        LinearRegression(),
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
        cv=cv,
        residual=residual,
    )


@pytest.mark.parametrize("residual", [False, True])
def test_residual_target_and_independent_inference(data, residual):
    X, y = data
    model = model_for(residual=residual).fit(X, y)
    state = model.encode(X[["u", "v"]])
    design = np.column_stack([state.values, X[["item", "fresh"]]])
    expected_target = y - state.values[:, 0] if residual else y
    reference = LinearRegression().fit(design, expected_target)
    expected = reference.predict(design) + (state.values[:, 0] if residual else 0)
    np.testing.assert_allclose(
        model.predict_from_state(X[["item", "fresh"]], state), expected
    )
    np.testing.assert_allclose(model.predict(X), expected)


def test_residual_rejects_vector_encoding(data):
    X, y = data
    model = model_for(residual=True)
    model.encoder = FunctionTransformer()
    with pytest.raises(ValueError, match="single prediction"):
        model.fit(X, y)


@pytest.mark.parametrize("splitter", [KFold(4), TimeSeriesSplit(4), GroupKFold(4)])
def test_cross_fit_rows_weights_and_exclusion(data, splitter):
    X, y = data
    groups = np.repeat(np.arange(12), 10) if isinstance(splitter, GroupKFold) else None
    weights = np.linspace(1, 3, len(y))
    RecordingRegressor.calls = []
    model = StagecoachRegressor(
        PredictionEncoder(RecordingRegressor()),
        RecordingRegressor(),
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
        cv=splitter,
    ).fit(X, y, sample_weight=weights, groups=groups)
    calls = RecordingRegressor.calls
    assert len(calls) == 6
    np.testing.assert_array_equal(calls[0][2], weights)
    expected_scores = np.empty(len(y))
    seen = []
    for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
        np.testing.assert_array_equal(calls[fold + 1][1], y[train])
        np.testing.assert_array_equal(calls[fold + 1][2], weights[train])
        if isinstance(splitter, TimeSeriesSplit):
            assert max(train) < min(test)
        if groups is not None:
            assert set(groups[train]).isdisjoint(groups[test])
        expected_scores[test] = np.average(y[train], weights=weights[train])
        seen.extend(test)
    rows = np.sort(seen)
    np.testing.assert_array_equal(model.training_indices_, rows)
    np.testing.assert_allclose(calls[-1][0][:, 0], expected_scores[rows])
    np.testing.assert_array_equal(calls[-1][1], y[rows])
    np.testing.assert_array_equal(calls[-1][2], weights[rows])


@pytest.mark.parametrize(
    "folds",
    [
        [(np.array([0, 1]), np.array([1, 2]))],
        [(np.array([0, 1]), np.array([2, 3])), (np.array([0, 1]), np.array([3, 4]))],
        [(np.array([0, 0]), np.array([2, 3]))],
        [(np.array([0, 1]), np.array([2, 2]))],
        [],
    ],
)
def test_invalid_cross_fit_folds(data, folds):
    X, y = data
    with pytest.raises(ValueError, match=r"disjoint|duplicate|no held-out"):
        model_for(cv=folds).fit(X, y)


@pytest.mark.parametrize(
    "weights", [[1], np.zeros(120), -np.ones(120), np.full(120, np.nan)]
)
def test_invalid_weights(data, weights):
    X, y = data
    with pytest.raises(ValueError, match="sample_weight"):
        model_for().fit(X, y, sample_weight=weights)


def test_groups_require_correct_shape_and_splitter(data):
    X, y = data
    with pytest.raises(ValueError, match="one entry"):
        model_for(cv=3).fit(X, y, groups=[0])
    with pytest.raises(ValueError, match="explicit cv"):
        model_for().fit(X, y, groups=np.zeros(len(y)))


def test_weight_unsupported_is_not_silently_discarded(data):
    X, y = data
    model = model_for()
    model.encoder = FunctionTransformer()
    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X, y, sample_weight=np.ones(len(y)))


def test_frozen_predictor_cannot_claim_cross_fitting(data):
    from sklearn.frozen import FrozenEstimator

    X, y = data
    predictor = LinearRegression().fit(X[["u", "v"]], y)
    model = model_for(cv=3)
    model.encoder = PredictionEncoder(FrozenEstimator(predictor))
    with pytest.raises(ValueError, match="frozen prediction"):
        model.fit(X, y)


@pytest.mark.parametrize("residual", ["False", 0.5, 1, None])
def test_residual_requires_boolean(data, residual):
    X, y = data
    with pytest.raises(ValueError, match="residual must be a boolean"):
        model_for(residual=residual).fit(X, y)
