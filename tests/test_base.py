"""Serving contracts, schemas, state lifecycle and sklearn composition."""

from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from stagecoachml import EncodedState, StagecoachClassifier, StagecoachRegressor


def make_model(classifier=False, **kwargs):
    cls = StagecoachClassifier if classifier else StagecoachRegressor
    estimator = LogisticRegression() if classifier else LinearRegression()
    return cls(
        FunctionTransformer(),
        estimator,
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
        **kwargs,
    )


@pytest.mark.parametrize("classifier", [False, True])
def test_independent_serving_and_candidate_mapping(data, classifier):
    X, y = data
    model = make_model(classifier).fit(X, y > 0 if classifier else y)
    upstream = X.iloc[:3][["u", "v"]]
    state = model.encode(upstream, entity_ids=np.array([30, 10, 20]))
    indices = state.rows_for([20, 30, 20, 10])
    online = X.iloc[6:10][["item", "fresh"]]
    combined = online.copy()
    combined[["u", "v"]] = upstream.to_numpy()[indices]
    expected = model.predict(combined)
    with patch.object(
        model.encoder_, "transform", side_effect=AssertionError("upstream called")
    ):
        np.testing.assert_allclose(
            model.predict_from_state(online, state, state_indices=indices), expected
        )
        np.testing.assert_allclose(
            model.predictor_.predict(online, state, state_indices=indices), expected
        )
        if classifier:
            proba = model.predict_proba_from_state(online, state, state_indices=indices)
            assert proba.shape == (4, 2)
            np.testing.assert_allclose(proba.sum(axis=1), 1)


@pytest.mark.parametrize("classifier", [False, True])
def test_sklearn_clone_search_and_scoring(data, classifier):
    X, y = data
    model = make_model(classifier)
    target = (y > 0).astype(int) if classifier else y
    assert is_classifier(model) == classifier
    assert is_regressor(model) != classifier
    assert not hasattr(clone(model), "predictor_")
    search = GridSearchCV(model, {"estimator__fit_intercept": [True, False]}, cv=3).fit(
        X, target
    )
    assert np.isfinite(search.best_score_)
    assert np.all(np.isfinite(cross_val_score(search.best_estimator_, X, target, cv=3)))


def test_frozen_encoder_preserves_coordinates(data):
    X, y = data
    encoder = PCA(n_components=1).fit(X.iloc[:30][["u", "v"]])
    components = encoder.components_.copy()
    model = StagecoachRegressor(
        FrozenEstimator(encoder),
        LinearRegression(),
        upstream_features=["u", "v"],
        online_features=["item", "fresh"],
    )
    with patch.object(
        encoder, "fit", side_effect=AssertionError("frozen encoder refitted")
    ):
        model.fit(X.iloc[30:], y[30:])
    np.testing.assert_array_equal(encoder.components_, components)
    np.testing.assert_allclose(
        model.encode(X[["u", "v"]]).values, encoder.transform(X[["u", "v"]])
    )


def test_serialize_online_artifact_without_encoder(data, tmp_path):
    X, y = data
    model = make_model().fit(X, y)
    state = model.encode(X[["u", "v"]])
    expected = model.predict(X)
    path = tmp_path / "online.joblib"
    joblib.dump((model.predictor_, state), path)
    predictor, restored_state = joblib.load(path)
    assert not hasattr(predictor, "encoder")
    np.testing.assert_allclose(
        predictor.predict(X[["item", "fresh"]], restored_state), expected
    )
    assert not hasattr(predictor, "predict_proba")


def test_refit_invalidates_state_but_old_deployment_remains_valid(data):
    X, y = data
    model = make_model().fit(X, y)
    state = model.encode(X[["u", "v"]])
    old_predictor = model.predictor_
    expected = model.predict(X)
    model.fit(X, y + 10)
    with pytest.raises(ValueError, match="different encoder fit"):
        model.predict_from_state(X[["item", "fresh"]], state)
    np.testing.assert_allclose(
        old_predictor.predict(X[["item", "fresh"]], state), expected
    )


def test_failed_refit_cannot_serve_old_estimator(data):
    X, y = data
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="inconsistent numbers"):
        model.fit(X, y[:-1])
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_column_order_and_arrays(data):
    X, y = data
    model = make_model().fit(X, y)
    expected = model.predict(X)
    np.testing.assert_allclose(model.predict(X[X.columns[::-1]]), expected)
    state = model.encode(X[["v", "u"]])
    np.testing.assert_allclose(
        model.predict_from_state(X[["fresh", "item"]], state), expected
    )
    state_array = model.encode(X[["u", "v"]].to_numpy())
    np.testing.assert_allclose(
        model.predict_from_state(X[["item", "fresh"]].to_numpy(), state_array), expected
    )
    positional = StagecoachRegressor(
        StandardScaler(),
        LinearRegression(),
        upstream_features=[0, 1],
        online_features=[2, 3],
    ).fit(X.to_numpy(), y)
    np.testing.assert_allclose(positional.predict(X.to_numpy()), expected, atol=1e-12)


def test_no_online_features(data):
    X, y = data
    model = StagecoachRegressor(
        FunctionTransformer(),
        LinearRegression(),
        upstream_features=[0, 1, 2, 3],
        online_features=[],
    ).fit(X.to_numpy(), y)
    state = model.encode(X.to_numpy())
    np.testing.assert_allclose(
        model.predict_from_state(np.empty((len(X), 0)), state),
        model.predict(X.to_numpy()),
    )


@pytest.mark.parametrize(
    ("upstream", "online", "message"),
    [
        ([], ["item"], "At least one"),
        (["missing"], ["item"], "missing"),
        (["u", "u"], ["item"], "duplicates"),
        (["u", 1], ["item"], "only strings"),
        ([-1], ["item"], "outside"),
        ([False], ["item"], "only strings"),
        ("u", ["item"], "lists or tuples"),
    ],
)
def test_invalid_feature_selectors(data, upstream, online, message):
    X, y = data
    model = StagecoachRegressor(
        FunctionTransformer(),
        LinearRegression(),
        upstream_features=upstream,
        online_features=online,
    )
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_schema_failures(data):
    X, y = data
    model = make_model().fit(X, y)
    state = model.encode(X[["u", "v"]])
    for bad in [X[["u"]], X[["u", "v", "u"]]]:
        with pytest.raises(ValueError, match=r"missing|unique"):
            model.encode(bad)
    with pytest.raises(ValueError, match="width"):
        model.encode(X.to_numpy())
    with pytest.raises(ValueError, match="missing"):
        model.predict(X[["u", "v"]])
    with pytest.raises(ValueError, match="missing"):
        model.predict_from_state(X[["item"]], state)
    with pytest.raises(ValueError, match="width"):
        model.predict_from_state(X.to_numpy(), state)
    with pytest.raises(ValueError, match="unique"):
        model.predict_from_state(X[["item", "fresh", "item"]], state)
    with pytest.raises(ValueError, match="unique"):
        make_model().fit(X[["u", "v", "item", "item"]], y)
    with pytest.raises(ValueError, match="Named feature"):
        make_model().fit(X.to_numpy(), y)


def test_encoder_contract_failures(data):
    X, y = data
    model = make_model()
    model.encoder = LinearRegression()
    with pytest.raises(ValueError, match="PredictionEncoder"):
        model.fit(X, y)
    model.encoder = FunctionTransformer()
    model.cv = 3
    with pytest.raises(ValueError, match="freeze other encoders"):
        model.fit(X, y)
    model.cv = None
    model.fit(X, y)
    with (
        patch.object(model.encoder_, "transform", return_value=np.ones((len(X), 3))),
        pytest.raises(ValueError, match="width changed"),
    ):
        model.encode(X[["u", "v"]])
    with (
        patch.object(model.encoder_, "transform", return_value=np.ones((1, 2))),
        pytest.raises(ValueError, match="number of input rows"),
    ):
        model.encode(X[["u", "v"]])


def test_state_shape_and_alignment_failures(data):
    X, y = data
    model = make_model().fit(X, y)
    state = model.encode(X[["u", "v"]])
    malformed = EncodedState(np.ones((len(X), 3)), state.encoder_id)
    with pytest.raises(ValueError, match="representation width"):
        model.predict_from_state(X[["item", "fresh"]], malformed)
    with pytest.raises(ValueError, match="require state_indices"):
        model.predict_from_state(X.iloc[:2][["item", "fresh"]], state)
    with pytest.raises(ValueError, match="one entry per candidate"):
        model.predict_from_state(
            X.iloc[:2][["item", "fresh"]], state, state_indices=[1]
        )


@pytest.mark.parametrize("operation", ["predict", "encode", "predict_proba"])
def test_unfitted_classifier(data, operation):
    X, _ = data
    with pytest.raises(NotFittedError):
        getattr(make_model(True), operation)(X)


@pytest.mark.parametrize("classifier", [False, True])
def test_wrong_downstream_estimator_type(data, classifier):
    X, y = data
    model = make_model(classifier)
    model.estimator = LinearRegression() if classifier else LogisticRegression()
    with pytest.raises(ValueError, match="estimator must be"):
        model.fit(X, y > 0 if classifier else y)


def test_categorical_upstream_pipeline(data):
    from sklearn.preprocessing import OneHotEncoder

    X, y = data
    X = X.copy()
    X["category"] = np.where(X.u > 0, "positive", "negative")
    model = StagecoachRegressor(
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        LinearRegression(),
        upstream_features=["category"],
        online_features=["item", "fresh"],
    ).fit(X, y)
    state = model.encode(X[["category"]])
    np.testing.assert_allclose(
        model.predict_from_state(X[["item", "fresh"]], state), model.predict(X)
    )
