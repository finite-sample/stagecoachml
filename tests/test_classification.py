"""Tests for StagecoachClassifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from stagecoachml import StagecoachClassifier


@pytest.fixture
def binary_classification_data():
    """Create synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, n_redundant=0, n_informative=8, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, X


@pytest.fixture
def multiclass_classification_data():
    """Create synthetic multiclass classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_redundant=0, n_informative=8, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, X


@pytest.fixture
def basic_classifiers():
    """Create basic classifiers for testing."""
    return LogisticRegression(random_state=42), DecisionTreeClassifier(random_state=42)


class TestStagecoachClassifierBasic:
    """Basic functionality tests."""

    def test_init_default_parameters(self, basic_classifiers):
        """Test initialization with default parameters."""
        stage1, stage2 = basic_classifiers
        model = StagecoachClassifier(stage1, stage2)

        assert model.stage1_estimator is stage1
        assert model.stage2_estimator is stage2
        assert model.early_features is None
        assert model.late_features is None
        assert model.use_stage1_pred_as_feature is True
        assert model.inner_cv is None
        assert model.random_state is None

    def test_init_custom_parameters(self, basic_classifiers):
        """Test initialization with custom parameters."""
        stage1, stage2 = basic_classifiers
        early_features = ["feature_0", "feature_1"]
        late_features = ["feature_2", "feature_3"]

        model = StagecoachClassifier(
            stage1,
            stage2,
            early_features=early_features,
            late_features=late_features,
            use_stage1_pred_as_feature=False,
            inner_cv=3,
            random_state=123,
        )

        assert model.early_features == early_features
        assert model.late_features == late_features
        assert model.use_stage1_pred_as_feature is False
        assert model.inner_cv == 3
        assert model.random_state == 123

    def test_fit_binary_dataframe(self, binary_classification_data, basic_classifiers):
        """Test basic fit functionality with binary classification DataFrame."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        fitted_model = model.fit(X_df, y)

        assert fitted_model is model
        assert hasattr(model, "stage1_estimator_")
        assert hasattr(model, "stage2_estimator_")
        assert hasattr(model, "classes_")
        assert hasattr(model, "feature_names_in_")
        assert hasattr(model, "n_features_in_")

        np.testing.assert_array_equal(model.feature_names_in_, X_df.columns)
        assert model.n_features_in_ == X_df.shape[1]
        assert len(model.classes_) == 2

    def test_fit_multiclass_dataframe(self, multiclass_classification_data, basic_classifiers):
        """Test basic fit functionality with multiclass classification DataFrame."""
        X_df, y, _ = multiclass_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        fitted_model = model.fit(X_df, y)

        assert fitted_model is model
        assert hasattr(model, "classes_")
        assert len(model.classes_) == 3

    def test_fit_array(self, binary_classification_data, basic_classifiers):
        """Test basic fit functionality with numpy array."""
        _, y, X = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        fitted_model = model.fit(X, y)

        assert fitted_model is model
        assert model.feature_names_in_ is None
        assert model.n_features_in_ == X.shape[1]


class TestStagecoachClassifierPrediction:
    """Prediction functionality tests."""

    def test_predict_stage1_binary(self, binary_classification_data, basic_classifiers):
        """Test stage1 prediction with binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        predictions = model.predict_stage1(X_df)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)
        assert set(predictions).issubset(set(model.classes_))

    def test_predict_stage1_proba_binary(self, binary_classification_data, basic_classifiers):
        """Test stage1 probability prediction with binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        probabilities = model.predict_stage1_proba(X_df)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_df),)  # Binary returns single probability
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_predict_stage1_proba_multiclass(
        self, multiclass_classification_data, basic_classifiers
    ):
        """Test stage1 probability prediction with multiclass classification."""
        X_df, y, _ = multiclass_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        probabilities = model.predict_stage1_proba(X_df)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_df), 3)  # 3 classes
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_full_binary(self, binary_classification_data, basic_classifiers):
        """Test full prediction with binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        predictions = model.predict(X_df)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)
        assert set(predictions).issubset(set(model.classes_))

    def test_predict_proba_binary(self, binary_classification_data, basic_classifiers):
        """Test full probability prediction with binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        probabilities = model.predict_proba(X_df)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_df), 2)  # Binary has 2 classes
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_multiclass(self, multiclass_classification_data, basic_classifiers):
        """Test full probability prediction with multiclass classification."""
        X_df, y, _ = multiclass_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        probabilities = model.predict_proba(X_df)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_df), 3)  # 3 classes
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestStagecoachClassifierFeatures:
    """Feature handling tests."""

    def test_explicit_feature_specification(self, binary_classification_data, basic_classifiers):
        """Test with explicitly specified early/late features."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        early_features = ["feature_0", "feature_1", "feature_2"]
        late_features = ["feature_3", "feature_4", "feature_5"]

        model = StagecoachClassifier(
            stage1, stage2, early_features=early_features, late_features=late_features
        )
        model.fit(X_df, y)

        predictions = model.predict(X_df)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)

    def test_invalid_early_features(self, binary_classification_data, basic_classifiers):
        """Test with invalid early feature names."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        early_features = ["invalid_feature"]

        model = StagecoachClassifier(stage1, stage2, early_features=early_features)

        with pytest.raises(ValueError, match="Early features not found in data"):
            model.fit(X_df, y)


class TestStagecoachClassifierCaching:
    """Stage1 prediction caching tests."""

    def test_stage1_cache_functionality_binary(self, binary_classification_data, basic_classifiers):
        """Test stage1 prediction caching for binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        # Create a test dataset for caching (use arrays for consistency)
        X_test = X_df[:3].copy()
        X_early, _ = model._split_features(X_test.values)

        # Cache some probabilities
        cached_probs = np.array([0.1, 0.7, 0.9])
        model.set_stage1_cache(X_early, cached_probs)

        # Predictions should now use cached values when using same data
        stage1_probs = model.predict_stage1_proba(X_test.values)
        np.testing.assert_array_equal(stage1_probs, cached_probs)

    def test_clear_cache(self, binary_classification_data, basic_classifiers):
        """Test cache clearing functionality."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y)

        X_early, _ = model._split_features(X_df[:3])
        cached_probs = np.array([0.2, 0.8, 0.6])
        model.set_stage1_cache(X_early, cached_probs)

        # Clear cache
        model.clear_stage1_cache()

        # Should now use actual predictions instead of cached
        stage1_probs = model.predict_stage1_proba(X_df[:3])
        assert not np.allclose(stage1_probs, cached_probs)


class TestStagecoachClassifierCrossValidation:
    """Cross-validation compatibility tests."""

    def test_cross_val_score_compatibility_binary(
        self, binary_classification_data, basic_classifiers
    ):
        """Test compatibility with sklearn cross-validation for binary classification."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)

        # This should work without errors
        scores = cross_val_score(model, X_df, y, cv=3)

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)

    def test_cross_val_score_with_array_input(self, binary_classification_data, basic_classifiers):
        """Test cross-validation with array input."""
        _, y, X = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)

        scores = cross_val_score(model, X, y, cv=3)

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)

    def test_inner_cv_functionality(self, binary_classification_data, basic_classifiers):
        """Test inner cross-validation functionality."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2, inner_cv=3)
        model.fit(X_df, y)

        predictions = model.predict(X_df)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)


class TestStagecoachClassifierEstimatorValidation:
    """Estimator validation tests."""

    def test_stage1_estimator_validation_works(self, binary_classification_data):
        """Test that stage1 estimator validation logic works."""
        X_df, y, _ = binary_classification_data

        # Test that normal estimators work fine
        stage1 = LogisticRegression(random_state=42)
        stage2 = LogisticRegression(random_state=42)

        model = StagecoachClassifier(stage1, stage2)

        # This should work without error
        model.fit(X_df, y)
        predictions = model.predict(X_df)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)

    def test_stage2_estimator_without_predict_proba(self, binary_classification_data):
        """Test error when stage2 estimator doesn't support predict_proba."""
        X_df, y, _ = binary_classification_data

        # Use Perceptron which is a real sklearn classifier without predict_proba
        from sklearn.linear_model import Perceptron

        stage1 = LogisticRegression(random_state=42)
        stage2 = Perceptron(random_state=42)

        model = StagecoachClassifier(stage1, stage2)

        with pytest.raises(ValueError, match="stage2_estimator must implement predict_proba"):
            model.fit(X_df, y)


class TestStagecoachClassifierEdgeCases:
    """Edge case and error handling tests."""

    def test_unfitted_estimator_error(self, binary_classification_data, basic_classifiers):
        """Test error when using unfitted estimator."""
        X_df, _, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2)

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            model.predict(X_df)

    def test_sample_weights(self, binary_classification_data, basic_classifiers):
        """Test fitting with sample weights."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        # Create sample weights
        sample_weight = np.random.rand(len(y))

        model = StagecoachClassifier(stage1, stage2)
        model.fit(X_df, y, sample_weight=sample_weight)

        predictions = model.predict(X_df)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)

    def test_without_stage1_pred_as_feature(self, binary_classification_data, basic_classifiers):
        """Test mode where stage1 prediction is not used as stage2 feature."""
        X_df, y, _ = binary_classification_data
        stage1, stage2 = basic_classifiers

        model = StagecoachClassifier(stage1, stage2, use_stage1_pred_as_feature=False)
        model.fit(X_df, y)

        predictions = model.predict(X_df)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_df),)

    def test_more_tags(self, basic_classifiers):
        """Test _more_tags method."""
        stage1, stage2 = basic_classifiers
        model = StagecoachClassifier(stage1, stage2)

        tags = model._more_tags()

        assert tags["requires_y"] is True
        assert tags["requires_fit"] is True
        assert "X_types" in tags
        assert tags["allow_nan"] is False
