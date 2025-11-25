"""Tests for base classes and utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from stagecoachml._base import StagecoachBase
from stagecoachml._validation import (
    check_consistent_length_features,
    validate_cv_parameter,
    validate_estimator,
)


class MockStagecoachEstimator(StagecoachBase):
    """Mock implementation for testing base functionality."""

    def fit(self, X, y, sample_weight=None):
        self._validate_features(X)
        self.is_fitted_ = True
        return self

    def predict_stage1(self, X):
        X_early, _ = self._split_features(X)
        return np.ones(len(X_early))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "early_1": [1, 2, 3, 4, 5],
        "early_2": [2, 4, 6, 8, 10],
        "late_1": [3, 6, 9, 12, 15],
        "late_2": [4, 8, 12, 16, 20],
    }
    df = pd.DataFrame(data)
    array = df.values
    y = np.array([1, 2, 3, 4, 5])
    return df, array, y


class TestStagecoachBase:
    """Tests for StagecoachBase class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        assert base.stage1_estimator is stage1
        assert base.stage2_estimator is stage2
        assert base.early_features is None
        assert base.late_features is None
        assert base.use_stage1_pred_as_feature is True
        assert base.inner_cv is None
        assert base.random_state is None
        assert isinstance(base._stage1_cache, dict)
        assert len(base._stage1_cache) == 0

    def test_validate_features_dataframe(self, sample_data):
        """Test feature validation with DataFrame."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(
            stage1,
            stage2,
            early_features=["early_1", "early_2"],
            late_features=["late_1", "late_2"],
        )

        # Should not raise any errors
        base._validate_features(df)

        assert hasattr(base, "feature_names_in_")
        assert hasattr(base, "n_features_in_")
        np.testing.assert_array_equal(base.feature_names_in_, df.columns)
        assert base.n_features_in_ == 4

    def test_validate_features_array(self, sample_data):
        """Test feature validation with numpy array."""
        _, array, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        # Should not raise any errors
        base._validate_features(array)

        assert hasattr(base, "feature_names_in_")
        assert hasattr(base, "n_features_in_")
        assert base.feature_names_in_ is None
        assert base.n_features_in_ == 4

    def test_validate_features_missing_early(self, sample_data):
        """Test feature validation with missing early features."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2, early_features=["nonexistent_feature"])

        with pytest.raises(ValueError, match="Early features not found in data"):
            base._validate_features(df)

    def test_validate_features_missing_late(self, sample_data):
        """Test feature validation with missing late features."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2, late_features=["nonexistent_feature"])

        with pytest.raises(ValueError, match="Late features not found in data"):
            base._validate_features(df)

    def test_split_features_explicit_dataframe(self, sample_data):
        """Test feature splitting with explicit feature names on DataFrame."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(
            stage1,
            stage2,
            early_features=["early_1", "early_2"],
            late_features=["late_1", "late_2"],
        )

        X_early, X_late = base._split_features(df)

        assert isinstance(X_early, pd.DataFrame)
        assert isinstance(X_late, pd.DataFrame)
        assert list(X_early.columns) == ["early_1", "early_2"]
        assert list(X_late.columns) == ["late_1", "late_2"]

    def test_split_features_default_dataframe(self, sample_data):
        """Test feature splitting with default behavior on DataFrame."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, X_late = base._split_features(df)

        assert isinstance(X_early, pd.DataFrame)
        assert isinstance(X_late, pd.DataFrame)
        assert len(X_early.columns) == 2  # First half
        assert len(X_late.columns) == 2  # Second half

    def test_split_features_array_with_stored_names(self, sample_data):
        """Test feature splitting on array with stored feature names."""
        df, array, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(
            stage1,
            stage2,
            early_features=["early_1", "early_2"],
            late_features=["late_1", "late_2"],
        )

        # First validate with DataFrame to store feature names
        base._validate_features(df)

        # Now split array using stored feature names
        X_early, X_late = base._split_features(array)

        assert isinstance(X_early, np.ndarray)
        assert isinstance(X_late, np.ndarray)
        assert X_early.shape == (5, 2)
        assert X_late.shape == (5, 2)

    def test_split_features_array_default(self, sample_data):
        """Test feature splitting with default behavior on array."""
        _, array, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, X_late = base._split_features(array)

        assert isinstance(X_early, np.ndarray)
        assert isinstance(X_late, np.ndarray)
        assert X_early.shape == (5, 2)  # First half
        assert X_late.shape == (5, 2)  # Second half

    def test_cache_functionality_dataframe(self, sample_data):
        """Test stage1 caching with DataFrame."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, _ = base._split_features(df)
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Cache predictions
        base.set_stage1_cache(X_early, predictions)

        # Retrieve cached predictions
        cached = base._get_cached_stage1_pred(X_early)

        assert cached is not None
        np.testing.assert_array_equal(cached, predictions)

    def test_cache_functionality_array(self, sample_data):
        """Test stage1 caching with array."""
        _, array, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, _ = base._split_features(array)
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Cache predictions
        base.set_stage1_cache(X_early, predictions)

        # Retrieve cached predictions
        cached = base._get_cached_stage1_pred(X_early)

        assert cached is not None
        np.testing.assert_array_equal(cached, predictions)

    def test_clear_cache(self, sample_data):
        """Test cache clearing functionality."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, _ = base._split_features(df)
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Cache predictions
        base.set_stage1_cache(X_early, predictions)
        assert len(base._stage1_cache) == 1

        # Clear cache
        base.clear_stage1_cache()
        assert len(base._stage1_cache) == 0

        # Should return None now
        cached = base._get_cached_stage1_pred(X_early)
        assert cached is None

    def test_cache_key_generation_different_data(self, sample_data):
        """Test that different data generates different cache keys."""
        df, _, _ = sample_data
        stage1 = LinearRegression()
        stage2 = LinearRegression()

        base = MockStagecoachEstimator(stage1, stage2)

        X_early, _ = base._split_features(df)

        # Create modified data
        df_modified = df.copy()
        df_modified.iloc[0, 0] = 999
        X_early_modified, _ = base._split_features(df_modified)

        key1 = base._get_cache_key(X_early)
        key2 = base._get_cache_key(X_early_modified)

        assert key1 != key2


class TestValidationUtilities:
    """Tests for validation utilities."""

    def test_validate_estimator_regressor(self):
        """Test regressor validation."""
        regressor = LinearRegression()

        # Should not raise
        validate_estimator(regressor, "regressor")

        # Should raise for wrong type
        classifier = LogisticRegression()
        with pytest.raises(ValueError, match="Expected regressor"):
            validate_estimator(classifier, "regressor")

    def test_validate_estimator_classifier(self):
        """Test classifier validation."""
        classifier = LogisticRegression()

        # Should not raise
        validate_estimator(classifier, "classifier")

        # Should raise for wrong type
        regressor = LinearRegression()
        with pytest.raises(ValueError, match="Expected classifier"):
            validate_estimator(regressor, "classifier")

    def test_validate_cv_parameter_valid(self):
        """Test valid CV parameter validation."""
        # Should not raise
        validate_cv_parameter(None)
        validate_cv_parameter(3)
        validate_cv_parameter(5)

    def test_validate_cv_parameter_invalid(self):
        """Test invalid CV parameter validation."""
        with pytest.raises(ValueError, match="inner_cv must be None or an integer >= 2"):
            validate_cv_parameter(1)

        with pytest.raises(ValueError, match="inner_cv must be None or an integer >= 2"):
            validate_cv_parameter("invalid")

        with pytest.raises(ValueError, match="inner_cv must be None or an integer >= 2"):
            validate_cv_parameter(0)

    def test_check_consistent_length_features_valid(self):
        """Test feature consistency check with valid features."""
        X = pd.DataFrame(
            {
                "early_1": [1, 2, 3],
                "early_2": [2, 4, 6],
                "late_1": [3, 6, 9],
                "late_2": [4, 8, 12],
            }
        )

        early_features = ["early_1", "early_2"]
        late_features = ["late_1", "late_2"]

        # Should not raise
        check_consistent_length_features(early_features, late_features, X)

    def test_check_consistent_length_features_overlap(self):
        """Test feature consistency check with overlapping features."""
        X = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [2, 4, 6],
                "feature_3": [3, 6, 9],
            }
        )

        early_features = ["feature_1", "feature_2"]
        late_features = ["feature_2", "feature_3"]  # feature_2 overlaps

        with pytest.raises(ValueError, match="Features cannot be both early and late"):
            check_consistent_length_features(early_features, late_features, X)
