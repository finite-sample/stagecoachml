"""Shared fixtures and configuration for tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from stagecoachml import StagecoachClassifier, StagecoachRegressor


@pytest.fixture
def sample_classification_dataframe() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a sample classification dataframe for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return df, y


@pytest.fixture
def sample_regression_dataframe() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a sample regression dataframe for testing."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        noise=0.1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return df, y


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_regression_estimators():
    """Create basic regression estimators for testing."""
    return LinearRegression(), RandomForestRegressor(n_estimators=5, random_state=42)


@pytest.fixture
def basic_classification_estimators():
    """Create basic classification estimators for testing."""
    return LogisticRegression(random_state=42), RandomForestClassifier(
        n_estimators=5, random_state=42
    )


@pytest.fixture
def fitted_stagecoach_regressor(sample_regression_dataframe, basic_regression_estimators):
    """Create a fitted StagecoachRegressor for testing."""
    X_df, y = sample_regression_dataframe
    stage1, stage2 = basic_regression_estimators

    early_features = ["feature_0", "feature_1", "feature_2"]
    late_features = ["feature_3", "feature_4", "feature_5"]

    model = StagecoachRegressor(
        stage1, stage2, early_features=early_features, late_features=late_features
    )
    model.fit(X_df, y)
    return model, X_df, y


@pytest.fixture
def fitted_stagecoach_classifier(sample_classification_dataframe, basic_classification_estimators):
    """Create a fitted StagecoachClassifier for testing."""
    X_df, y = sample_classification_dataframe
    stage1, stage2 = basic_classification_estimators

    early_features = ["feature_0", "feature_1", "feature_2"]
    late_features = ["feature_3", "feature_4", "feature_5"]

    model = StagecoachClassifier(
        stage1, stage2, early_features=early_features, late_features=late_features
    )
    model.fit(X_df, y)
    return model, X_df, y
