"""Base classes for StagecoachML estimators."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class StagecoachBase(BaseEstimator, ABC):
    """Base class for two-stage estimators."""

    def __init__(
        self,
        stage1_estimator,
        stage2_estimator,
        early_features: list[str] | None = None,
        late_features: list[str] | None = None,
        use_stage1_pred_as_feature: bool = True,
        inner_cv: int | None = None,
        random_state: int | None = None,
    ):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.early_features = early_features
        self.late_features = late_features
        self.use_stage1_pred_as_feature = use_stage1_pred_as_feature
        self.inner_cv = inner_cv
        self.random_state = random_state

        # Cache for stage1 predictions (for latency optimization)
        self._stage1_cache: dict[str, np.ndarray] = {}

    def _validate_features(self, X):
        """Validate feature specifications against input data."""
        # Store the original feature names for later use (only during initial fit)
        if not hasattr(self, "feature_names_in_"):
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = np.array(X.columns)
                self.n_features_in_ = X.shape[1]
            else:
                self.feature_names_in_ = None
                self.n_features_in_ = X.shape[1]

        # Only validate if we have a DataFrame and feature names specified
        if isinstance(X, pd.DataFrame) and (
            self.early_features is not None or self.late_features is not None
        ):
            available_features = list(X.columns)

            if self.early_features is not None:
                missing_early = set(self.early_features) - set(available_features)
                if missing_early:
                    raise ValueError(f"Early features not found in data: {missing_early}")

            if self.late_features is not None:
                missing_late = set(self.late_features) - set(available_features)
                if missing_late:
                    raise ValueError(f"Late features not found in data: {missing_late}")

    def _split_features(self, X):
        """Split features into early and late groups."""
        if isinstance(X, pd.DataFrame):
            if self.early_features is not None:
                X_early = X[self.early_features]
            else:
                # Default: use first half of features as early
                mid = len(X.columns) // 2
                X_early = X.iloc[:, :mid]

            if self.late_features is not None:
                X_late = X[self.late_features]
            else:
                # Default: use second half of features as late
                mid = len(X.columns) // 2
                X_late = X.iloc[:, mid:]
        # Array input
        elif (
            self.early_features is not None
            and hasattr(self, "feature_names_in_")
            and self.feature_names_in_ is not None
        ):
            # Convert to DataFrame for feature selection, then back to array
            df = pd.DataFrame(X, columns=self.feature_names_in_)
            X_early = df[self.early_features].values
            X_late = df[self.late_features].values if self.late_features is not None else None

            if X_late is None:
                # Use remaining features
                remaining_features = [
                    f for f in self.feature_names_in_ if f not in self.early_features
                ]
                X_late = df[remaining_features].values
        else:
            # Use indices for feature splitting
            mid = X.shape[1] // 2
            X_early = X[:, :mid]
            X_late = X[:, mid:]

        return X_early, X_late

    def _get_cache_key(self, X_early) -> str:
        """Generate cache key for stage1 predictions."""
        if isinstance(X_early, pd.DataFrame):
            # Use hash of values and index
            return str(hash((X_early.values.tobytes(), str(X_early.index.tolist()))))
        else:
            return str(hash(X_early.tobytes()))

    def set_stage1_cache(self, X_early, predictions: np.ndarray) -> None:
        """Cache stage1 predictions for latency optimization.

        Parameters
        ----------
        X_early : array-like
            Early features used for stage1 prediction
        predictions : array-like
            Stage1 predictions to cache

        """
        cache_key = self._get_cache_key(X_early)
        self._stage1_cache[cache_key] = np.asarray(predictions)

    def clear_stage1_cache(self) -> None:
        """Clear all cached stage1 predictions."""
        self._stage1_cache.clear()

    def _get_cached_stage1_pred(self, X_early) -> np.ndarray | None:
        """Retrieve cached stage1 predictions if available."""
        cache_key = self._get_cache_key(X_early)
        return self._stage1_cache.get(cache_key)

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the two-stage model."""
        pass

    @abstractmethod
    def predict_stage1(self, X):
        """Make predictions using only early features (stage1)."""
        pass
