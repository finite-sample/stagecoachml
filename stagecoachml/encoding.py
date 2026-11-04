"""Adapters for estimators whose predictions form reusable representations."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


def fit_component(estimator: Any, X: Any, y: Any, sample_weight: Any) -> Any:
    """Fit a component, forwarding weights explicitly when supplied."""
    if sample_weight is None:
        return estimator.fit(X, y)
    return estimator.fit(X, y, sample_weight=sample_weight)


class PredictionEncoder(TransformerMixin, BaseEstimator):
    """Expose prediction scores as a two-dimensional representation.

    Args:
        estimator: A scikit-learn predictor, cloned before fitting.
        method: ``"auto"`` chooses ``predict_proba`` for classifiers and
            ``predict`` for regressors. Explicit ``"decision_function"`` is
            also supported; margins are never converted into probabilities.

    Attributes:
        estimator_: Fitted predictor.
        method_: Resolved prediction method.
        classes_: Class order when the predictor exposes classes.
    """

    def __init__(self, estimator: Any, *, method: str = "auto") -> None:
        self.estimator = estimator
        self.method = method

    def fit(
        self, X: Any, y: Any, sample_weight: np.ndarray | None = None
    ) -> "PredictionEncoder":
        """Fit the predictor on upstream data.

        Args:
            X: Upstream feature matrix.
            y: Training targets.
            sample_weight: Optional weights forwarded to the predictor.

        Returns:
            The fitted encoder.

        Raises:
            ValueError: If the selected prediction method is unsupported.
        """
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        method = self.method
        if method == "auto":
            method = "predict_proba" if is_classifier(self.estimator) else "predict"
        if method not in {"predict", "predict_proba", "decision_function"}:
            raise ValueError(
                "method must be auto, predict, predict_proba or decision_function"
            )
        if not hasattr(self.estimator, method):
            raise ValueError(f"The encoder estimator does not support {method}")
        validate_data(self, X, y, dtype=None, skip_check_array=True)
        estimator = fit_component(clone(self.estimator), X, y, sample_weight)
        self.estimator_ = estimator
        self.method_ = method
        if hasattr(estimator, "classes_"):
            self.classes_ = np.asarray(estimator.classes_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Produce numeric scores in stable column order.

        Args:
            X: Upstream feature matrix.

        Returns:
            A matrix of predictions, probabilities, or raw margins.
        """
        check_is_fitted(self, ["estimator_", "method_"])
        validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        values = np.asarray(getattr(self.estimator_, self.method_)(X))
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return check_array(values, dtype="numeric")
