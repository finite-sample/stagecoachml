"""Regression with reusable upstream representations."""

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from ._base import StagecoachBase


class StagecoachRegressor(RegressorMixin, StagecoachBase):
    """Compose an upstream encoder with an online regressor.

    Args:
        encoder: Transformer, PredictionEncoder, or frozen fitted transformer.
        estimator: Online regressor exposing fit and predict.
        upstream_features: Explicit upstream column names or integer positions.
        online_features: Explicit online column names or integer positions.
        cv: Optional fold count or splitter for PredictionEncoder cross-fitting.
        residual: If True, require one encoded prediction column, train the online
            estimator on its residual, and add that prediction back when serving.

    Attributes:
        encoder_: Fitted upstream transformer.
        estimator_: Fitted online regressor.
        predictor_: Encoder-free online serving artifact.
        training_indices_: Rows used to fit the downstream regressor.
        encoder_id_: Unique identifier for this fitted representation contract.
    """

    def __init__(
        self,
        encoder: Any,
        estimator: Any,
        *,
        upstream_features: list[str] | list[int],
        online_features: list[str] | list[int],
        cv: Any = None,
        residual: bool = False,
    ) -> None:
        super().__init__(
            encoder,
            estimator,
            upstream_features=upstream_features,
            online_features=online_features,
            cv=cv,
        )
        self.residual = residual

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
        *,
        groups: Any = None,
    ) -> "StagecoachRegressor":
        """Fit the encoder and online regressor.

        Args:
            X: Full training feature matrix.
            y: Single-output regression targets.
            sample_weight: Optional nonnegative weights, forwarded to both trainable
                components and sliced consistently within cross-fitting folds.
            groups: Optional group labels for the supplied cross-validation splitter.

        Returns:
            The fitted regressor.
        """
        self._fit(X, y, sample_weight, groups)
        return self
