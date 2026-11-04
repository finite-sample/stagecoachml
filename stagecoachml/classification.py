"""Classification with reusable upstream representations."""

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._base import StagecoachBase
from .state import EncodedState


class StagecoachClassifier(ClassifierMixin, StagecoachBase):
    """Compose an upstream encoder with an online probabilistic classifier.

    Args:
        encoder: A transformer exposing fit/transform, a PredictionEncoder, or
            a fitted transformer wrapped in sklearn.frozen.FrozenEstimator.
        estimator: Online classifier exposing fit, predict and predict_proba.
        upstream_features: Explicit upstream column names or integer positions.
        online_features: Explicit online column names or integer positions.
            Use an empty list when only representations are needed.
        cv: Optional fold count or splitter for PredictionEncoder cross-fitting.
            Rows never held out are excluded from downstream training. None fits
            both stages on the supplied rows; supervised encoders may overfit.

    Attributes:
        encoder_: Fitted upstream transformer.
        estimator_: Fitted online classifier.
        predictor_: Encoder-free online serving artifact.
        classes_: Downstream class labels in probability-column order.
        training_indices_: Rows used to fit the downstream classifier.
        encoder_id_: Unique identifier for this fitted representation contract.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
        *,
        groups: Any = None,
    ) -> "StagecoachClassifier":
        """Fit the encoder and online classifier.

        Args:
            X: Full training feature matrix.
            y: Single-output classification labels.
            sample_weight: Optional nonnegative weights, forwarded to both trainable
                components and sliced consistently within cross-fitting folds.
            groups: Optional group labels for the supplied cross-validation splitter.

        Returns:
            The fitted classifier.
        """
        self._fit(X, y, sample_weight, groups)
        return self

    def predict_proba_from_state(
        self,
        X_online: Any,
        state: EncodedState,
        *,
        state_indices: Any = None,
        as_of: Any = None,
        max_age: float | None = None,
    ) -> np.ndarray:
        """Predict probabilities without accessing the upstream encoder.

        Args:
            X_online: Online features only, one row per candidate.
            state: State produced by this fitted model.
            state_indices: Optional representation row per candidate.
            as_of: Optional request timestamps in Unix seconds.
            max_age: Optional maximum feature age in seconds.

        Returns:
            Probabilities in ``classes_`` order, including two columns for binary tasks.
        """
        check_is_fitted(self, "predictor_")
        return self.predictor_.predict_proba(
            X_online, state, state_indices=state_indices, as_of=as_of, max_age=max_age
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Run both stages and return final probabilities.

        Args:
            X: Full feature matrix with the fitted training schema.

        Returns:
            Probabilities in ``classes_`` order.
        """
        check_is_fitted(self, "predictor_")
        X_upstream, X_online = self._split(self._full_input(X, reset=False))
        return self.predict_proba_from_state(X_online, self.encode(X_upstream))
