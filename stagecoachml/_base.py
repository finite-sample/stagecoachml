"""Composition and input validation for reusable encoders and online estimators."""

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.frozen import FrozenEstimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
    validate_data,
)

from ._training import cross_fit, take_rows
from .encoding import PredictionEncoder, fit_component
from .serving import OnlinePredictor
from .state import EncodedState

if TYPE_CHECKING:
    from collections.abc import Callable


def feature_positions(X: Any, features: Any) -> list[int]:
    """Resolve explicit names or positions against the training schema."""
    if not isinstance(features, (list, tuple)):
        raise ValueError(
            "Feature selectors must be lists or tuples of names or positions"
        )
    if len(set(features)) != len(features):
        raise ValueError("Feature selectors must not contain duplicates")
    if not features:
        return []
    if all(isinstance(f, str) for f in features):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Named feature selectors require a DataFrame during fit")
        if not set(features).issubset(X.columns):
            raise ValueError("Selected features are missing from the input")
        return X.columns.get_indexer(features).tolist()
    if all(isinstance(f, int) and not isinstance(f, bool) for f in features):
        if min(features) < 0 or max(features) >= X.shape[1]:
            raise ValueError("Feature positions are outside the input columns")
        return list(features)
    raise ValueError("Feature selectors must contain only strings or only integers")


class StagecoachBase(BaseEstimator):
    """Shared training and encoding implementation."""

    feature_names_in_: np.ndarray

    def __init__(
        self,
        encoder: Any,
        estimator: Any,
        *,
        upstream_features: list[str] | list[int],
        online_features: list[str] | list[int],
        cv: Any = None,
    ) -> None:
        self.encoder = encoder
        self.estimator = estimator
        self.upstream_features = upstream_features
        self.online_features = online_features
        self.cv = cv

    def _full_input(self, X: Any, *, reset: bool) -> Any:
        if not isinstance(X, pd.DataFrame):
            X = cast("Callable[..., np.ndarray]", check_array)(X, dtype=None)
        elif not X.columns.is_unique:
            raise ValueError("DataFrame columns must be unique")
        elif not reset and self.input_columns_ is not None:
            if not set(self.input_columns_).issubset(X.columns):
                raise ValueError(
                    "Full prediction input is missing fitted feature columns"
                )
            X = X.loc[:, list(self.input_columns_)]
        cast("Callable[..., Any]", validate_data)(
            self, X, reset=reset, dtype=None, skip_check_array=True
        )
        if reset:
            self.input_columns_ = (
                tuple(X.columns) if isinstance(X, pd.DataFrame) else None
            )
        return X

    def _split(self, X: Any) -> tuple[Any, Any]:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.upstream_positions_], X.iloc[
                :, self.online_positions_
            ]
        return X[:, self.upstream_positions_], X[:, self.online_positions_]

    def _fit(self, X: Any, y: Any, sample_weight: Any, groups: Any) -> "StagecoachBase":
        # Invalidate a previous fit before any validation or component training.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        X = self._full_input(X, reset=True)
        _, y = cast("Callable[..., Any]", check_X_y)(X, y, dtype=None)
        self._validate_components(y)
        self.upstream_positions_ = feature_positions(X, self.upstream_features)
        self.online_positions_ = feature_positions(X, self.online_features)
        if not self.upstream_positions_:
            raise ValueError("At least one upstream feature is required")
        X_upstream, X_online = self._split(X)
        self.upstream_names_ = (
            tuple(X_upstream.columns) if isinstance(X_upstream, pd.DataFrame) else None
        )
        self.online_names_ = (
            tuple(X_online.columns) if isinstance(X_online, pd.DataFrame) else None
        )
        weights = self._weights(sample_weight, len(y))
        if groups is not None and np.asarray(groups).shape != (len(y),):
            raise ValueError("groups must have one entry per training row")
        if groups is not None and self.cv is None:
            raise ValueError("groups requires an explicit cv splitter")
        encoder = fit_component(clone(self.encoder), X_upstream, y, weights)
        values = self._representations(encoder.transform(X_upstream), len(y))
        rows = np.arange(len(y))
        if self.cv is not None:
            values, rows = cross_fit(
                self.encoder,
                X_upstream,
                y,
                cv=self.cv,
                classifier=is_classifier(self),
                groups=groups,
                sample_weight=weights,
                reference=values,
                classes=getattr(encoder, "classes_", None),
            )
        target = self._target(y[rows], values)
        online = check_array(X_online, ensure_min_features=0, dtype="numeric")
        design = np.column_stack([values, take_rows(online, rows)])
        estimator = fit_component(
            clone(self.estimator),
            design,
            target,
            None if weights is None else weights[rows],
        )
        if is_classifier(self):
            if not hasattr(estimator, "predict_proba"):
                raise ValueError("The online classifier must implement predict_proba")
            if not np.array_equal(estimator.classes_, np.unique(y)):
                raise ValueError(
                    "Held-out training rows must contain every target class"
                )
            self.classes_ = np.asarray(estimator.classes_)
        self.encoder_ = encoder
        self.estimator_ = estimator
        self.encoder_id_ = uuid4().hex
        self.n_encoded_features_ = values.shape[1]
        self.training_indices_ = rows
        self.predictor_ = OnlinePredictor(
            estimator=estimator,
            encoder_id=self.encoder_id_,
            n_encoded_features=self.n_encoded_features_,
            n_online_features=len(self.online_positions_),
            online_names=self.online_names_,
            residual=getattr(self, "residual", False),
        )
        return self

    def _validate_components(self, y: np.ndarray) -> None:
        if not isinstance(getattr(self, "residual", False), (bool, np.bool_)):
            raise ValueError("residual must be a boolean")
        if is_classifier(self):
            check_classification_targets(y)
            if not is_classifier(self.estimator):
                raise ValueError("estimator must be a classifier")
        elif not is_regressor(self.estimator):
            raise ValueError("estimator must be a regressor")
        if not hasattr(self.encoder, "transform"):
            raise ValueError(
                "encoder must implement fit and transform; "
                "use PredictionEncoder for scores"
            )
        if self.cv is not None and not isinstance(self.encoder, PredictionEncoder):
            raise ValueError(
                "Cross-fitting is supported only for PredictionEncoder; "
                "freeze other encoders"
            )
        if self.cv is not None and isinstance(self.encoder.estimator, FrozenEstimator):
            raise ValueError("A frozen prediction estimator cannot be cross-fitted")

    def _target(self, y: np.ndarray, values: np.ndarray) -> np.ndarray:
        if getattr(self, "residual", False):
            if values.shape[1] != 1:
                raise ValueError(
                    "Residual regression requires a single prediction column"
                )
            return y - values[:, 0]
        return y

    @staticmethod
    def _weights(sample_weight: Any, size: int) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (size,) or not np.all(np.isfinite(weights)):
            raise ValueError(
                "sample_weight must be finite with one entry per training row"
            )
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError(
                "sample_weight must be nonnegative with positive total weight"
            )
        return weights

    @staticmethod
    def _representations(values: Any, size: int) -> np.ndarray:
        values = check_array(values, dtype="numeric")
        if len(values) != size:
            raise ValueError("encoder.transform must preserve the number of input rows")
        return values

    def encode(
        self,
        X_upstream: Any,
        *,
        entity_ids: Any = None,
        feature_times: Any = None,
    ) -> EncodedState:
        """Prepare reusable state using only upstream inputs.

        Args:
            X_upstream: Upstream columns. DataFrame columns are selected by fitted
                name; arrays must contain only upstream columns in fitted order.
            entity_ids: Optional unique identifiers, one per upstream row.
            feature_times: Optional input cutoff timestamps in Unix seconds.

        Returns:
            Versioned representations for downstream serving.

        Raises:
            ValueError: If input or representation dimensions differ from training.
        """
        check_is_fitted(self, "predictor_")
        if isinstance(X_upstream, pd.DataFrame) and self.upstream_names_ is not None:
            if not set(self.upstream_names_).issubset(X_upstream.columns):
                raise ValueError("Upstream input is missing fitted feature columns")
            if not X_upstream.columns.is_unique:
                raise ValueError("DataFrame columns must be unique")
            X_upstream = X_upstream.loc[:, list(self.upstream_names_)]
        else:
            X_upstream = cast("Callable[..., np.ndarray]", check_array)(
                X_upstream, dtype=None
            )
            if X_upstream.shape[1] != len(self.upstream_positions_):
                raise ValueError("Upstream input width differs from the fitted schema")
            if self.upstream_names_ is not None:
                X_upstream = pd.DataFrame(
                    X_upstream, columns=pd.Index(self.upstream_names_)
                )
        values = self._representations(
            self.encoder_.transform(X_upstream), len(X_upstream)
        )
        if values.shape[1] != self.n_encoded_features_:
            raise ValueError("Encoder output width changed after fitting")
        return EncodedState(values, self.encoder_id_, entity_ids, feature_times)

    def predict_from_state(
        self,
        X_online: Any,
        state: EncodedState,
        *,
        state_indices: Any = None,
        as_of: Any = None,
        max_age: float | None = None,
    ) -> np.ndarray:
        """Predict using prepared state without executing or accessing the encoder.

        Args:
            X_online: Online features only, one row per candidate.
            state: Representations produced by this fitted model.
            state_indices: Representation row per candidate; defaults to row alignment.
            as_of: Optional request timestamps in Unix seconds.
            max_age: Optional maximum feature age in seconds.

        Returns:
            Final predictions, one per candidate.
        """
        check_is_fitted(self, "predictor_")
        return self.predictor_.predict(
            X_online, state, state_indices=state_indices, as_of=as_of, max_age=max_age
        )

    def predict(self, X: Any) -> np.ndarray:
        """Run both stages for ordinary scikit-learn evaluation.

        Args:
            X: Full feature matrix with the fitted training schema.

        Returns:
            Final predictions, one per input row.
        """
        check_is_fitted(self, "predictor_")
        X_upstream, X_online = self._split(self._full_input(X, reset=False))
        return self.predict_from_state(X_online, self.encode(X_upstream))
