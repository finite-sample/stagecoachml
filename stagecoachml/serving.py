"""An encoder-free artifact for online prediction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array

from .state import EncodedState


@dataclass
class OnlinePredictor:
    """Serve a fitted downstream estimator without loading its upstream encoder.

    Obtain this object from ``model.predictor_`` after fitting. Serialize it with
    the prepared state for deployment in an independent process.

    Args:
        estimator: Fitted downstream estimator.
        encoder_id: Accepted fitted encoder identifier.
        n_encoded_features: Expected representation width.
        n_online_features: Expected number of online input columns.
        online_names: Fitted online column names, if trained on a DataFrame.
        residual: Whether to add the first representation column to predictions.
    """

    estimator: Any
    encoder_id: str
    n_encoded_features: int
    n_online_features: int
    online_names: tuple | None = None
    residual: bool = False

    def _design(
        self,
        X_online: Any,
        state: EncodedState,
        state_indices: Any,
        as_of: Any,
        max_age: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(state, EncodedState) or state.encoder_id != self.encoder_id:
            raise ValueError(
                "State belongs to a different encoder fit; "
                "regenerate or use its matching predictor"
            )
        if state.values.shape[1] != self.n_encoded_features:
            raise ValueError("State representation width does not match the predictor")
        if isinstance(X_online, pd.DataFrame) and self.online_names is not None:
            if not X_online.columns.is_unique:
                raise ValueError("DataFrame columns must be unique")
            if not set(self.online_names).issubset(X_online.columns):
                raise ValueError("Online input is missing fitted feature columns")
            X_online = X_online.loc[:, list(self.online_names)]
        online = check_array(X_online, ensure_min_features=0, dtype="numeric")
        if online.shape[1] != self.n_online_features:
            raise ValueError("Online input width differs from the fitted schema")
        rows = np.arange(len(online)) if state_indices is None else state_indices
        values = state.select(rows, as_of=as_of, max_age=max_age)
        if state_indices is None and len(state.values) != len(online):
            raise ValueError(
                "Unequal state and candidate row counts require state_indices"
            )
        if len(values) != len(online):
            raise ValueError("state_indices must have one entry per candidate")
        return np.column_stack([values, online]), values

    def predict(
        self,
        X_online: Any,
        state: EncodedState,
        *,
        state_indices: Any = None,
        as_of: Any = None,
        max_age: float | None = None,
    ) -> np.ndarray:
        """Score online inputs using an explicitly supplied representation batch.

        Args:
            X_online: Numeric online features, one row per candidate.
            state: State from the matching encoder fit.
            state_indices: Optional representation row per candidate.
            as_of: Optional request times in Unix seconds, scalar or per candidate.
            max_age: Optional maximum feature age in seconds.

        Returns:
            Predicted labels or regression values.
        """
        design, values = self._design(X_online, state, state_indices, as_of, max_age)
        prediction = np.asarray(self.estimator.predict(design))
        return prediction + values[:, 0] if self.residual else prediction

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(
        self,
        X_online: Any,
        state: EncodedState,
        *,
        state_indices: Any = None,
        as_of: Any = None,
        max_age: float | None = None,
    ) -> np.ndarray:
        """Return probabilities in the fitted downstream estimator's class order.

        Args:
            X_online: Numeric online features, one row per candidate.
            state: State from the matching encoder fit.
            state_indices: Optional representation row per candidate.
            as_of: Optional request times in Unix seconds, scalar or per candidate.
            max_age: Optional maximum feature age in seconds.

        Returns:
            Probabilities with one column per class, including binary problems.
        """
        design, _ = self._design(X_online, state, state_indices, as_of, max_age)
        return np.asarray(self.estimator.predict_proba(design))
