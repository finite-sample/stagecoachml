"""Two-stage regression estimator."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import Matrix, StagecoachBase, Target
from ._validation import (
    validate_cv_parameter,
    validate_estimator,
    validate_stage2_estimator_for_residual,
)


class StagecoachRegressor(StagecoachBase, RegressorMixin):
    """Two-stage regressor for staggered feature arrival.

    This estimator handles scenarios where features arrive in batches at different
    times. It trains a stage1 model on early features and a stage2 model that can
    use late features plus (optionally) the stage1 prediction.

    Args:
        stage1_estimator: Sklearn regressor for the early features.
        stage2_estimator: Sklearn regressor for the late features (and
            optionally the stage1 prediction).
        early_features: Column names for the early features. If None, the first
            half of the columns is used.
        late_features: Column names for the late features. If None, the second
            half of the columns is used.
        residual: If True, stage2 predicts ``y - stage1_pred``; if False, it
            predicts ``y`` directly.
        use_stage1_pred_as_feature: If True, the stage1 prediction is included
            as an input to stage2.
        inner_cv: Number of folds for cross-fitting the stage1 predictions
            during training. Helps avoid overfitting when the stage1
            prediction is used as a stage2 feature.
        random_state: Random state for reproducibility.

    Attributes:
        stage1_estimator_: Fitted stage1 estimator.
        stage2_estimator_: Fitted stage2 estimator.
    """

    def __init__(
        self,
        stage1_estimator: BaseEstimator,
        stage2_estimator: BaseEstimator,
        early_features: list[str] | None = None,
        late_features: list[str] | None = None,
        residual: bool = True,
        use_stage1_pred_as_feature: bool = True,
        inner_cv: int | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            stage1_estimator=stage1_estimator,
            stage2_estimator=stage2_estimator,
            early_features=early_features,
            late_features=late_features,
            use_stage1_pred_as_feature=use_stage1_pred_as_feature,
            inner_cv=inner_cv,
            random_state=random_state,
        )
        self.residual = residual

    def fit(
        self,
        X: Matrix,
        y: Target,
        sample_weight: np.ndarray | None = None,
    ) -> "StagecoachRegressor":
        """Fit the two-stage regressor.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``.
            sample_weight: Per-sample weights of shape ``(n_samples,)``.

        Returns:
            The fitted estimator.
        """
        # Validation - validate features first to preserve DataFrame info
        self._validate_features(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        validate_estimator(self.stage1_estimator, "regressor")
        validate_estimator(self.stage2_estimator, "regressor")
        validate_cv_parameter(self.inner_cv)

        if self.residual:
            validate_stage2_estimator_for_residual(self.stage2_estimator)

        # Split features
        X_early, X_late = self._split_features(X)

        # Fit stage1
        self.stage1_estimator_ = clone(self.stage1_estimator)
        if sample_weight is not None:
            self.stage1_estimator_.fit(X_early, y, sample_weight=sample_weight)
        else:
            self.stage1_estimator_.fit(X_early, y)

        # Get stage1 predictions for stage2 training
        stage1_pred = None
        if self.use_stage1_pred_as_feature:
            if self.inner_cv is not None:
                # Cross-fitted predictions to avoid overfitting
                stage1_pred = np.asarray(
                    cross_val_predict(
                        clone(self.stage1_estimator), X_early, y, cv=self.inner_cv
                    )
                )
            else:
                # Use in-sample predictions (may overfit)
                stage1_pred = np.asarray(self.stage1_estimator_.predict(X_early))

        X_stage2 = self._build_stage2_input(X_late, stage1_pred)

        # Prepare stage2 targets
        if self.residual and self.use_stage1_pred_as_feature:
            y_stage2 = y - stage1_pred
        else:
            y_stage2 = y

        # Fit stage2
        self.stage2_estimator_ = clone(self.stage2_estimator)
        if sample_weight is not None:
            self.stage2_estimator_.fit(X_stage2, y_stage2, sample_weight=sample_weight)
        else:
            self.stage2_estimator_.fit(X_stage2, y_stage2)

        return self

    def _build_stage2_input(
        self,
        X_late: Matrix,
        stage1_pred: np.ndarray | None,
    ) -> Matrix:
        """Append the stage1 prediction to the late features.

        Args:
            X_late: Late features.
            stage1_pred: Stage1 predictions, or None to pass the late features
                through untouched.

        Returns:
            The design matrix stage2 is fitted on and predicts from.
        """
        if stage1_pred is None:
            return X_late
        if isinstance(X_late, pd.DataFrame):
            X_stage2 = X_late.copy()
            X_stage2["_stage1_pred"] = stage1_pred
            return X_stage2
        return np.column_stack([X_late, stage1_pred.reshape(-1, 1)])

    def predict_stage1(self, X: Matrix) -> np.ndarray:
        """Predict using only early features (stage1).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Stage1 predictions of shape ``(n_samples,)``.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, _ = self._split_features(X)

        # Check cache first
        cached_pred = self._get_cached_stage1_pred(X_early)
        if cached_pred is not None:
            return cached_pred

        return self.stage1_estimator_.predict(X_early)

    def predict(self, X: Matrix) -> np.ndarray:
        """Predict using both stages (full prediction).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Final predictions of shape ``(n_samples,)``.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        _, X_late = self._split_features(X)

        # Get stage1 predictions
        stage1_pred = self.predict_stage1(X)

        X_stage2 = self._build_stage2_input(
            X_late, stage1_pred if self.use_stage1_pred_as_feature else None
        )

        # Get stage2 predictions
        stage2_pred = self.stage2_estimator_.predict(X_stage2)

        # Combine predictions
        if self.residual and self.use_stage1_pred_as_feature:
            return stage1_pred + stage2_pred
        return stage2_pred

    def _more_tags(self) -> dict[str, object]:
        return {
            "requires_y": True,
            "requires_fit": True,
            "X_types": ["2darray"],
            "allow_nan": False,
            "stateless": False,
            "binary_only": False,
            "requires_positive_X": False,
        }
