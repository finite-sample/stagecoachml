"""Two-stage regression estimator."""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import StagecoachBase
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

    Parameters
    ----------
    stage1_estimator : estimator
        Sklearn regressor for early features
    stage2_estimator : estimator
        Sklearn regressor for late features (and optionally stage1 prediction)
    early_features : list of str, optional
        Column names for early features. If None, uses first half of columns.
    late_features : list of str, optional
        Column names for late features. If None, uses second half of columns.
    residual : bool, default=True
        If True, stage2 learns to predict y - stage1_pred (residual).
        If False, stage2 learns to predict y directly.
    use_stage1_pred_as_feature : bool, default=True
        If True, stage1 prediction is included as input to stage2.
    inner_cv : int, optional
        Number of folds for cross-fitting stage1 predictions during training.
        Helps avoid overfitting when using stage1 predictions as stage2 features.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    stage1_estimator_ : estimator
        Fitted stage1 estimator
    stage2_estimator_ : estimator
        Fitted stage2 estimator

    """

    def __init__(
        self,
        stage1_estimator,
        stage2_estimator,
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

    def fit(self, X, y, sample_weight=None):
        """Fit the two-stage regressor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        self : object
            Fitted estimator

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
        if self.use_stage1_pred_as_feature:
            if self.inner_cv is not None:
                # Cross-fitted predictions to avoid overfitting
                stage1_pred = cross_val_predict(
                    clone(self.stage1_estimator), X_early, y, cv=self.inner_cv
                )
            else:
                # Use in-sample predictions (may overfit)
                stage1_pred = self.stage1_estimator_.predict(X_early)

        # Prepare stage2 inputs
        if self.use_stage1_pred_as_feature:
            if isinstance(X_late, pd.DataFrame):
                X_stage2 = X_late.copy()
                X_stage2["_stage1_pred"] = stage1_pred
            else:
                X_stage2 = np.column_stack([X_late, stage1_pred.reshape(-1, 1)])
        else:
            X_stage2 = X_late

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

    def predict_stage1(self, X):
        """Predict using only early features (stage1).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Stage1 predictions

        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, _ = self._split_features(X)

        # Check cache first
        cached_pred = self._get_cached_stage1_pred(X_early)
        if cached_pred is not None:
            return cached_pred

        return self.stage1_estimator_.predict(X_early)

    def predict(self, X):
        """Predict using both stages (full prediction).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Final predictions

        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, X_late = self._split_features(X)

        # Get stage1 predictions
        stage1_pred = self.predict_stage1(X)

        # Prepare stage2 inputs
        if self.use_stage1_pred_as_feature:
            if isinstance(X_late, pd.DataFrame):
                X_stage2 = X_late.copy()
                X_stage2["_stage1_pred"] = stage1_pred
            else:
                X_stage2 = np.column_stack([X_late, stage1_pred.reshape(-1, 1)])
        else:
            X_stage2 = X_late

        # Get stage2 predictions
        stage2_pred = self.stage2_estimator_.predict(X_stage2)

        # Combine predictions
        if self.residual and self.use_stage1_pred_as_feature:
            return stage1_pred + stage2_pred
        else:
            return stage2_pred

    def _more_tags(self):
        return {
            "requires_y": True,
            "requires_fit": True,
            "X_types": ["2darray"],
            "allow_nan": False,
            "stateless": False,
            "binary_only": False,
            "requires_positive_X": False,
        }
