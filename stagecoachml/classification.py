"""Two-stage classification estimator."""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import StagecoachBase
from ._validation import validate_cv_parameter, validate_estimator


class StagecoachClassifier(StagecoachBase, ClassifierMixin):
    """Two-stage classifier for staggered feature arrival.

    This estimator handles scenarios where features arrive in batches at different
    times. It trains a stage1 model on early features and a stage2 model that can
    use late features plus (optionally) the stage1 prediction.

    Parameters
    ----------
    stage1_estimator : estimator
        Sklearn classifier for early features. Must support predict_proba or
        decision_function for probability estimation.
    stage2_estimator : estimator
        Sklearn classifier for late features (and optionally stage1 prediction).
        Must support predict_proba.
    early_features : list of str, optional
        Column names for early features. If None, uses first half of columns.
    late_features : list of str, optional
        Column names for late features. If None, uses second half of columns.
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
    classes_ : ndarray of shape (n_classes,)
        Class labels

    """

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
        super().__init__(
            stage1_estimator=stage1_estimator,
            stage2_estimator=stage2_estimator,
            early_features=early_features,
            late_features=late_features,
            use_stage1_pred_as_feature=use_stage1_pred_as_feature,
            inner_cv=inner_cv,
            random_state=random_state,
        )

    def _validate_classifier_requirements(self):
        """Validate that estimators support required methods for classification."""
        # Stage1 must support probability estimation
        if not (
            hasattr(self.stage1_estimator, "predict_proba")
            or hasattr(self.stage1_estimator, "decision_function")
        ):
            raise ValueError("stage1_estimator must implement predict_proba or decision_function")

        # Stage2 must support predict_proba for final probabilities
        if not hasattr(self.stage2_estimator, "predict_proba"):
            raise ValueError("stage2_estimator must implement predict_proba")

    def fit(self, X, y, sample_weight=None):
        """Fit the two-stage classifier.

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
        check_classification_targets(y)
        validate_estimator(self.stage1_estimator, "classifier")
        validate_estimator(self.stage2_estimator, "classifier")
        validate_cv_parameter(self.inner_cv)
        self._validate_classifier_requirements()

        # Store classes
        self.classes_ = np.unique(y)

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
                if hasattr(self.stage1_estimator, "predict_proba"):
                    stage1_pred = cross_val_predict(
                        clone(self.stage1_estimator),
                        X_early,
                        y,
                        cv=self.inner_cv,
                        method="predict_proba",
                    )
                    # For binary classification, use positive class probability
                    if len(self.classes_) == 2:
                        stage1_pred = stage1_pred[:, 1]
                else:
                    # Use decision_function
                    stage1_pred = cross_val_predict(
                        clone(self.stage1_estimator),
                        X_early,
                        y,
                        cv=self.inner_cv,
                        method="decision_function",
                    )
            else:
                # Use in-sample predictions (may overfit)
                stage1_pred = self._get_stage1_pred_values(X_early)

        # Prepare stage2 inputs
        if self.use_stage1_pred_as_feature:
            if isinstance(X_late, pd.DataFrame):
                X_stage2 = X_late.copy()
                if stage1_pred.ndim == 1:
                    X_stage2["_stage1_pred"] = stage1_pred
                else:
                    # Multi-class case - add all class probabilities
                    for i, class_label in enumerate(self.classes_):
                        X_stage2[f"_stage1_pred_class_{class_label}"] = stage1_pred[:, i]
            elif stage1_pred.ndim == 1:
                X_stage2 = np.column_stack([X_late, stage1_pred.reshape(-1, 1)])
            else:
                X_stage2 = np.column_stack([X_late, stage1_pred])
        else:
            X_stage2 = X_late

        # Fit stage2
        self.stage2_estimator_ = clone(self.stage2_estimator)
        if sample_weight is not None:
            self.stage2_estimator_.fit(X_stage2, y, sample_weight=sample_weight)
        else:
            self.stage2_estimator_.fit(X_stage2, y)

        return self

    def _get_stage1_pred_values(self, X_early):
        """Get stage1 prediction values for use as features."""
        if hasattr(self.stage1_estimator_, "predict_proba"):
            proba = self.stage1_estimator_.predict_proba(X_early)
            if len(self.classes_) == 2:
                return proba[:, 1]  # Positive class probability
            else:
                return proba  # All class probabilities
        else:
            # Use decision function
            return self.stage1_estimator_.decision_function(X_early)

    def predict_stage1(self, X):
        """Predict classes using only early features (stage1).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Stage1 class predictions

        """
        proba = self.predict_stage1_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(proba >= 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_stage1_proba(self, X):
        """Predict class probabilities using only early features (stage1).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_proba : array of shape (n_samples,) or (n_samples, n_classes)
            Stage1 probability predictions. For binary classification,
            returns probabilities for the positive class.

        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, _ = self._split_features(X)

        # Check cache first
        cached_pred = self._get_cached_stage1_pred(X_early)
        if cached_pred is not None:
            return cached_pred

        if hasattr(self.stage1_estimator_, "predict_proba"):
            proba = self.stage1_estimator_.predict_proba(X_early)
            if len(self.classes_) == 2:
                return proba[:, 1]  # Positive class probability
            else:
                return proba
        else:
            # Use decision function and convert to probabilities
            decision = self.stage1_estimator_.decision_function(X_early)
            if len(self.classes_) == 2:
                # Binary case: sigmoid transform
                return 1 / (1 + np.exp(-decision))
            else:
                # Multi-class case: softmax
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        """Predict classes using both stages (full prediction).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Final class predictions

        """
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities using both stages (full prediction).

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            Final probability predictions

        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, X_late = self._split_features(X)

        # Get stage1 predictions
        if self.use_stage1_pred_as_feature:
            stage1_pred = self._get_stage1_pred_values(X_early)

            # Prepare stage2 inputs
            if isinstance(X_late, pd.DataFrame):
                X_stage2 = X_late.copy()
                if stage1_pred.ndim == 1:
                    X_stage2["_stage1_pred"] = stage1_pred
                else:
                    for i, class_label in enumerate(self.classes_):
                        X_stage2[f"_stage1_pred_class_{class_label}"] = stage1_pred[:, i]
            elif stage1_pred.ndim == 1:
                X_stage2 = np.column_stack([X_late, stage1_pred.reshape(-1, 1)])
            else:
                X_stage2 = np.column_stack([X_late, stage1_pred])
        else:
            X_stage2 = X_late

        # Get final predictions from stage2
        return self.stage2_estimator_.predict_proba(X_stage2)

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
