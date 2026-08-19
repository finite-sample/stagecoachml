"""Two-stage classification estimator."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import Matrix, StagecoachBase, Target
from ._validation import validate_cv_parameter, validate_estimator


class StagecoachClassifier(StagecoachBase, ClassifierMixin):
    """Two-stage classifier for staggered feature arrival.

    This estimator handles scenarios where features arrive in batches at different
    times. It trains a stage1 model on early features and a stage2 model that can
    use late features plus (optionally) the stage1 prediction.

    Args:
        stage1_estimator: Sklearn classifier for the early features. Must
            support ``predict_proba`` or ``decision_function``.
        stage2_estimator: Sklearn classifier for the late features (and
            optionally the stage1 prediction). Must support ``predict_proba``.
        early_features: Column names for the early features. If None, the first
            half of the columns is used.
        late_features: Column names for the late features. If None, the second
            half of the columns is used.
        use_stage1_pred_as_feature: If True, the stage1 prediction is included
            as an input to stage2.
        inner_cv: Number of folds for cross-fitting the stage1 predictions
            during training. Helps avoid overfitting when the stage1
            prediction is used as a stage2 feature.
        random_state: Random state for reproducibility.

    Attributes:
        stage1_estimator_: Fitted stage1 estimator.
        stage2_estimator_: Fitted stage2 estimator.
        classes_: Class labels, of shape ``(n_classes,)``.
    """

    def __init__(
        self,
        stage1_estimator: BaseEstimator,
        stage2_estimator: BaseEstimator,
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

    def _validate_classifier_requirements(self) -> None:
        """Validate that estimators support required methods for classification."""
        # Stage1 must support probability estimation
        if not (
            hasattr(self.stage1_estimator, "predict_proba")
            or hasattr(self.stage1_estimator, "decision_function")
        ):
            raise ValueError(
                "stage1_estimator must implement predict_proba or decision_function"
            )

        # Stage2 must support predict_proba for final probabilities
        if not hasattr(self.stage2_estimator, "predict_proba"):
            raise ValueError("stage2_estimator must implement predict_proba")

    def fit(
        self,
        X: Matrix,
        y: Target,
        sample_weight: np.ndarray | None = None,
    ) -> "StagecoachClassifier":
        """Fit the two-stage classifier.

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
        stage1_pred = (
            self._stage1_training_pred(X_early, y)
            if self.use_stage1_pred_as_feature
            else None
        )
        X_stage2 = self._build_stage2_input(X_late, stage1_pred)

        # Fit stage2
        self.stage2_estimator_ = clone(self.stage2_estimator)
        if sample_weight is not None:
            self.stage2_estimator_.fit(X_stage2, y, sample_weight=sample_weight)
        else:
            self.stage2_estimator_.fit(X_stage2, y)

        return self

    def _stage1_training_pred(self, X_early: Matrix, y: Target) -> np.ndarray:
        """Produce the stage1 prediction that stage2 trains against.

        Args:
            X_early: Early features of the training data.
            y: Training targets, needed for cross-fitting.

        Returns:
            Stage1 scores: cross-fitted when ``inner_cv`` is set, in-sample
            (and therefore optimistic) otherwise.
        """
        if self.inner_cv is None:
            return self._get_stage1_pred_values(X_early)

        if not hasattr(self.stage1_estimator, "predict_proba"):
            return np.asarray(
                cross_val_predict(
                    clone(self.stage1_estimator),
                    X_early,
                    y,
                    cv=self.inner_cv,
                    method="decision_function",
                )
            )

        proba = np.asarray(
            cross_val_predict(
                clone(self.stage1_estimator),
                X_early,
                y,
                cv=self.inner_cv,
                method="predict_proba",
            )
        )
        # For binary classification, the positive-class column carries all the
        # information; keeping both columns would make stage2 collinear.
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def _build_stage2_input(
        self,
        X_late: Matrix,
        stage1_pred: np.ndarray | None,
    ) -> Matrix:
        """Append the stage1 score(s) to the late features.

        Args:
            X_late: Late features.
            stage1_pred: Stage1 scores, or None to pass the late features
                through untouched.

        Returns:
            The design matrix stage2 is fitted on and predicts from.
        """
        if stage1_pred is None:
            return X_late

        if isinstance(X_late, pd.DataFrame):
            X_stage2 = X_late.copy()
            if stage1_pred.ndim == 1:
                X_stage2["_stage1_pred"] = stage1_pred
            else:
                # Multi-class case - add all class probabilities
                for i, class_label in enumerate(self.classes_):
                    X_stage2[f"_stage1_pred_class_{class_label}"] = stage1_pred[:, i]
            return X_stage2

        if stage1_pred.ndim == 1:
            return np.column_stack([X_late, stage1_pred.reshape(-1, 1)])
        return np.column_stack([X_late, stage1_pred])

    def _get_stage1_pred_values(self, X_early: Matrix) -> np.ndarray:
        """Get stage1 prediction values for use as features."""
        if hasattr(self.stage1_estimator_, "predict_proba"):
            proba = np.asarray(self.stage1_estimator_.predict_proba(X_early))
            if len(self.classes_) == 2:
                return proba[:, 1]  # Positive class probability
            return proba  # All class probabilities
        # Use decision function
        return np.asarray(self.stage1_estimator_.decision_function(X_early))

    def predict_stage1(self, X: Matrix) -> np.ndarray:
        """Predict classes using only early features (stage1).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Stage1 class predictions of shape ``(n_samples,)``.
        """
        proba = self.predict_stage1_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(proba >= 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_stage1_proba(self, X: Matrix) -> np.ndarray:
        """Predict class probabilities using only early features (stage1).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Stage1 probability predictions, of shape ``(n_samples,)`` for
            binary problems (the positive class) and
            ``(n_samples, n_classes)`` otherwise.
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
            return proba
        # Use decision function and convert to probabilities
        decision = self.stage1_estimator_.decision_function(X_early)
        if len(self.classes_) == 2:
            # Binary case: sigmoid transform
            return 1 / (1 + np.exp(-decision))
        # Multi-class case: softmax
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: Matrix) -> np.ndarray:
        """Predict classes using both stages (full prediction).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Final class predictions of shape ``(n_samples,)``.
        """
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Matrix) -> np.ndarray:
        """Predict class probabilities using both stages (full prediction).

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Final probability predictions of shape
            ``(n_samples, n_classes)``.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_early, X_late = self._split_features(X)

        # Get stage1 predictions
        stage1_pred = (
            self._get_stage1_pred_values(X_early)
            if self.use_stage1_pred_as_feature
            else None
        )
        X_stage2 = self._build_stage2_input(X_late, stage1_pred)

        # Get final predictions from stage2
        return self.stage2_estimator_.predict_proba(X_stage2)

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
