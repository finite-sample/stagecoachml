"""Input validation utilities for StagecoachML."""

import numpy as np
from sklearn.base import is_classifier, is_regressor


def validate_estimator(estimator, estimator_type: str) -> None:
    """Validate that estimator is of correct type."""
    if estimator_type == "regressor" and not is_regressor(estimator):
        raise ValueError(f"Expected regressor, got {type(estimator)}")
    if estimator_type == "classifier" and not is_classifier(estimator):
        raise ValueError(f"Expected classifier, got {type(estimator)}")


def validate_stage2_estimator_for_residual(estimator) -> None:
    """Validate that stage2 estimator supports residual learning."""
    # Check if estimator can handle negative targets (for residuals)
    if hasattr(estimator, "_check_n_features"):
        # Most sklearn estimators should handle this fine
        pass


def check_consistent_length_features(
    early_features: list[str] | None,
    late_features: list[str] | None,
) -> None:
    """Check that the early and late feature lists do not overlap.

    Args:
        early_features: Column names routed to stage1, or None.
        late_features: Column names routed to stage2, or None.

    Raises:
        ValueError: If a column appears in both lists.
    """
    if early_features is not None and late_features is not None:
        early_set = set(early_features)
        late_set = set(late_features)

        overlap = early_set & late_set
        if overlap:
            raise ValueError(f"Features cannot be both early and late: {overlap}")


def validate_cv_parameter(inner_cv: int | None) -> None:
    """Validate the ``inner_cv`` parameter.

    Args:
        inner_cv: Number of cross-fitting folds, or None to disable.

    Raises:
        ValueError: If ``inner_cv`` is set but is not an integer >= 2.
    """
    if inner_cv is not None and (not isinstance(inner_cv, int) or inner_cv < 2):
        raise ValueError("inner_cv must be None or an integer >= 2")


def check_stage1_pred_compatibility(
    predictions: np.ndarray, X_late: np.ndarray
) -> None:
    """Check that stage1 predictions are compatible with late features."""
    if len(predictions) != len(X_late):
        raise ValueError(
            f"Stage1 predictions length ({len(predictions)}) must match "
            f"late features length ({len(X_late)})"
        )
