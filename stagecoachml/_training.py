"""Cross-fitting for representations with a stable prediction interpretation."""

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import check_cv

from .encoding import PredictionEncoder, fit_component
from .state import row_indices


def take_rows(X: Any, rows: np.ndarray) -> Any:
    """Select positional rows without discarding DataFrame column names."""
    return X.iloc[rows] if hasattr(X, "iloc") else X[rows]


def cross_fit(
    encoder: PredictionEncoder,
    X: Any,
    y: np.ndarray,
    *,
    cv: Any,
    classifier: bool,
    groups: Any,
    sample_weight: np.ndarray | None,
    reference: np.ndarray,
    classes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build out-of-fold scores, excluding rows never held out by the splitter."""
    values = np.empty_like(reference, dtype=float)
    seen = np.zeros(len(y), dtype=bool)
    splitter: Any = check_cv(cv, y=y, classifier=classifier)
    for train_indices, test_indices in splitter.split(X, y, groups):
        train = row_indices(train_indices, len(y))
        test = row_indices(test_indices, len(y))
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError(
                "Cross-fitting folds must not contain duplicate row indices"
            )
        if np.intersect1d(train, test).size or np.any(seen[test]):
            raise ValueError(
                "Cross-fitting requires disjoint train/test rows "
                "and unique test coverage"
            )
        weights = None if sample_weight is None else sample_weight[train]
        fitted = fit_component(clone(encoder), take_rows(X, train), y[train], weights)
        if classes is not None and not np.array_equal(fitted.classes_, classes):
            raise ValueError(
                "Every encoder training fold must contain the same classes"
            )
        fold_values = fitted.transform(take_rows(X, test))
        if fold_values.shape != (len(test), reference.shape[1]):
            raise ValueError(
                "Cross-fitted prediction columns do not match the full encoder"
            )
        values[test] = fold_values
        seen[test] = True
    rows = np.flatnonzero(seen)
    if not rows.size:
        raise ValueError("Cross-fitting produced no held-out rows")
    return values[rows], rows
