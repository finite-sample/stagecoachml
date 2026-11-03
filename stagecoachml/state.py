"""Portable representations shared between preparation and online scoring."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array


def row_indices(indices: object, size: int) -> np.ndarray:
    """Validate an explicit, nonempty positional row mapping."""
    result = np.asarray(indices)
    if result.ndim != 1 or result.size == 0 or result.dtype.kind not in "iu":
        raise ValueError("Row indices must be a nonempty one-dimensional integer array")
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError("Row indices are outside the available rows")
    return result


def timestamps(values: object, size: int) -> np.ndarray:
    """Normalize Unix timestamps in seconds to one finite value per row."""
    result = np.asarray(values, dtype=float)
    if result.ndim == 0:
        result = np.full(size, result.item())
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "Timestamps must be finite Unix seconds, one per row or scalar"
        )
    return result.copy()


@dataclass(frozen=True, eq=False)
class EncodedState:
    """A versioned, dense representation batch with optional entity and time metadata.

    Arrays are copied and made read-only. Store this object alongside its fitted
    model using ordinary Python serialization. The encoder identifier binds state
    to a particular model fit, including after serialization.

    Args:
        values: Finite numeric matrix, one row per entity or observation.
        encoder_id: Identifier of the fitted encoder that produced these values.
        entity_ids: Optional unique, nonmissing identifiers in representation order.
        feature_times: Optional feature cutoff times in Unix seconds, scalar or
            one per representation. These describe input freshness, not write time.
    """

    values: np.ndarray
    encoder_id: str
    entity_ids: np.ndarray | None = None
    feature_times: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = check_array(self.values, dtype="numeric", copy=True)
        if not isinstance(self.encoder_id, str) or not self.encoder_id:
            raise ValueError("encoder_id must be a nonempty string")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        if self.entity_ids is not None:
            ids = np.asarray(self.entity_ids, dtype=object).copy()
            if ids.shape != (len(values),):
                raise ValueError("entity_ids must have one entry per representation")
            index = pd.Index(ids)
            if index.hasnans or not index.is_unique:
                raise ValueError("entity_ids must be unique and nonmissing")
            ids.setflags(write=False)
            object.__setattr__(self, "entity_ids", ids)
        if self.feature_times is not None:
            times = timestamps(self.feature_times, len(values))
            times.setflags(write=False)
            object.__setattr__(self, "feature_times", times)

    def rows_for(self, entity_ids: object) -> np.ndarray:
        """Map candidate entity identifiers to representation rows, preserving order.

        Args:
            entity_ids: One entity identifier per candidate; repeats are allowed.

        Returns:
            Integer row indices suitable for ``predict_from_state``.

        Raises:
            ValueError: If state has no entity identifiers or a candidate is missing.
        """
        if self.entity_ids is None:
            raise ValueError("State has no entity_ids")
        requested = np.asarray(entity_ids, dtype=object)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("Requested entity_ids must be a nonempty vector")
        indices = pd.Index(self.entity_ids).get_indexer(requested)
        if np.any(indices < 0):
            raise ValueError(
                "No representation is available for one or more entity_ids"
            )
        return indices

    def select(
        self,
        indices: object,
        *,
        as_of: object = None,
        max_age: float | None = None,
    ) -> np.ndarray:
        """Select rows and optionally enforce feature freshness.

        Args:
            indices: Positional representation rows, one per candidate.
            as_of: Request times in Unix seconds, scalar or one per candidate.
                Providing this rejects representations from the future.
            max_age: Maximum allowed age in seconds; requires ``as_of`` and
                stored feature times. The boundary is inclusive.

        Returns:
            Representation rows aligned with the candidates.

        Raises:
            ValueError: If indices, timestamps, or freshness constraints are invalid.
        """
        rows = row_indices(indices, len(self.values))
        if max_age is not None:
            if not np.isfinite(max_age) or max_age < 0:
                raise ValueError("max_age must be finite and nonnegative")
            if as_of is None:
                raise ValueError("max_age requires as_of")
        if as_of is not None:
            if self.feature_times is None:
                raise ValueError("Freshness checks require feature_times in state")
            age = timestamps(as_of, len(rows)) - self.feature_times[rows]
            if np.any(age < 0):
                raise ValueError("State contains features from after the request time")
            if max_age is not None and np.any(age > max_age):
                raise ValueError("State exceeds max_age; refresh it before scoring")
        return self.values[rows]
