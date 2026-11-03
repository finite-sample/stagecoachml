"""Representation alignment, ownership and freshness validation."""

import numpy as np
import pytest

from stagecoachml import EncodedState


def test_state_copies_data_and_retains_metadata():
    values = np.ones((2, 3))
    ids = np.array([10, 20])
    times = np.array([100.0, 110.0])
    state = EncodedState(values, "encoder", ids, times)
    values[:] = 0
    ids[:] = 30
    times[:] = 0
    np.testing.assert_array_equal(state.values, np.ones((2, 3)))
    np.testing.assert_array_equal(state.entity_ids, [10, 20])
    np.testing.assert_array_equal(state.feature_times, [100, 110])
    for array in [state.values, state.entity_ids, state.feature_times]:
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder_id": ""},
        {"values": [[np.nan]]},
        {"entity_ids": [1]},
        {"entity_ids": [1, 1]},
        {"entity_ids": [1, None]},
        {"feature_times": [1]},
        {"feature_times": [1, np.inf]},
    ],
)
def test_invalid_state(kwargs):
    args = {"values": np.ones((2, 3)), "encoder_id": "a"} | kwargs
    with pytest.raises(ValueError, match=r"encoder_id|NaN|entity_ids|Timestamps"):
        EncodedState(**args)


@pytest.mark.parametrize("indices", [[-1], [2], [0.0], [True], [[0]], []])
def test_invalid_mapping(indices):
    state = EncodedState(np.ones((2, 3)), "a")
    with pytest.raises(ValueError, match="Row indices"):
        state.select(indices)


def test_entity_mapping_requires_known_entities():
    state = EncodedState(np.ones((2, 3)), "a", entity_ids=np.array([1, 2]))
    np.testing.assert_array_equal(state.rows_for([2, 2, 1]), [1, 1, 0])
    for ids in [[3], [], [[1]]]:
        with pytest.raises(ValueError, match="entity_ids"):
            state.rows_for(ids)
    with pytest.raises(ValueError, match="no entity_ids"):
        EncodedState(np.ones((2, 3)), "a").rows_for([1])


def test_freshness_is_checked_after_candidate_mapping():
    state = EncodedState(np.ones((2, 3)), "a", feature_times=np.array([10, 20]))
    state.select([1, 0, 1], as_of=[25, 15, 25], max_age=5)
    with pytest.raises(ValueError, match="after the request"):
        state.select([1], as_of=19)
    with pytest.raises(ValueError, match="exceeds max_age"):
        state.select([0], as_of=16, max_age=5)
    for age in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="nonnegative"):
            state.select([0], as_of=20, max_age=age)
    with pytest.raises(ValueError, match="requires as_of"):
        state.select([0], max_age=5)
    with pytest.raises(ValueError, match="finite Unix"):
        state.select([0], as_of=np.nan)
    with pytest.raises(ValueError, match="feature_times"):
        EncodedState(np.ones((2, 3)), "a").select([0], as_of=10)
    scalar = EncodedState(np.ones((2, 3)), "a", feature_times=10)
    np.testing.assert_array_equal(scalar.feature_times, [10, 10])
