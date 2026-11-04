"""Deterministic data for staged-model behavior tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["u", "v", "item", "fresh"])
    y = 2 * X.u - X.v + X.item * X.u + 0.2 * X.fresh
    return X, y.to_numpy()
