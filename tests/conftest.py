from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    n = 80
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    x1 = np.sin(np.arange(n) / 5.0)
    x2 = np.cos(np.arange(n) / 7.0)
    x3 = (np.arange(n) % 4).astype(float)
    y = 10.0 + 0.7 * x1 - 0.3 * x2 + 0.1 * x3 + np.arange(n) * 0.02
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=dates)
    return X, pd.Series(y, index=dates), dates
