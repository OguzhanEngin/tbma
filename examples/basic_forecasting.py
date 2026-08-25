"""Minimal runnable TBMA example."""

import numpy as np
import pandas as pd

from tbma import TBMA


def main() -> None:
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame(
        {
            "lag_1": np.sin(np.arange(n) / 6.0),
            "lag_7": np.cos(np.arange(n) / 9.0),
        },
        index=dates,
    )
    y = 0.8 * X["lag_1"] - 0.3 * X["lag_7"]

    model = TBMA(
        ma_order=range(1, 8),
        n_estimators=50,
        rf_params={"max_depth": 6, "min_samples_leaf": 2},
        random_state=42,
    ).fit(X, y, seasonal_period=7)

    forecast = model.predict(X.iloc[-10:], horizon=3)
    features = model.generate_features(X.iloc[-10:])

    print(forecast)
    print(features.iloc[:, :5])


if __name__ == "__main__":
    main()
