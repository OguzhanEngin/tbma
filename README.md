# TBMA

`tbma` is a Python implementation of **Tree-Based Moving Average (TBMA)** for supervised temporal feature learning and standalone time-series forecasting.

TBMA fits a random forest to prepared autoregressive predictors and a one-step target. The learned tree partitions define supervised neighborhoods. Within each tree, unique in-bag observations are used to build node-specific, one-sided pooled moving-average (PMA) curves. Seasonally aligned values from those curves can be returned as the full tree-level TBMA representation or aggregated into standalone forecasts.

The package intentionally keeps data preparation outside the estimator. It does **not** create lags, scale or difference series, construct multi-step target tables, infer seasonality, or choose moving-average orders from dataset metadata.

## Installation

After publication:

```bash
python -m pip install tbma
```

For development from a clone:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required. The release CI matrix tests Python 3.10–3.14 on GitHub-hosted Linux, Windows, and macOS runners. The published wheel is pure Python (`py3-none-any`); platform support still depends on compatible NumPy, pandas, SciPy, and scikit-learn builds for the user's Python/platform combination.

## Platform support

The reusable TBMA package contains no OS-specific path handling, shell commands,
or compiled extension code of its own. Repository text files are normalized with
`.gitattributes`, persistence uses `pathlib`, and the package tests include nested
paths containing spaces and Unicode characters.

Every CI and release run gates the core package on this matrix:

| OS runner | Python versions | What is tested |
| --- | --- | --- |
| `ubuntu-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | built wheel install, full tests/coverage, general example |
| `windows-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | built wheel install, full tests/coverage, general example |
| `macos-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | built wheel install, full tests/coverage, general example |

The GitHub-only paper reproduction workflow is separately exercised on Linux,
Windows, and macOS with Python 3.13 and its CatBoost/OpenPyXL dependencies. A
release is not built or published unless all of those jobs pass.

## Quick start

```python
import numpy as np
import pandas as pd

from tbma import TBMA

n = 120
dates = pd.date_range("2024-01-01", periods=n, freq="D")

X = pd.DataFrame(
    {
        "lag_1": np.sin(np.arange(n) / 6.0),
        "lag_7": np.cos(np.arange(n) / 11.0),
    },
    index=dates,
)
y = 0.8 * X["lag_1"] - 0.3 * X["lag_7"]

model = TBMA(
    ma_order=range(1, 8),
    n_estimators=100,
    rf_params={"max_depth": 6, "min_samples_leaf": 2},
    random_state=42,
)

# Frequency is inferred from the DatetimeIndex. The seasonal period is supplied
# explicitly and does not determine the moving-average orders.
model.fit(X, y, seasonal_period=7)

# Standalone TBMA readout.
forecast = model.predict(X.iloc[-20:], horizon=3)

# Full TBMA feature representation: one feature per tree when feature_window=1.
features = model.generate_features(X.iloc[-20:])
```

`predict()` returns `horizon_1`, `horizon_2`, ... through the requested forecast horizon. `generate_features()` defaults to the complete tree-level TBMA representation.

## Moving-average orders

`ma_order` is supplied directly by the caller and is measured in observation periods.

Use one positive integer to apply the same order to every tree:

```python
model = TBMA(ma_order=7)
```

Or supply a sequence of candidate orders. Each tree deterministically draws one candidate using `random_state`:

```python
model = TBMA(ma_order=range(1, 8), random_state=42)
```

If domain knowledge suggests a useful range related to seasonality, it can be constructed explicitly:

```python
seasonal_period = 24
ma_orders = range(1, seasonal_period + 1)

model = TBMA(ma_order=ma_orders)
model.fit(X, y, seasonal_period=seasonal_period)
```

## Full TBMA feature representation

The paper defines the learned representation by concatenating tree-level, seasonally aligned PMA values. That is the default behavior of `generate_features()`:

```python
features = model.generate_features(
    X_new,
    feature_window=1,
)
```

With `K` trees and a feature window of length `w_g`, the result has `K * w_g` columns.

For example, with 100 trees and `feature_window=3`:

```python
features = model.generate_features(X_new, feature_window=3)
assert features.shape[1] == 300
```

`feature_window` is independent of the forecast `horizon`. It controls how many seasonally aligned PMA positions each tree contributes to the feature representation.

## Optional feature summaries

Summary methods are selected when features are generated, not when the TBMA model is fitted.

### Quantiles

```python
quantile_features = model.generate_features(
    X_new,
    feature_window=3,
    summary_method="quantile",
)
```

For each feature position, this returns the 25th, 50th, and 75th percentiles across trees.

### PCA

```python
pca_features = model.generate_features(
    X_new,
    feature_window=3,
    summary_method="pca",
    pca_components=2,
    pca_include_mean=True,
)
```

`pca_components` can be a positive integer or a variance threshold strictly between 0 and 1:

```python
pca_features = model.generate_features(
    X_new,
    feature_window=3,
    summary_method="pca",
    pca_components=0.75,
    pca_include_mean=False,
)
```

PCA bases are fitted from the fitted **training** TBMA representation for the requested `feature_window` and cached on the model. Later calls transform new rows with the same basis, so training and test features remain in the same PCA coordinate system.

## Dates, frequency, and seasonality

TBMA's pooled moving averages are date-aware, so every row needs a timestamp.

If `X` has a `DatetimeIndex`, `dates` can be omitted:

```python
model.fit(X, y)
pred = model.predict(X_future, horizon=2)
```

Otherwise pass timestamps explicitly:

```python
model.fit(X, y, dates=train_dates, frequency="D")
pred = model.predict(X_future, dates=future_dates, horizon=2)
```

When `frequency` is omitted, TBMA attempts to infer it from at least three unique regular training timestamps.

Supported pandas-style frequency families include:

- seconds, minutes, and hours, such as `"s"`, `"10min"`, and `"2h"`;
- calendar days and business days, such as `"D"`, `"2D"`, and `"B"`;
- weekly frequencies, including anchored forms such as `"W-MON"`;
- month starts and ends, such as `"MS"` and `"ME"`;
- quarter starts and ends, such as `"QS"` and `"QE-DEC"`;
- year starts and ends, such as `"YS"` and `"YE-DEC"`.

Daily and weekly calendar shifts preserve local wall-clock time for timezone-aware data across daylight-saving transitions.

`seasonal_period` controls seasonal alignment of the PMA reference positions and is separate from `ma_order`:

```python
# Non-seasonal/default reference period.
model.fit(X, y)

# Weekly seasonality for daily data.
model.fit(X, y, seasonal_period=7)

# Daily seasonality for hourly data.
model.fit(X_hourly, y_hourly, seasonal_period=24)
```

## Input expectations

`fit(X, y, ...)` accepts a pandas DataFrame or a two-dimensional NumPy-compatible array for `X` and a one-dimensional target for `y`.

The package expects predictors to already contain the autoregressive or other covariates you want the random forest to use. Lag construction, scaling, seasonal differencing, missing-value policy, categorical encoding, train/test splitting, and other preprocessing belong in the calling application or pipeline.

When fitting with a DataFrame, later DataFrames must contain exactly the same feature columns. Their order may differ; TBMA restores fit-time order internally.

## Standalone forecasts

The standalone readout averages the tree-level TBMA values, matching the mean aggregation defined by the method:

```python
forecast = model.predict(X_new, horizon=6)
```

The forecast horizon is specified at prediction time and is not fixed during `fit()`.

## Random-forest configuration

`n_estimators` and `random_state` are first-class TBMA parameters. Additional `sklearn.ensemble.RandomForestRegressor` options can be supplied through `rf_params`:

```python
model = TBMA(
    ma_order=[2, 4, 6],
    n_estimators=200,
    random_state=7,
    rf_params={
        "max_depth": 8,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "n_jobs": -1,
    },
)
```

Values for `n_estimators` or `random_state` inside `rf_params` are ignored in favor of the explicit TBMA arguments.

`random_state` must be an integer in the NumPy/scikit-learn seed range `0 <= random_state <= 2**32 - 1`.

## Persistence

Fitting never writes files automatically. Save a fitted estimator explicitly:

```python
model.save("tbma.pkl")
loaded = TBMA.load("tbma.pkl")
```

Pickle can execute code while loading. Only load files from trusted sources.

## Paper reproduction workflow (GitHub repository only)

The manuscript evaluation pipeline is intentionally **not part of the PyPI distribution**. The source repository contains a separate `paper_reproduction/` directory, a repository-root `dataset_info.xlsx`, and expects the `.tsf` files under `./Datasets/`. That workflow reproduces the paper preprocessing, TBMA feature augmentation, standalone TBMA readout, downstream Random Forest / Multi-task Elastic Net / CatBoost comparisons, 10-seed evaluation, and MASE/statistical summaries.

The reproduction code and its dependencies are excluded from both the wheel and sdist. Installing `tbma` therefore installs only the reusable library. The repository workflow checkpoints each completed model result to CSV during long runs. From a GitHub/source checkout, see `paper_reproduction/README.md` for the exact workbook columns, dependencies, commands, model settings, checkpoint behavior, and outputs.

## Method background

The package implements **“TBMA: Temporal Feature Learning for Multi-Series Forecasting.”** The method uses random-forest-defined autoregressive neighborhoods together with node-specific, one-sided pooled moving-average curves. The same learned representation can be returned as tree-level temporal features or averaged into a standalone forecast.

The library deliberately exposes the method separately from dataset preparation and evaluation code: callers provide prepared predictors, a one-step target, timestamps, explicit MA order(s), and an optional seasonal period.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest --cov=tbma --cov-branch --cov-report=term-missing --cov-fail-under=100
python -m build
python -m twine check dist/*
```

See `VALIDATION.md` for numerical-equivalence and package checks, and `PUBLISHING.md` for the release procedure.

## Authors and citation

TBMA is authored by **Mustafa Baydoğan**, **Berk Görgülü**, and **Oğuzhan Engin**.

For research use, cite the software using the repository's `CITATION.cff` metadata and cite the associated manuscript **“TBMA: Temporal Feature Learning for Multi-Series Forecasting.”** The canonical source repository is [https://github.com/OguzhanEngin/tbma](https://github.com/OguzhanEngin/tbma).

## License

TBMA is released under the **MIT License**. See `LICENSE` for the complete license text.

Project repository: [https://github.com/OguzhanEngin/tbma](https://github.com/OguzhanEngin/tbma)  
Issue tracker: [https://github.com/OguzhanEngin/tbma/issues](https://github.com/OguzhanEngin/tbma/issues)
