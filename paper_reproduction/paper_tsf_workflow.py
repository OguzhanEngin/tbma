"""Reproduce the paper evaluation from repository-local TSF datasets.

This module is repository-only reproducibility code.  It is intentionally not
part of the installable :mod:`tbma` package or either PyPI distribution.

The default layout is::

    dataset_info.xlsx
    Datasets/
        *.tsf
    paper_reproduction/
        paper_tsf_workflow.py

Dataset-specific settings are read from the ``repo_data`` worksheet in
``./dataset_info.xlsx``.  Rows with ``run == 1`` are evaluated.  The workbook
provides the displayed dataset name, TSF filename, forecast horizon,
autoregressive lookback ``L`` (``predetermined_lag``), and optional
integer-output/evaluation flags.  Other workbook columns are left untouched
and ignored by this paper workflow.

By default seeds 1 through 10 are run and paper-style MASE summaries are
written under ``./paper_results``.  The model hyperparameters remain fixed to
the manuscript settings unless this module is called programmatically with an
alternative :class:`PaperModelSettings` instance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNetCV

from tbma import TBMA

TSF_FREQUENCY_MAP = {
    "minutely": "min",
    "10_minutes": "10min",
    "half_hourly": "30min",
    "hourly": "h",
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QS",
    "yearly": "YS",
}

TSF_SEASONAL_PERIOD_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": round(365.25 / 7),
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

MODEL_ORDER = ["CB", "CB_TBMA", "MEN", "MEN_TBMA", "RF", "RF_TBMA", "TBMA"]
FEATURE_PAIRS = [("CB", "CB_TBMA"), ("MEN", "MEN_TBMA"), ("RF", "RF_TBMA")]


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific settings read from ``dataset_info.xlsx``."""

    name: str
    file_name: str
    horizon: int
    lookback: int
    integer_conversion: bool = False
    evaluate: bool = True


@dataclass(frozen=True)
class PaperModelSettings:
    """Model settings reported in the paper's reproducibility table."""

    tbma_n_estimators: int = 256
    tbma_max_depth: int = 12
    tbma_min_samples_leaf: int = 4
    tbma_max_features: float = 1 / 3
    tbma_ma_order: tuple[int, ...] = (1, 2, 3, 4, 5)
    feature_window: int = 1

    rf_n_estimators: int = 256
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 4
    rf_max_features: float = 1 / 3

    catboost_iterations: int = 512
    catboost_learning_rate: float = 0.025
    catboost_depth: int = 5
    catboost_min_child_samples: int = 32
    catboost_colsample_bylevel: float = 1 / 3

    men_l1_ratio: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    men_eps: float = 1e-4
    men_max_iter: int = 1000
    men_n_alphas: int = 20


@dataclass(frozen=True)
class TSFMetadata:
    """Metadata and series parsed from a TSF file."""

    attributes: pd.DataFrame
    value_column: str
    frequency: str | None
    horizon: int | None
    contains_missing: bool | None
    equal_length: bool | None
    date_attribute: str | None


@dataclass
class PreparedDataset:
    """Prepared global forecasting problem for one dataset."""

    name: str
    path: Path
    full: pd.DataFrame
    tbma_train: pd.DataFrame
    downstream_train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    actual_columns: list[str]
    reference_columns: list[str]
    series_means: pd.Series
    mase_scales: pd.Series
    horizon: int
    lookback: int
    frequency: str
    seasonal_period: int
    seasonal_diff_lag: int
    seasonal_differencing: bool
    seasonal_dominance_ratio: float
    integer_conversion: bool
    evaluate: bool


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid TSF boolean value: {value!r}")


def _parse_attribute(value: str, kind: str) -> Any:
    if kind == "numeric":
        number = float(value)
        return int(number) if number.is_integer() else number
    if kind == "string":
        return value
    if kind == "date":
        try:
            return pd.to_datetime(value, format="%Y-%m-%d %H-%M-%S").to_pydatetime()
        except ValueError:
            return pd.Timestamp(value).to_pydatetime()
    raise ValueError(f"Unsupported TSF attribute type: {kind!r}")


def read_tsf(path: str | Path, *, value_column: str = "series_value") -> TSFMetadata:
    """Read a Monash-style TSF file without adding a runtime dependency."""
    path = Path(path)
    attribute_names: list[str] = []
    attribute_types: list[str] = []
    attribute_values: dict[str, list[Any]] = {}
    all_series: list[np.ndarray] = []
    frequency: str | None = None
    horizon: int | None = None
    contains_missing: bool | None = None
    equal_length: bool | None = None
    date_attribute: str | None = None
    found_data = False

    with path.open("r", encoding="cp1252") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                parts = line.split()
                tag = parts[0].lower()
                if tag == "@attribute":
                    if len(parts) != 3:
                        raise ValueError(f"Invalid @attribute line in {path}: {line}")
                    name, kind = parts[1], parts[2].lower()
                    if kind not in {"numeric", "string", "date"}:
                        raise ValueError(f"Unsupported attribute type {kind!r} in {path}")
                    attribute_names.append(name)
                    attribute_types.append(kind)
                    attribute_values[name] = []
                    if kind == "date":
                        date_attribute = name
                elif tag == "@frequency":
                    if len(parts) != 2:
                        raise ValueError(f"Invalid @frequency line in {path}: {line}")
                    frequency = parts[1].lower()
                elif tag == "@horizon":
                    if len(parts) != 2:
                        raise ValueError(f"Invalid @horizon line in {path}: {line}")
                    horizon = int(parts[1])
                elif tag == "@missing":
                    if len(parts) != 2:
                        raise ValueError(f"Invalid @missing line in {path}: {line}")
                    contains_missing = _parse_bool(parts[1])
                elif tag == "@equallength":
                    if len(parts) != 2:
                        raise ValueError(f"Invalid @equallength line in {path}: {line}")
                    equal_length = _parse_bool(parts[1])
                elif tag == "@data":
                    if not attribute_names:
                        raise ValueError(f"{path} has @data before any @attribute lines")
                    found_data = True
                continue

            if not found_data:
                raise ValueError(f"Series data appears before @data in {path}")

            fields = line.split(":")
            if len(fields) != len(attribute_names) + 1:
                raise ValueError(
                    f"Expected {len(attribute_names)} attributes plus values in {path}, "
                    f"got {len(fields) - 1} attributes"
                )

            for name, kind, raw_value in zip(
                attribute_names, attribute_types, fields[:-1], strict=True
            ):
                attribute_values[name].append(_parse_attribute(raw_value, kind))

            values = [np.nan if item == "?" else float(item) for item in fields[-1].split(",")]
            if not values or np.isnan(values).all():
                raise ValueError(f"A series in {path} contains no observed values")
            all_series.append(np.asarray(values, dtype=float))

    if not found_data or not all_series:
        raise ValueError(f"No TSF series were found in {path}")

    attribute_values[value_column] = all_series
    frame = pd.DataFrame(attribute_values)
    actual_missing = any(np.isnan(values).any() for values in all_series)
    if contains_missing is False and actual_missing:
        raise ValueError(f"{path} declares @missing false but contains '?' values")

    return TSFMetadata(
        attributes=frame,
        value_column=value_column,
        frequency=frequency,
        horizon=horizon,
        contains_missing=contains_missing,
        equal_length=equal_length,
        date_attribute=date_attribute,
    )


def _series_identifier(frame: pd.DataFrame, value_column: str, date_attribute: str | None) -> pd.Series:
    if "series_name" in frame.columns:
        identifiers = frame["series_name"].astype(str)
    else:
        metadata_columns = [
            column
            for column in frame.columns
            if column not in {value_column, date_attribute}
        ]
        if metadata_columns:
            identifiers = frame[metadata_columns].astype(str).agg("|".join, axis=1)
        else:
            identifiers = pd.Series(
                [f"series_{idx}" for idx in range(len(frame))], index=frame.index
            )
    if identifiers.duplicated().any():
        raise ValueError("TSF series identifiers must be unique")
    return identifiers


def tsf_to_long(
    metadata: TSFMetadata,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Convert parsed TSF arrays to a tidy series/date/value table."""
    frame = metadata.attributes
    identifiers = _series_identifier(frame, metadata.value_column, metadata.date_attribute)
    rows: list[pd.DataFrame] = []

    for idx, identifier in identifiers.items():
        values = np.asarray(frame.at[idx, metadata.value_column], dtype=float)
        if np.isnan(values).any():
            raise ValueError(
                "The paper workflow does not define an imputation rule; TSF series "
                f"{identifier!r} contains missing values"
            )

        if metadata.date_attribute is None:
            start = pd.Timestamp("1900-01-01")
        else:
            start = pd.Timestamp(frame.at[idx, metadata.date_attribute])

        start = start.as_unit("s")
        dates = pd.date_range(
            start=start, periods=len(values), freq=frequency, unit="s"
        )
        rows.append(
            pd.DataFrame(
                {
                    "series": str(identifier),
                    "date": dates,
                    "y": values,
                }
            )
        )

    return pd.concat(rows, ignore_index=True).sort_values(
        ["series", "date"], ignore_index=True
    )


def _training_raw_rows(long_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    reverse_position = long_df.groupby("series", sort=False).cumcount(ascending=False)
    training = long_df.loc[reverse_position >= horizon].copy()
    if training.empty:
        raise ValueError("No training observations remain after holding out the horizon")
    return training


def _seasonal_dominance(training_raw: pd.DataFrame, seasonal_period: int) -> float:
    comparisons: list[bool] = []
    for _, group in training_raw.groupby("series", sort=False):
        values = group["y"]
        lag1 = values.autocorr(lag=1)
        seasonal = values.autocorr(lag=seasonal_period)
        if np.isfinite(lag1) and np.isfinite(seasonal):
            comparisons.append(abs(seasonal) > abs(lag1))
    return float(np.mean(comparisons)) if comparisons else 0.0


def _mase_scales(training_raw: pd.DataFrame, seasonal_period: int) -> pd.Series:
    scales: dict[str, float] = {}
    for series, group in training_raw.groupby("series", sort=False):
        values = group["y"].astype(float)
        seasonal_scale = values.diff(seasonal_period).abs().mean()
        if not np.isfinite(seasonal_scale) or seasonal_scale <= 0:
            seasonal_scale = values.diff(1).abs().mean()
        if not np.isfinite(seasonal_scale) or seasonal_scale <= 0:
            raise ValueError(
                f"Series {series!r} has no positive finite MASE scaling denominator"
            )
        scales[str(series)] = float(seasonal_scale)
    return pd.Series(scales, name="mase_scale", dtype=float)


def prepare_dataset(
    path: str | Path,
    *,
    lookback: int,
    horizon: int | None = None,
    frequency: str | None = None,
    seasonal_period: int | None = None,
    name: str | None = None,
    integer_conversion: bool = False,
    evaluate: bool = True,
) -> PreparedDataset:
    """Apply the paper's training-only preprocessing to one TSF dataset."""
    path = Path(path)
    if lookback < 1:
        raise ValueError("lookback must be positive")

    metadata = read_tsf(path)
    resolved_horizon = int(horizon if horizon is not None else metadata.horizon or 0)
    if resolved_horizon < 1:
        raise ValueError(
            f"{path.name} has no positive @horizon; provide a horizon override"
        )

    if frequency is None:
        if metadata.frequency not in TSF_FREQUENCY_MAP:
            raise ValueError(
                f"Unsupported or missing TSF frequency {metadata.frequency!r} for "
                f"{path.name}; provide a pandas frequency override"
            )
        resolved_frequency = TSF_FREQUENCY_MAP[metadata.frequency]
    else:
        resolved_frequency = frequency

    if seasonal_period is None:
        if metadata.frequency not in TSF_SEASONAL_PERIOD_MAP:
            raise ValueError(
                f"No default seasonal period for TSF frequency {metadata.frequency!r}; "
                "provide seasonal_period"
            )
        resolved_seasonal_period = TSF_SEASONAL_PERIOD_MAP[metadata.frequency]
    else:
        resolved_seasonal_period = int(seasonal_period)
    if resolved_seasonal_period < 1:
        raise ValueError("seasonal_period must be positive")

    long_df = tsf_to_long(metadata, frequency=resolved_frequency)
    training_raw = _training_raw_rows(long_df, resolved_horizon)
    series_means = training_raw.groupby("series", sort=False)["y"].mean().astype(float)
    series_means = series_means.mask(series_means == 0, 1.0).rename("series_mean")
    mase_scales = _mase_scales(training_raw, resolved_seasonal_period)

    dominance_ratio = _seasonal_dominance(training_raw, resolved_seasonal_period)
    seasonal_differencing = dominance_ratio >= 0.5
    seasonal_diff_lag = ceil(resolved_horizon / resolved_seasonal_period) * resolved_seasonal_period

    work = long_df.copy()
    work["scaled_y"] = work["y"] / work["series"].map(series_means)
    if seasonal_differencing:
        seasonal_reference = work.groupby("series", sort=False)["scaled_y"].shift(
            seasonal_diff_lag
        )
        work["model_y"] = work["scaled_y"] - seasonal_reference
    else:
        work["model_y"] = work["scaled_y"]

    feature_columns = [f"lag_{lag}" for lag in range(1, lookback + 1)]
    for lag, column in enumerate(feature_columns, start=1):
        work[column] = work.groupby("series", sort=False)["model_y"].shift(lag)

    target_columns: list[str] = []
    actual_columns: list[str] = []
    reference_columns: list[str] = []
    for step in range(resolved_horizon):
        target_column = f"target_{step + 1}"
        actual_column = f"actual_{step + 1}"
        reference_column = f"reference_{step + 1}"
        target_columns.append(target_column)
        actual_columns.append(actual_column)
        reference_columns.append(reference_column)
        work[target_column] = work.groupby("series", sort=False)["model_y"].shift(-step)
        work[actual_column] = work.groupby("series", sort=False)["y"].shift(-step)
        if seasonal_differencing:
            work[reference_column] = work.groupby("series", sort=False)["scaled_y"].shift(
                seasonal_diff_lag - step
            )
        else:
            work[reference_column] = 0.0

    required_columns = feature_columns + target_columns + actual_columns
    if seasonal_differencing:
        required_columns += reference_columns
    processed = work.dropna(subset=required_columns).copy()

    counts = processed.groupby("series", sort=False)["date"].transform("size")
    processed = processed.loc[counts > 1].copy()
    if processed.empty:
        raise ValueError(
            f"{path.name} has no usable series after lag/target construction; "
            "reduce lookback or check the dataset length"
        )

    reverse_position = processed.groupby("series", sort=False).cumcount(ascending=False)
    test = processed.loc[reverse_position == 0].copy()
    tbma_train = processed.loc[reverse_position >= 1].copy()

    downstream_reverse = tbma_train.groupby("series", sort=False).cumcount(ascending=False)
    downstream_train = tbma_train.loc[
        downstream_reverse >= resolved_horizon - 1
    ].copy()
    if downstream_train.empty:
        raise ValueError(
            f"{path.name} has no leakage-free downstream training rows; "
            "reduce lookback or horizon"
        )

    surviving_series = set(processed["series"])
    series_means = series_means.loc[
        [series for series in series_means.index if series in surviving_series]
    ]
    mase_scales = mase_scales.loc[
        [series for series in mase_scales.index if series in surviving_series]
    ]

    return PreparedDataset(
        name=name or path.stem,
        path=path,
        full=processed,
        tbma_train=tbma_train,
        downstream_train=downstream_train,
        test=test,
        feature_columns=feature_columns,
        target_columns=target_columns,
        actual_columns=actual_columns,
        reference_columns=reference_columns,
        series_means=series_means,
        mase_scales=mase_scales,
        horizon=resolved_horizon,
        lookback=lookback,
        frequency=resolved_frequency,
        seasonal_period=resolved_seasonal_period,
        seasonal_diff_lag=seasonal_diff_lag,
        seasonal_differencing=seasonal_differencing,
        seasonal_dominance_ratio=dominance_ratio,
        integer_conversion=bool(integer_conversion),
        evaluate=bool(evaluate),
    )


def _as_2d(prediction: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != horizon:
        raise ValueError(
            f"Expected predictions with {horizon} columns, got shape {values.shape}"
        )
    return values


def _reconstruct_predictions(prediction: np.ndarray, rows: pd.DataFrame, data: PreparedDataset) -> np.ndarray:
    values = _as_2d(prediction, data.horizon).copy()
    if data.seasonal_differencing:
        values += rows[data.reference_columns].to_numpy(dtype=float)
    means = rows["series"].map(data.series_means).to_numpy(dtype=float)[:, None]
    values *= means
    if data.integer_conversion:
        values = np.rint(values)
    return values


def _mase_values(actual: np.ndarray, prediction: np.ndarray, rows: pd.DataFrame, data: PreparedDataset) -> np.ndarray:
    actual_values = _as_2d(actual, data.horizon)
    predicted_values = _as_2d(prediction, data.horizon)
    mae = np.mean(np.abs(actual_values - predicted_values), axis=1)
    scales = rows["series"].map(data.mase_scales).to_numpy(dtype=float)
    return mae / scales


def _result_row(data: PreparedDataset, seed: int, model: str, prediction: np.ndarray) -> dict[str, Any]:
    actual = data.test[data.actual_columns].to_numpy(dtype=float)
    mase = _mase_values(actual, prediction, data.test, data)
    return {
        "dataset": data.name,
        "file_name": data.path.name,
        "evaluate": data.evaluate,
        "seed": seed,
        "model": model,
        "mean_MASE": float(np.mean(mase)),
        "median_MASE": float(np.median(mase)),
        "n_series": len(mase),
        "horizon": data.horizon,
        "lookback": data.lookback,
        "seasonal_period": data.seasonal_period,
        "seasonal_differencing": data.seasonal_differencing,
        "seasonal_dominance_ratio": data.seasonal_dominance_ratio,
    }


def _paper_models(
    seed: int, settings: PaperModelSettings, *, n_jobs: int
) -> dict[str, Any]:
    version_parts = sklearn_version.split(".")[:2]
    sklearn_major_minor = tuple(int(part) for part in version_parts)
    men_kwargs: dict[str, Any] = {
        "l1_ratio": list(settings.men_l1_ratio),
        "eps": settings.men_eps,
        "max_iter": settings.men_max_iter,
    }
    if sklearn_major_minor >= (1, 7):
        men_kwargs["alphas"] = settings.men_n_alphas
    else:
        men_kwargs["n_alphas"] = settings.men_n_alphas
    men = MultiTaskElasticNetCV(**men_kwargs)

    return {
        "CB": CatBoostRegressor(
            iterations=settings.catboost_iterations,
            learning_rate=settings.catboost_learning_rate,
            depth=settings.catboost_depth,
            min_child_samples=settings.catboost_min_child_samples,
            colsample_bylevel=settings.catboost_colsample_bylevel,
            loss_function="MultiRMSE",
            verbose=0,
            random_seed=seed,
            allow_writing_files=False,
            thread_count=n_jobs,
        ),
        "MEN": men,
        "RF": RandomForestRegressor(
            n_estimators=settings.rf_n_estimators,
            max_depth=settings.rf_max_depth,
            min_samples_leaf=settings.rf_min_samples_leaf,
            max_features=settings.rf_max_features,
            random_state=seed,
            n_jobs=n_jobs,
        ),
    }


def _fit_predict(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    horizon: int,
    *,
    drop_missing_train: bool = False,
) -> np.ndarray:
    train = X_train
    targets = y_train
    if drop_missing_train:
        complete = ~train.isna().any(axis=1)
        train = train.loc[complete]
        targets = targets[complete.to_numpy()]
        if train.empty:
            raise ValueError("No complete training rows remain for the downstream model")
        if X_test.isna().any().any():
            raise ValueError(
                "The downstream model cannot evaluate test rows with missing "
                "TBMA features"
            )

    model.fit(train.to_numpy(dtype=float), targets)
    return _as_2d(model.predict(X_test.to_numpy(dtype=float)), horizon)


def run_seed(
    data: PreparedDataset,
    seed: int,
    *,
    settings: PaperModelSettings | None = None,
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    """Run TBMA, base models, and TBMA-augmented models for one seed."""
    settings = settings or PaperModelSettings()

    tbma = TBMA(
        ma_order=settings.tbma_ma_order,
        n_estimators=settings.tbma_n_estimators,
        rf_params={
            "max_depth": settings.tbma_max_depth,
            "min_samples_leaf": settings.tbma_min_samples_leaf,
            "max_features": settings.tbma_max_features,
            "n_jobs": n_jobs,
        },
        random_state=seed,
    )
    tbma.fit(
        data.tbma_train[data.feature_columns],
        data.tbma_train[data.target_columns[0]],
        dates=data.tbma_train["date"],
        frequency=data.frequency,
        seasonal_period=data.seasonal_period,
    )

    full_tbma_features = tbma.generate_features(
        data.full[data.feature_columns],
        dates=data.full["date"],
        feature_window=settings.feature_window,
        summary_method=None,
    )
    test_tbma_prediction = tbma.predict(
        data.test[data.feature_columns],
        dates=data.test["date"],
        horizon=data.horizon,
    ).to_numpy(dtype=float)
    test_tbma_prediction = _reconstruct_predictions(test_tbma_prediction, data.test, data)
    results = [_result_row(data, seed, "TBMA", test_tbma_prediction)]

    base_train = data.downstream_train[data.feature_columns]
    base_test = data.test[data.feature_columns]
    augmented = pd.concat(
        [data.full[data.feature_columns], full_tbma_features], axis=1
    )
    augmented_train = augmented.loc[data.downstream_train.index]
    augmented_test = augmented.loc[data.test.index]
    y_train = data.downstream_train[data.target_columns].to_numpy(dtype=float)

    base_models = _paper_models(seed, settings, n_jobs=n_jobs)
    augmented_models = _paper_models(seed, settings, n_jobs=n_jobs)
    for model_name, base_model in base_models.items():
        base_prediction = _fit_predict(
            base_model,
            base_train,
            y_train,
            base_test,
            data.horizon,
        )
        base_prediction = _reconstruct_predictions(base_prediction, data.test, data)
        results.append(_result_row(data, seed, model_name, base_prediction))

        augmented_prediction = _fit_predict(
            augmented_models[model_name],
            augmented_train,
            y_train,
            augmented_test,
            data.horizon,
            drop_missing_train=model_name == "MEN",
        )
        augmented_prediction = _reconstruct_predictions(
            augmented_prediction, data.test, data
        )
        results.append(
            _result_row(data, seed, f"{model_name}_TBMA", augmented_prediction)
        )

    return results


def symmetric_percent_difference(first: float, second: float) -> float:
    """Return 100 * (first - second) divided by their arithmetic mean."""
    denominator = (first + second) / 2
    if denominator == 0:
        return np.nan
    return 100 * (first - second) / denominator


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_idx])
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted.tolist()


def _wilcoxon_table(dataset_means: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    pivot = dataset_means.pivot(index="dataset", columns="model", values="mean_MASE")
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for baseline, candidate in pairs:
        paired = pivot[[candidate, baseline]].dropna()
        if len(paired) < 2:
            statistic = np.nan
            p_value = np.nan
        else:
            result = wilcoxon(
                paired[candidate],
                paired[baseline],
                alternative="less",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        rows.append(
            {
                "comparison": f"{candidate} vs {baseline}",
                "test_statistic": statistic,
                "raw_p": p_value,
                "n_datasets": len(paired),
            }
        )
        raw_p.append(p_value)

    finite_indices = [idx for idx, value in enumerate(raw_p) if np.isfinite(value)]
    finite_adjusted = _holm_adjust([raw_p[idx] for idx in finite_indices])
    for row in rows:
        row["holm_adjusted_p"] = np.nan
    for idx, adjusted in zip(finite_indices, finite_adjusted, strict=True):
        rows[idx]["holm_adjusted_p"] = adjusted
    return pd.DataFrame(rows)


def summarize_results(per_seed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the dataset-level tables used by the paper-style evaluation."""
    dataset_means_all = (
        per_seed.groupby(["dataset", "model"], as_index=False)["mean_MASE"].mean()
    )
    if "evaluate" in per_seed.columns:
        evaluation_rows = per_seed.loc[per_seed["evaluate"].astype(bool)]
    else:
        evaluation_rows = per_seed
    dataset_means = (
        evaluation_rows.groupby(["dataset", "model"], as_index=False)["mean_MASE"]
        .mean()
    )
    if dataset_means.empty:
        raise ValueError("No datasets are marked for evaluation")
    table2 = dataset_means.pivot(index="dataset", columns="model", values="mean_MASE")
    table2 = table2.reindex(columns=[column for column in MODEL_ORDER if column in table2])

    feature_spd_rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []
    for baseline, enhanced in FEATURE_PAIRS:
        if baseline not in table2 or enhanced not in table2:
            continue
        for dataset in table2.index:
            feature_spd_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{baseline}->{enhanced}",
                    "SPD": symmetric_percent_difference(
                        table2.at[dataset, enhanced], table2.at[dataset, baseline]
                    ),
                }
            )
        wins = int((table2[enhanced] < table2[baseline]).sum())
        losses = int((table2[enhanced] >= table2[baseline]).sum())
        win_rows.append(
            {
                "comparison": f"{enhanced} vs {baseline}",
                "tbma_enhanced_wins": wins,
                "base_wins_or_ties": losses,
            }
        )

    standalone_spd_rows: list[dict[str, Any]] = []
    for baseline in ("CB", "MEN", "RF"):
        if baseline not in table2 or "TBMA" not in table2:
            continue
        for dataset in table2.index:
            standalone_spd_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{baseline}->TBMA",
                    "SPD": symmetric_percent_difference(
                        table2.at[dataset, "TBMA"], table2.at[dataset, baseline]
                    ),
                }
            )

    return {
        "dataset_mean_mase": dataset_means_all,
        "table2_mean_mase": table2.reset_index(),
        "feature_spd": pd.DataFrame(feature_spd_rows),
        "win_counts": pd.DataFrame(win_rows),
        "standalone_spd": pd.DataFrame(standalone_spd_rows),
        "wilcoxon_feature": _wilcoxon_table(dataset_means, FEATURE_PAIRS),
        "wilcoxon_standalone": _wilcoxon_table(
            dataset_means,
            [("CB", "TBMA"), ("MEN", "TBMA"), ("RF", "TBMA")],
        ),
    }


def write_results(per_seed: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write per-seed and paper-style aggregate tables to CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed_mase.csv", index=False)
    summaries = summarize_results(per_seed)
    for name, table in summaries.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return summaries


def _flag(value: Any, *, default: bool, column: str, row_number: int) -> bool:
    """Convert common Excel flag values to ``bool`` with clear validation."""
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if (
        isinstance(value, (float, np.floating))
        and float(value).is_integer()
        and int(value) in {0, 1}
    ):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise ValueError(
        f"dataset_info.xlsx row {row_number}: {column!r} must be a 0/1 flag"
    )


def _positive_int(value: Any, *, column: str, row_number: int) -> int:
    """Return a positive integer from a workbook cell."""
    if pd.isna(value):
        raise ValueError(
            f"dataset_info.xlsx row {row_number}: {column!r} is required"
        )
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(
            f"dataset_info.xlsx row {row_number}: {column!r} must be a positive "
            "integer"
        )
    return int(number)


def load_dataset_info(
    path: str | Path = "dataset_info.xlsx",
    *,
    sheet_name: str = "repo_data",
) -> list[DatasetConfig]:
    """Read enabled paper-workflow datasets from the repository workbook.

    Required columns are ``Dataset``, ``file_name``, ``horizon``,
    ``predetermined_lag``, and ``run``.  ``integer_conversion`` and ``evaluate``
    are optional.  Additional columns are intentionally ignored so the same
    repository workbook can contain information for other analyses.
    """
    workbook_path = Path(path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Dataset information file not found: {workbook_path}")

    try:
        frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    except ImportError as exc:
        raise ImportError(
            "Reading dataset_info.xlsx requires the repository-only "
            "dependency 'openpyxl'. Install paper_reproduction/requirements.txt."
        ) from exc

    required = {
        "Dataset",
        "file_name",
        "horizon",
        "predetermined_lag",
        "run",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{workbook_path} sheet {sheet_name!r} is missing required columns: "
            f"{missing}"
        )

    configs: list[DatasetConfig] = []
    enabled_names: set[str] = set()
    enabled_files: set[str] = set()
    for dataframe_index, row in frame.iterrows():
        excel_row = int(dataframe_index) + 2
        if not _flag(row["run"], default=False, column="run", row_number=excel_row):
            continue

        if pd.isna(row["Dataset"]) or not str(row["Dataset"]).strip():
            raise ValueError(
                f"dataset_info.xlsx row {excel_row}: 'Dataset' is required when run=1"
            )
        if pd.isna(row["file_name"]) or not str(row["file_name"]).strip():
            raise ValueError(
                f"dataset_info.xlsx row {excel_row}: 'file_name' is required when run=1"
            )

        dataset_name = str(row["Dataset"]).strip()
        file_name = str(row["file_name"]).strip()
        if dataset_name in enabled_names:
            raise ValueError(f"Duplicate enabled Dataset value: {dataset_name!r}")
        if file_name in enabled_files:
            raise ValueError(f"Duplicate enabled file_name value: {file_name!r}")
        enabled_names.add(dataset_name)
        enabled_files.add(file_name)

        integer_conversion = _flag(
            row.get("integer_conversion", 0),
            default=False,
            column="integer_conversion",
            row_number=excel_row,
        )
        evaluate = _flag(
            row.get("evaluate", 1),
            default=True,
            column="evaluate",
            row_number=excel_row,
        )
        configs.append(
            DatasetConfig(
                name=dataset_name,
                file_name=file_name,
                horizon=_positive_int(
                    row["horizon"], column="horizon", row_number=excel_row
                ),
                lookback=_positive_int(
                    row["predetermined_lag"],
                    column="predetermined_lag",
                    row_number=excel_row,
                ),
                integer_conversion=integer_conversion,
                evaluate=evaluate,
            )
        )

    if not configs:
        raise ValueError(
            f"{workbook_path} sheet {sheet_name!r} contains no rows with run=1"
        )
    return configs


def _select_configs(
    configs: list[DatasetConfig], selected: list[str] | None
) -> list[DatasetConfig]:
    if not selected:
        return configs
    requested = {item.strip().lower() for item in selected}
    chosen = [
        config
        for config in configs
        if config.name.lower() in requested
        or config.file_name.lower() in requested
        or Path(config.file_name).stem.lower() in requested
    ]
    matched = {
        token
        for token in requested
        if any(
            token
            in {
                config.name.lower(),
                config.file_name.lower(),
                Path(config.file_name).stem.lower(),
            }
            for config in chosen
        )
    }
    unknown = sorted(requested - matched)
    if unknown:
        raise ValueError(
            "Requested datasets are not enabled (run=1) in dataset_info.xlsx: "
            f"{unknown}"
        )
    return chosen


def run_dataset_configs(
    configs: list[DatasetConfig],
    *,
    datasets_dir: str | Path = "Datasets",
    seeds: list[int] | tuple[int, ...] = tuple(range(1, 11)),
    output_dir: str | Path = "paper_results",
    settings: PaperModelSettings | None = None,
    n_jobs: int = -1,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Run the paper workflow for workbook-selected datasets."""
    settings = settings or PaperModelSettings()
    dataset_root = Path(datasets_dir)
    all_results: list[dict[str, Any]] = []

    for config in configs:
        path = dataset_root / config.file_name
        if not path.exists():
            message = f"Configured TSF file does not exist: {path}"
            if skip_missing:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)

        data = prepare_dataset(
            path,
            lookback=config.lookback,
            horizon=config.horizon,
            name=config.name,
            integer_conversion=config.integer_conversion,
            evaluate=config.evaluate,
        )
        print(
            f"[{data.name}] H={data.horizon}, L={data.lookback}, "
            f"m={data.seasonal_period}, frequency={data.frequency}, "
            f"integer_conversion={data.integer_conversion}, "
            f"evaluate={data.evaluate}, seasonal_diff={data.seasonal_differencing} "
            f"(ratio={data.seasonal_dominance_ratio:.3f})"
        )
        for seed in seeds:
            print(f"  seed {seed}")
            all_results.extend(
                run_seed(data, int(seed), settings=settings, n_jobs=n_jobs)
            )

    if not all_results:
        raise ValueError("No configured datasets were evaluated")
    per_seed = pd.DataFrame(all_results)
    summaries = write_results(per_seed, output_dir)
    print("\nMean MASE across evaluated datasets and seeds:")
    print(summaries["table2_mean_mase"].to_string(index=False))
    return per_seed


def run_from_dataset_info(
    dataset_info: str | Path = "dataset_info.xlsx",
    *,
    sheet_name: str = "repo_data",
    datasets_dir: str | Path = "Datasets",
    selected: list[str] | None = None,
    seeds: list[int] | tuple[int, ...] = tuple(range(1, 11)),
    output_dir: str | Path = "paper_results",
    settings: PaperModelSettings | None = None,
    n_jobs: int = -1,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Load ``dataset_info.xlsx`` and execute rows enabled by ``run``."""
    configs = load_dataset_info(dataset_info, sheet_name=sheet_name)
    configs = _select_configs(configs, selected)
    return run_dataset_configs(
        configs,
        datasets_dir=datasets_dir,
        seeds=seeds,
        output_dir=output_dir,
        settings=settings,
        n_jobs=n_jobs,
        skip_missing=skip_missing,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-info",
        type=Path,
        default=Path("dataset_info.xlsx"),
        help="Dataset settings workbook (default: ./dataset_info.xlsx)",
    )
    parser.add_argument(
        "--sheet-name",
        default="repo_data",
        help="Workbook sheet containing dataset settings (default: repo_data)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("Datasets"),
        help="Directory containing configured .tsf files (default: ./Datasets)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Optional enabled dataset name, filename, or filename stem to run. "
            "Repeat the option to select multiple datasets."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Experiment seeds (default: 1 2 ... 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_results"),
        help="Directory for CSV results (default: ./paper_results)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for scikit-learn/CatBoost (-1 uses all cores)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip run=1 rows whose configured TSF file is not present",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_from_dataset_info(
        args.dataset_info,
        sheet_name=args.sheet_name,
        datasets_dir=args.datasets_dir,
        selected=args.dataset,
        seeds=args.seeds,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
