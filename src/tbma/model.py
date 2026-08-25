"""Public estimator for Tree-Based Moving Average (TBMA)."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import (
    BusinessDay,
    Day,
    Hour,
    Minute,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    Second,
    Week,
    YearBegin,
    YearEnd,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from ._core import TBMACore

_MAX_RANDOM_STATE = 2**32 - 1
_SUPPORTED_OFFSET_TYPES = (
    Second,
    Minute,
    Hour,
    Day,
    BusinessDay,
    Week,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)
_LEGACY_FREQUENCY_ALIASES = {
    "T": "min",
    "H": "h",
    "M": "ME",
    "Q": "QE-DEC",
    "Y": "YE-DEC",
}


@dataclass(frozen=True)
class _FrequencySpec:
    label: str


class TBMA(RegressorMixin, BaseEstimator):
    """Tree-Based Moving Average estimator.

    TBMA fits a supervised random forest to prepared predictors and a one-step
    target. The forest partitions define supervised neighborhoods, and each
    tree receives node-specific, one-sided pooled moving-average (PMA) curves
    built from its unique in-bag training observations.

    Parameters
    ----------
    ma_order : int or sequence of int
        Moving-average order(s), measured in observation periods. A single
        positive integer applies the same order to every tree. When a sequence
        is supplied, each tree deterministically draws one order from it using
        ``random_state``.
    n_estimators : int, default=100
        Number of trees in the supervised random forest.
    rf_params : dict or None, default=None
        Additional ``RandomForestRegressor`` parameters. ``n_estimators`` and
        ``random_state`` are controlled by the corresponding TBMA arguments.
    random_state : int, default=0
        Seed used by the random forest and per-tree MA-order sampling. Valid
        values are integers from 0 through ``2**32 - 1``.

    Notes
    -----
    TBMA expects already-prepared predictors and a one-dimensional one-step
    target. It does not create lags, scale or difference series, construct
    multi-step target tables, infer seasonality, or choose moving-average
    orders from seasonal metadata.
    """

    def __init__(
        self,
        ma_order: int | Sequence[int] | np.ndarray,
        n_estimators: int = 100,
        rf_params: dict[str, Any] | None = None,
        random_state: int = 0,
    ) -> None:
        self.ma_order = ma_order
        self.n_estimators = n_estimators
        self.rf_params = rf_params
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        *,
        dates: Iterable[Any] | None = None,
        frequency: str | None = None,
        seasonal_period: int | None = None,
    ) -> TBMA:
        """Fit TBMA's supervised forest and node-specific PMA curves.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Prepared predictors used to learn the forest neighborhoods.
            DataFrame column names are retained and checked at prediction time.
        y : array-like of shape (n_samples,)
            One-step target values used to train the forest and build PMA
            curves.
        dates : array-like of datetime-like, optional
            Observation timestamps in the same row order as ``X``. May be
            omitted when ``X`` has a ``DatetimeIndex``.
        frequency : str, optional
            Regular pandas-style sampling frequency, for example ``"D"``,
            ``"B"``, ``"W-MON"``, ``"MS"``, ``"ME"``, ``"QS"``,
            ``"QE-DEC"``, ``"YS"``, ``"YE-DEC"``, ``"h"``, or
            ``"10min"``. If omitted, frequency is inferred from at least three
            unique regular training timestamps.
        seasonal_period : int or None, default=None
            Number of observation periods in a season. ``None`` uses 1. This
            controls seasonal alignment of PMA reference positions and is
            independent of ``ma_order``.

        Returns
        -------
        TBMA
            The fitted estimator.
        """
        ma_order = self._validated_ma_order()
        self._validate_hyperparameters()
        feature_df = self._as_feature_frame(X)
        target_arr = self._as_target_array(y, len(feature_df))
        date_index = self._resolve_dates(feature_df.index, dates, len(feature_df))
        frequency_spec = self._resolve_frequency(frequency, date_index)
        resolved_seasonal_period = self._validate_seasonal_period(seasonal_period)

        rf_params = self._resolved_rf_params()
        self._validate_rf_params(rf_params)

        core = TBMACore(
            ma_order=ma_order,
            rf_params=rf_params,
            random_state=int(self.random_state),
            frequency=frequency_spec.label,
            seasonal_period=resolved_seasonal_period,
        )
        core.fit(feature_df, target_arr, date_index)

        self.core_ = core
        self.feature_columns_ = list(feature_df.columns)
        if all(isinstance(column, str) for column in self.feature_columns_):
            self.feature_names_in_ = np.asarray(self.feature_columns_, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_features_in_ = feature_df.shape[1]
        self.frequency_ = frequency_spec.label
        self.seasonal_period_ = resolved_seasonal_period
        self.ma_order_ = ma_order
        return self

    def predict(
        self,
        X,
        *,
        dates: Iterable[Any] | None = None,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Return standalone TBMA forecasts for horizons 1 through ``horizon``."""
        feature_df, date_index = self._prepare_prediction_input(X, dates)
        return self.core_.predict(
            feature_df,
            date_index,
            horizon=self._validate_positive_integer(horizon, "horizon"),
        )

    def generate_features(
        self,
        X,
        *,
        dates: Iterable[Any] | None = None,
        feature_window: int = 1,
        summary_method: str | None = None,
        pca_components: float = 2,
        pca_include_mean: bool = True,
    ) -> pd.DataFrame:
        """Generate the TBMA temporal feature representation.

        By default, every tree contributes one feature for each position in the
        seasonally aligned ``feature_window``. With ``K`` fitted trees this
        returns ``K * feature_window`` columns, matching the full TBMA
        tree-level representation defined by the TBMA feature construction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Prepared predictors whose forest neighborhoods determine the TBMA
            feature values.
        dates : array-like of datetime-like, optional
            Observation timestamps. May be omitted when ``X`` has a
            ``DatetimeIndex``.
        feature_window : int, default=1
            Number of seasonally aligned PMA positions contributed by each
            tree. This is the paper's feature-window length ``w_g`` and is
            separate from the forecast ``horizon`` used by :meth:`predict`.
        summary_method : {None, "quantile", "pca"}, default=None
            ``None`` returns all tree-level TBMA features. ``"quantile"``
            returns the 25th, 50th, and 75th percentiles across trees for each
            feature position. ``"pca"`` returns PCA summaries across trees.
        pca_components : int or float, default=2
            Used only when ``summary_method="pca"``. A positive integer requests
            that many components. A float strictly between 0 and 1 selects the
            smallest component count reaching that cumulative explained
            variance threshold. PCA bases are fitted from the fitted training
            TBMA representation and reused for later calls.
        pca_include_mean : bool, default=True
            Used only when ``summary_method="pca"``. If true, include the mean
            tree-level TBMA value for each feature position together with the
            PCA components.

        Returns
        -------
        pandas.DataFrame
            TBMA feature columns indexed like ``X``.
        """
        feature_df, date_index = self._prepare_prediction_input(X, dates)
        feature_window = self._validate_positive_integer(
            feature_window, "feature_window"
        )
        summary_method = self._validate_summary_method(summary_method)

        if summary_method is None:
            self._validate_unused_pca_options(pca_components, pca_include_mean)
            return self.core_._generate_tree_features(
                feature_df,
                date_index,
                feature_window=feature_window,
            )
        if summary_method == "quantile":
            self._validate_unused_pca_options(pca_components, pca_include_mean)
            return self.core_.generate_quantile_features(
                feature_df,
                date_index,
                feature_window=feature_window,
            )

        components = self._validate_pca_components(pca_components)
        if not isinstance(pca_include_mean, (bool, np.bool_)):
            raise TypeError("pca_include_mean must be a boolean")
        return self.core_.generate_pca_features(
            feature_df,
            date_index,
            feature_window=feature_window,
            pca_components=components,
            include_mean=bool(pca_include_mean),
        )

    def save(self, filename: str | Path) -> Path:
        """Serialize a fitted estimator with pickle.

        Only load pickle files from trusted sources because unpickling can
        execute arbitrary code.
        """
        check_is_fitted(self, attributes=["core_"])
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, filename: str | Path) -> TBMA:
        """Load an estimator previously written by :meth:`save`."""
        path = Path(filename)
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} contains {type(obj).__name__}, not {cls.__name__}")
        return obj

    @property
    def tree_ensemble_(self) -> RandomForestRegressor:
        """Fitted ``RandomForestRegressor``."""
        check_is_fitted(self, attributes=["core_"])
        return self.core_.tree_ensemble

    @property
    def tree_ma_(self) -> dict[str, pd.DataFrame]:
        """Node-level pooled moving-average curves for each fitted tree."""
        check_is_fitted(self, attributes=["core_"])
        return self.core_.tree_ma_dict

    @property
    def tree_parent_map_(self) -> dict[str, dict[int, int]]:
        """Child-to-parent node mappings for each fitted tree."""
        check_is_fitted(self, attributes=["core_"])
        return self.core_.tree_parent_map

    def _prepare_prediction_input(
        self,
        X,
        dates: Iterable[Any] | None,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        check_is_fitted(self, attributes=["core_", "feature_columns_"])
        feature_df = self._as_feature_frame(X)
        self._validate_feature_columns(feature_df)
        feature_df = feature_df.loc[:, self.feature_columns_].copy()
        date_index = self._resolve_dates(feature_df.index, dates, len(feature_df))
        return feature_df, date_index

    @staticmethod
    def _as_feature_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be two-dimensional")
            frame = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])

        if frame.shape[1] == 0:
            raise ValueError("X must contain at least one feature column")
        if frame.empty:
            raise ValueError("X must not be empty")
        if not frame.columns.is_unique:
            raise ValueError("X feature names must be unique")
        return frame

    @staticmethod
    def _as_target_array(y, n_rows: int) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.Index)):
            arr = y.to_numpy()
        else:
            arr = np.asarray(y)
        if arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(arr) != n_rows:
            raise ValueError(f"y has {len(arr)} rows but X has {n_rows} rows")
        if len(arr) == 0:
            raise ValueError("y must not be empty")
        return arr

    @classmethod
    def _resolve_dates(
        cls,
        frame_index: pd.Index,
        dates: Iterable[Any] | None,
        n_rows: int,
    ) -> pd.DatetimeIndex:
        values = dates
        if values is None and isinstance(frame_index, pd.DatetimeIndex):
            values = frame_index
        if values is None:
            raise ValueError("dates are required unless X uses a DatetimeIndex")

        if isinstance(values, (pd.Series, pd.Index)):
            arr = values.to_numpy()
        else:
            arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError("dates must be one-dimensional")
        if len(arr) != n_rows:
            raise ValueError(f"dates has {len(arr)} rows but X has {n_rows} rows")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = pd.DatetimeIndex(pd.to_datetime(arr))
        except Exception as exc:
            raise ValueError("dates contains invalid timestamps") from exc
        if result.isna().any():
            raise ValueError("dates contains missing timestamps")
        return result

    def _validate_feature_columns(self, feature_df: pd.DataFrame) -> None:
        actual = list(feature_df.columns)
        expected = list(self.feature_columns_)
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if missing or extra:
            raise ValueError(
                "Prediction features must exactly match fit-time features. "
                f"Missing={missing}, extra={extra}"
            )

    def _validated_ma_order(self) -> int | list[int]:
        value = self.ma_order
        if isinstance(value, (str, bytes)):
            raise TypeError(
                "ma_order must be a positive integer or a sequence of positive integers"
            )
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            if value < 1:
                raise ValueError("ma_order must be a positive integer")
            return int(value)

        if not isinstance(value, Sequence) and not isinstance(value, np.ndarray):
            raise TypeError(
                "ma_order must be a positive integer or a repeatable sequence "
                "of positive integers"
            )
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError("ma_order array must be one-dimensional")
        windows = list(value)
        if not windows:
            raise ValueError("ma_order must not be empty")
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or item < 1
            for item in windows
        ):
            raise ValueError("all ma_order values must be positive integers")
        return [int(item) for item in windows]

    def _resolved_rf_params(self) -> dict[str, Any]:
        if self.rf_params is None:
            params: dict[str, Any] = {}
        elif isinstance(self.rf_params, dict):
            params = deepcopy(self.rf_params)
        else:
            raise TypeError("rf_params must be a dictionary or None")
        params["n_estimators"] = int(self.n_estimators)
        params.pop("random_state", None)
        return params

    @staticmethod
    def _validate_rf_params(params: dict[str, Any]) -> None:
        try:
            RandomForestRegressor(**params)
        except TypeError as exc:
            raise ValueError(
                f"Invalid RandomForestRegressor parameters: {exc}"
            ) from exc

    def _validate_hyperparameters(self) -> None:
        if (
            isinstance(self.n_estimators, (bool, np.bool_))
            or not isinstance(self.n_estimators, (int, np.integer))
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")
        if not 0 <= int(self.random_state) <= _MAX_RANDOM_STATE:
            raise ValueError(f"random_state must be between 0 and {_MAX_RANDOM_STATE}")

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _validate_seasonal_period(seasonal_period: int | None) -> int:
        if seasonal_period is None:
            return 1
        if (
            isinstance(seasonal_period, (bool, np.bool_))
            or not isinstance(seasonal_period, (int, np.integer))
            or seasonal_period < 1
        ):
            raise ValueError("seasonal_period must be a positive integer or None")
        return int(seasonal_period)

    @staticmethod
    def _validate_summary_method(summary_method: str | None) -> str | None:
        if summary_method is not None and not isinstance(summary_method, str):
            raise TypeError("summary_method must be None, 'quantile', or 'pca'")
        if summary_method not in {None, "quantile", "pca"}:
            raise ValueError("summary_method must be None, 'quantile', or 'pca'")
        return summary_method

    @staticmethod
    def _validate_pca_components(value: float) -> int | float:
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            if value < 1:
                raise ValueError("integer pca_components must be positive")
            return int(value)
        if isinstance(value, (float, np.floating)) and 0 < value < 1:
            return float(value)
        raise ValueError(
            "pca_components must be a positive integer or a float strictly "
            "between 0 and 1"
        )

    @staticmethod
    def _validate_unused_pca_options(
        pca_components: float,
        pca_include_mean: bool,
    ) -> None:
        valid_include_mean = isinstance(pca_include_mean, (bool, np.bool_)) and bool(
            pca_include_mean
        )
        if pca_components != 2 or not valid_include_mean:
            raise ValueError(
                "pca_components and pca_include_mean are only applicable when "
                "summary_method='pca'"
            )

    @classmethod
    def _resolve_frequency(
        cls,
        frequency: str | None,
        dates: pd.DatetimeIndex,
    ) -> _FrequencySpec:
        if frequency is None:
            unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
            if len(unique_dates) < 3:
                raise ValueError(
                    "frequency could not be inferred from fewer than three unique "
                    "dates; pass frequency explicitly"
                )
            inferred = pd.infer_freq(unique_dates)
            if inferred is None:
                raise ValueError(
                    "frequency could not be inferred from the training dates; "
                    "pass frequency explicitly"
                )
            frequency = inferred
        return cls._canonicalize_frequency(frequency)

    @staticmethod
    def _canonicalize_frequency(frequency: str) -> _FrequencySpec:
        if not isinstance(frequency, str) or not frequency.strip():
            raise TypeError("frequency must be a non-empty string")

        frequency_text = frequency.strip()
        normalized_frequency = _LEGACY_FREQUENCY_ALIASES.get(
            frequency_text, frequency_text
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                offset = to_offset(normalized_frequency)
        except ValueError as exc:
            raise ValueError(
                "Unsupported frequency. Use a regular pandas frequency such as "
                "'D', 'B', 'W-MON', 'MS', 'ME', 'QS', 'QE-DEC', 'YS', "
                "'YE-DEC', 'h', or '10min'."
            ) from exc

        if offset.n < 1:
            raise ValueError("frequency multiplier must be positive")
        if not isinstance(offset, _SUPPORTED_OFFSET_TYPES):
            # The caller supplied a string; this is a value-domain error.
            raise ValueError(  # noqa: TRY004
                f"Unsupported frequency '{frequency}'. TBMA supports regular "
                "second/minute/hour/day/business-day/week/month/quarter/year "
                "start or end frequencies."
            )
        return _FrequencySpec(label=offset.freqstr)
