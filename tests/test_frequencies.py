from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbma import TBMA


def fit_small(dates, *, frequency=None, seasonal_period=1):
    n = len(dates)
    X = pd.DataFrame({"x": np.sin(np.arange(n) / 5.0)}, index=dates)
    y = np.cos(np.arange(n) / 7.0)
    return TBMA(
        ma_order=2,
        n_estimators=5,
        rf_params={"max_depth": 3},
        random_state=2,
    ).fit(X, y, frequency=frequency, seasonal_period=seasonal_period)


@pytest.mark.parametrize(
    ("frequency", "pandas_frequency", "canonical"),
    [
        ("s", "s", "s"),
        ("10min", "10min", "10min"),
        ("T", "min", "min"),
        ("h", "h", "h"),
        ("H", "h", "h"),
        ("2h", "2h", "2h"),
        ("D", "D", "D"),
        ("2D", "2D", "2D"),
        ("B", "B", "B"),
        ("W", "W", "W-SUN"),
        ("W-MON", "W-MON", "W-MON"),
        ("MS", "MS", "MS"),
        ("M", "ME", "ME"),
        ("ME", "ME", "ME"),
        ("QS", "QS", "QS-JAN"),
        ("QS-OCT", "QS-OCT", "QS-OCT"),
        ("Q", "QE-DEC", "QE-DEC"),
        ("QE-DEC", "QE-DEC", "QE-DEC"),
        ("YS", "YS", "YS-JAN"),
        ("YS-JUL", "YS-JUL", "YS-JUL"),
        ("Y", "YE-DEC", "YE-DEC"),
        ("YE-DEC", "YE-DEC", "YE-DEC"),
    ],
)
def test_supported_frequency_strings(frequency, pandas_frequency, canonical):
    dates = pd.date_range("2012-01-01", periods=40, freq=pandas_frequency)
    model = fit_small(dates, frequency=frequency)
    assert model.frequency_ == canonical
    assert model.predict(model.core_._train_X, horizon=2).shape == (40, 2)


@pytest.mark.parametrize(
    "pandas_frequency",
    ["15min", "2h", "D", "7D", "B", "MS", "ME", "QS", "QE", "YS", "YE"],
)
def test_frequency_is_inferred_for_regular_dates(pandas_frequency):
    dates = pd.date_range("2012-01-01", periods=40, freq=pandas_frequency)
    model = fit_small(dates)
    expected = TBMA._canonicalize_frequency(pd.infer_freq(dates)).label
    assert model.frequency_ == expected


def test_frequency_inference_uses_unique_dates():
    base = pd.date_range("2020-01-01", periods=30, freq="D")
    dates = pd.DatetimeIndex(np.repeat(base.to_numpy(), 2))
    model = fit_small(dates)
    assert model.frequency_ == "D"


def test_irregular_dates_require_explicit_frequency():
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-07"])
    with pytest.raises(ValueError, match="could not be inferred"):
        fit_small(dates)


def test_too_few_unique_dates_require_explicit_frequency():
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
    with pytest.raises(ValueError, match="fewer than three"):
        fit_small(dates)


def test_calendar_month_start_shift():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    model = fit_small(dates, frequency="MS", seasonal_period=12)
    shifted = model.core_._shift_dates_by_periods(
        pd.DatetimeIndex(["2020-03-01"]), -1
    )
    assert shifted[0] == np.datetime64("2020-02-01")


def test_calendar_month_end_shift():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    model = fit_small(dates, frequency="ME", seasonal_period=12)
    shifted = model.core_._shift_dates_by_periods(
        pd.DatetimeIndex(["2020-03-31"]), -1
    )
    assert shifted[0] == np.datetime64("2020-02-29")


def test_business_day_shift_skips_weekend():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    model = fit_small(dates, frequency="B")
    shifted = model.core_._shift_dates_by_periods(
        pd.DatetimeIndex(["2020-01-06"]), -1
    )
    assert shifted[0] == np.datetime64("2020-01-03")


def test_daily_timezone_shift_preserves_local_midnight_across_dst():
    dates = pd.date_range(
        "2024-02-15", periods=45, freq="D", tz="America/New_York"
    )
    model = fit_small(dates, frequency="D")
    shifted = model.core_._shift_dates_by_periods(
        pd.DatetimeIndex([pd.Timestamp("2024-03-11 00:00", tz="America/New_York")]),
        -1,
    )
    result = pd.DatetimeIndex(shifted)[0]
    assert result == pd.Timestamp("2024-03-10 00:00", tz="America/New_York")


def test_weekly_timezone_shift_preserves_local_time_across_dst():
    dates = pd.date_range(
        "2024-01-01", periods=20, freq="W-MON", tz="America/New_York"
    )
    model = fit_small(dates, frequency="W-MON")
    shifted = model.core_._shift_dates_by_periods(
        pd.DatetimeIndex([pd.Timestamp("2024-03-11 00:00", tz="America/New_York")]),
        -1,
    )
    result = pd.DatetimeIndex(shifted)[0]
    assert result == pd.Timestamp("2024-03-04 00:00", tz="America/New_York")
