from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbma import TBMA


def make_model(**kwargs):
    params = {"ma_order": 3, "n_estimators": 5, "random_state": 3}
    params.update(kwargs)
    return TBMA(**params)


def test_X_validation(sample_data):
    X, y, dates = sample_data
    with pytest.raises(ValueError, match="two-dimensional"):
        make_model().fit(np.arange(len(X)), y, dates=dates)
    with pytest.raises(ValueError, match="must not be empty"):
        make_model().fit(X.iloc[:0], y.iloc[:0], dates=dates[:0], frequency="D")

    no_columns = pd.DataFrame(index=X.index)
    with pytest.raises(ValueError, match="at least one feature column"):
        make_model().fit(no_columns, y, dates=dates)

    duplicate = X.copy()
    duplicate.columns = ["x", "x", "z"]
    with pytest.raises(ValueError, match="feature names must be unique"):
        make_model().fit(duplicate, y, dates=dates)


def test_y_validation(sample_data):
    X, y, dates = sample_data
    with pytest.raises(ValueError, match="one-dimensional"):
        make_model().fit(X, y.to_numpy()[:, None], dates=dates)
    with pytest.raises(ValueError, match="rows but X has"):
        make_model().fit(X.iloc[:-1], y, dates=dates[:-1])


def test_date_validation(sample_data):
    X, y, dates = sample_data
    plain_X = X.reset_index(drop=True)
    with pytest.raises(ValueError, match="dates are required"):
        make_model().fit(plain_X, y.to_numpy())
    with pytest.raises(ValueError, match="one-dimensional"):
        make_model().fit(plain_X, y.to_numpy(), dates=np.asarray(dates)[:, None])
    with pytest.raises(ValueError, match="rows but X has"):
        make_model().fit(plain_X, y.to_numpy(), dates=dates[:-1])

    invalid = np.asarray(dates, dtype=object).copy()
    invalid[0] = "not-a-date"
    with pytest.raises(ValueError, match="invalid timestamps"):
        make_model().fit(plain_X, y.to_numpy(), dates=invalid)

    missing = pd.DatetimeIndex([pd.NaT, *dates[1:]])
    with pytest.raises(ValueError, match="missing timestamps"):
        make_model().fit(plain_X, y.to_numpy(), dates=missing, frequency="D")


@pytest.mark.parametrize(
    ("ma_order", "error", "message"),
    [
        (0, ValueError, "positive integer"),
        (-1, ValueError, "positive integer"),
        (True, TypeError, "positive integer or a repeatable sequence"),
        ([], ValueError, "must not be empty"),
        ([1, 0, 3], ValueError, "positive integers"),
        ([1, True, 3], ValueError, "positive integers"),
        ([1, 2.5], ValueError, "positive integers"),
        ("seasonal", TypeError, "positive integer or a sequence"),
        ((value for value in [1, 2]), TypeError, "repeatable sequence"),
        (np.array([[1, 2]]), ValueError, "one-dimensional"),
    ],
)
def test_ma_order_validation(sample_data, ma_order, error, message):
    X, y, _ = sample_data
    with pytest.raises(error, match=message):
        make_model(ma_order=ma_order).fit(X, y)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_estimators": 0}, ValueError, "n_estimators"),
        ({"n_estimators": True}, ValueError, "n_estimators"),
        ({"random_state": None}, TypeError, "random_state"),
        ({"random_state": True}, TypeError, "random_state"),
        ({"random_state": -1}, ValueError, "random_state"),
        ({"random_state": 2**32}, ValueError, "random_state"),
        ({"rf_params": []}, TypeError, "rf_params"),
    ],
)
def test_hyperparameter_validation(sample_data, kwargs, error, message):
    X, y, _ = sample_data
    with pytest.raises(error, match=message):
        make_model(**kwargs).fit(X, y)


def test_maximum_random_state_is_supported(sample_data):
    X, y, _ = sample_data
    model = make_model(random_state=2**32 - 1).fit(X, y)
    assert model.tree_ensemble_.random_state == 2**32 - 1


def test_unknown_random_forest_parameter_is_rejected(sample_data):
    X, y, _ = sample_data
    with pytest.raises(ValueError, match="Invalid RandomForestRegressor"):
        make_model(rf_params={"not_a_parameter": 1}).fit(X, y)


@pytest.mark.parametrize("seasonal_period", [0, -1, 1.5, "7", [7], True])
def test_seasonal_period_must_be_positive_integer(sample_data, seasonal_period):
    X, y, _ = sample_data
    with pytest.raises(ValueError, match="seasonal_period"):
        make_model().fit(X, y, seasonal_period=seasonal_period)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "3", True])
def test_horizon_must_be_positive_integer(sample_data, horizon):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="horizon"):
        model.predict(X, horizon=horizon)


@pytest.mark.parametrize("feature_window", [0, -1, 1.5, "3", True])
def test_feature_window_must_be_positive_integer(sample_data, feature_window):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="feature_window"):
        model.generate_features(X, feature_window=feature_window)


@pytest.mark.parametrize(
    ("summary_method", "error"),
    [("other", ValueError), (1, TypeError), ([], TypeError)],
)
def test_summary_method_validation(sample_data, summary_method, error):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(error, match="summary_method"):
        model.generate_features(X, summary_method=summary_method)


@pytest.mark.parametrize("value", [0, -1, 1.0, 1.5, True, "2", None])
def test_pca_components_validation(sample_data, value):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="pca_components"):
        model.generate_features(X, summary_method="pca", pca_components=value)


def test_pca_include_mean_must_be_boolean(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(TypeError, match="pca_include_mean"):
        model.generate_features(X, summary_method="pca", pca_include_mean=1)


def test_pca_only_options_are_rejected_for_other_summary_methods(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="only applicable"):
        model.generate_features(X, pca_components=3)
    with pytest.raises(ValueError, match="only applicable"):
        model.generate_features(
            X, summary_method="quantile", pca_include_mean=False
        )


@pytest.mark.parametrize(
    "frequency",
    ["", "1_D", "daily", "0D", "SME", "CBME", "bh"],
)
def test_unsupported_frequency_is_rejected(sample_data, frequency):
    X, y, _ = sample_data
    error = TypeError if frequency == "" else ValueError
    with pytest.raises(error, match="frequency|Unsupported"):
        make_model().fit(X, y, frequency=frequency)
