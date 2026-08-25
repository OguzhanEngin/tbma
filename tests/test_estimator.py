from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tbma import TBMA


def make_model(**kwargs):
    params = {
        "ma_order": 3,
        "n_estimators": 9,
        "rf_params": {"max_depth": 4, "min_samples_leaf": 1},
        "random_state": 11,
    }
    params.update(kwargs)
    return TBMA(**params)


def test_fit_predict_with_datetime_index(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    pred = model.predict(X.iloc[-12:], horizon=3)

    assert pred.shape == (12, 3)
    assert list(pred.columns) == ["horizon_1", "horizon_2", "horizon_3"]
    assert pred.index.equals(X.index[-12:])
    assert model.frequency_ == "D"
    assert model.seasonal_period_ == 7
    assert model.ma_order_ == 3


def test_fit_predict_with_numpy_and_explicit_dates(sample_data):
    X, y, dates = sample_data
    model = make_model().fit(X.to_numpy(), y.to_numpy(), dates=dates, frequency="D")
    pred = model.predict(X.to_numpy(), dates=dates, horizon=2)

    assert pred.shape == (len(X), 2)
    assert model.feature_columns_ == ["x0", "x1", "x2"]
    assert list(model.feature_names_in_) == ["x0", "x1", "x2"]


def test_feature_columns_are_reordered_to_fit_order(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    expected = model.predict(X, horizon=2)
    actual = model.predict(X[["x3", "x1", "x2"]], horizon=2)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_prediction_rejects_missing_or_extra_features(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y)

    with pytest.raises(ValueError, match="Missing=.*x3"):
        model.predict(X[["x1", "x2"]])
    with pytest.raises(ValueError, match="extra=.*x4"):
        model.predict(X.assign(x4=1.0))


def test_repeated_fit_is_deterministic_and_does_not_mutate_params(sample_data):
    X, y, _ = sample_data
    rf_params = {"max_depth": 4, "n_estimators": 999, "random_state": 999}
    model = make_model(ma_order=[1, 3, 7], rf_params=rf_params)
    original_rf_params = dict(rf_params)
    original_ma_order = list(model.ma_order)

    first = model.fit(X, y, seasonal_period=7).predict(X, horizon=3)
    second = model.fit(X, y, seasonal_period=7).predict(X, horizon=3)

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert rf_params == original_rf_params
    assert model.rf_params == original_rf_params
    assert model.ma_order == original_ma_order
    assert model.tree_ensemble_.n_estimators == 9
    assert model.tree_ensemble_.random_state == 11


def test_sequence_and_range_ma_orders_resolve_without_mutation(sample_data):
    X, y, _ = sample_data
    for supplied in ([1, 2, 5], (1, 2, 5), range(1, 6), np.array([1, 2, 5])):
        model = make_model(ma_order=supplied).fit(X, y)
        assert model.ma_order_ == list(supplied)


def test_duplicate_dataframe_index_does_not_change_bootstrap_mapping(sample_data):
    X, y, dates = sample_data
    X_reset = X.reset_index(drop=True)
    X_duplicate = X_reset.copy()
    X_duplicate.index = np.arange(len(X_duplicate)) // 2

    first = make_model().fit(X_reset, y.to_numpy(), dates=dates).predict(
        X_reset, dates=dates, horizon=3
    )
    second = make_model().fit(X_duplicate, y.to_numpy(), dates=dates).predict(
        X_duplicate, dates=dates, horizon=3
    )

    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_arbitrary_prediction_horizon_is_not_fixed_at_fit(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)

    assert model.predict(X, horizon=1).shape[1] == 1
    assert model.predict(X, horizon=5).shape[1] == 5


def test_sklearn_clone_round_trip():
    model = make_model(ma_order=(1, 3, 7))
    cloned = clone(model)
    assert cloned.get_params(deep=True) == model.get_params(deep=True)


def test_prediction_methods_require_fit(sample_data):
    X, _, _ = sample_data
    model = make_model()
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.generate_features(X)


def test_model_accessors_expose_fitted_objects(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    assert model.tree_ensemble_ is model.core_.tree_ensemble
    assert model.tree_ma_ is model.core_.tree_ma_dict
    assert model.tree_parent_map_ is model.core_.tree_parent_map
