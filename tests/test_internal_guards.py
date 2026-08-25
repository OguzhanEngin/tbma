from __future__ import annotations

import importlib
import importlib.metadata

import numpy as np
import pandas as pd
import pytest

import tbma as tbma_package
from tbma import TBMA
from tbma._core import TBMACore


def make_model(**kwargs):
    params = {
        "ma_order": [1, 3, 7],
        "n_estimators": 7,
        "rf_params": {"max_depth": 4},
        "random_state": 5,
    }
    params.update(kwargs)
    return TBMA(**params)


def test_source_checkout_version_fallback(monkeypatch):
    def missing_distribution(_name):
        raise importlib.metadata.PackageNotFoundError

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_distribution)
        reloaded = importlib.reload(tbma_package)
        assert reloaded.__version__ == "0.1.0"

    importlib.reload(tbma_package)


def test_internal_window_guards(sample_data):
    X, y, dates = sample_data
    model = make_model().fit(X, y, seasonal_period=7)

    with pytest.raises(ValueError, match="horizon must be positive"):
        model.core_.predict(X, dates, horizon=0)
    with pytest.raises(ValueError, match="feature_window must be positive"):
        model.core_._generate_tree_features(X, dates, feature_window=0)
    with pytest.raises(ValueError, match="feature_window must be positive"):
        model.core_.generate_quantile_features(X, dates, feature_window=0)
    with pytest.raises(ValueError, match="feature_window must be positive"):
        model.core_.generate_pca_features(
            X,
            dates,
            feature_window=0,
            pca_components=2,
            include_mean=True,
        )
    with pytest.raises(ValueError, match="feature_window must be positive"):
        model.core_._tree_feature_array(X, dates, feature_window=0)


def test_invalid_internal_frequency_raises(sample_data, monkeypatch):
    X, y, _ = sample_data
    model = make_model().fit(X, y, frequency="D")
    monkeypatch.setattr(model.core_, "frequency", "not-a-frequency")
    with pytest.raises(ValueError, match="Invalid frequency"):
        model.core_._shift_dates_by_periods(X.index[:2], 1)


def test_missing_parent_mapping_is_skipped(sample_data, monkeypatch):
    X, y, _ = sample_data
    model = make_model().fit(X, y)
    parent_map = model.core_.tree_parent_map["tree_0"]
    node_idx = next(iter(parent_map))
    monkeypatch.delitem(parent_map, node_idx)

    ma_arr, ma_dates = model.core_._node_ma_lookup(0)
    assert ma_arr.shape[0] == len(ma_dates)


def test_empty_inbag_rows_raise_internal_error():
    core = TBMACore(
        ma_order=1,
        rf_params={"n_estimators": 1},
        random_state=0,
        frequency="D",
        seasonal_period=1,
    )
    core._train_X = pd.DataFrame({"x": [1.0]})
    core._train_y = np.array([1.0])
    core._train_dates = pd.DatetimeIndex(["2020-01-01"])

    class EmptyInBagForest:
        estimators_ = [object()]
        estimators_samples_ = [np.array([], dtype=int)]

    core.tree_ensemble = EmptyInBagForest()

    with pytest.raises(RuntimeError, match="Tree 0 has no in-bag rows"):
        core._calculate_ma_curves()


def test_feature_name_metadata_is_removed_when_columns_stop_being_strings(sample_data):
    X, y, _ = sample_data
    numeric_columns = X.copy()
    numeric_columns.columns = [0, 1, 2]

    fresh = make_model().fit(numeric_columns, y)
    assert not hasattr(fresh, "feature_names_in_")

    refitted = make_model().fit(X, y)
    assert hasattr(refitted, "feature_names_in_")
    refitted.fit(numeric_columns, y)
    assert not hasattr(refitted, "feature_names_in_")


def test_empty_target_guard_is_explicit():
    with pytest.raises(ValueError, match="y must not be empty"):
        TBMA._as_target_array(np.array([]), 0)


def test_pca_requires_complete_training_rows(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=1)
    with pytest.raises(ValueError, match="contain no complete rows"):
        model.generate_features(
            X,
            feature_window=len(X) + 20,
            summary_method="pca",
        )


def test_pca_cache_key_distinguishes_component_settings(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    model.generate_features(X, summary_method="pca", pca_components=2)
    model.generate_features(
        X,
        summary_method="pca",
        pca_components=0.75,
        pca_include_mean=False,
    )
    assert len(model.core_._pca_cache) == 2


def test_reference_lag_rounds_up_to_complete_season(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    assert model.core_._reference_lag(1) == 7
    assert model.core_._reference_lag(7) == 7
    assert model.core_._reference_lag(8) == 14
