from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbma import TBMA


def make_model(**kwargs):
    params = {
        "ma_order": [1, 3, 7],
        "n_estimators": 9,
        "rf_params": {"max_depth": 4},
        "random_state": 11,
    }
    params.update(kwargs)
    return TBMA(**params)


def test_default_features_return_full_tree_representation(sample_data):
    X, y, _ = sample_data
    model = make_model(n_estimators=5).fit(X, y, seasonal_period=7)
    features = model.generate_features(X, feature_window=3)

    assert features.shape == (len(X), 15)
    assert list(features.columns[:3]) == [
        "tbma_tree_0_feature_3",
        "tbma_tree_0_feature_2",
        "tbma_tree_0_feature_1",
    ]
    assert list(features.columns[-3:]) == [
        "tbma_tree_4_feature_3",
        "tbma_tree_4_feature_2",
        "tbma_tree_4_feature_1",
    ]


def test_default_feature_window_is_one(sample_data):
    X, y, _ = sample_data
    model = make_model(n_estimators=7).fit(X, y, seasonal_period=7)
    features = model.generate_features(X)
    assert features.shape == (len(X), 7)
    assert all(column.endswith("_feature_1") for column in features.columns)


def test_quantile_summary_matches_raw_tree_features(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    raw = model.generate_features(X, feature_window=2)
    quantiles = model.generate_features(
        X, feature_window=2, summary_method="quantile"
    )

    expected_columns = [
        "tbma_feature_1_q25",
        "tbma_feature_2_q25",
        "tbma_feature_1_q50",
        "tbma_feature_2_q50",
        "tbma_feature_1_q75",
        "tbma_feature_2_q75",
    ]
    assert list(quantiles.columns) == expected_columns

    for position in (1, 2):
        tree_columns = [
            f"tbma_tree_{tree_idx}_feature_{position}"
            for tree_idx in range(model.n_estimators)
        ]
        tree_values = raw[tree_columns]
        for q, label in ((0.25, "q25"), (0.5, "q50"), (0.75, "q75")):
            expected = tree_values.quantile(q=q, axis=1)
            pd.testing.assert_series_equal(
                quantiles[f"tbma_feature_{position}_{label}"],
                expected,
                check_names=False,
                check_exact=True,
            )


def test_pca_default_layout_includes_mean_and_two_components(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    features = model.generate_features(
        X, feature_window=2, summary_method="pca"
    )
    assert list(features.columns) == [
        "tbma_feature_1_mean",
        "tbma_feature_2_mean",
        "tbma_feature_1_pc1",
        "tbma_feature_2_pc1",
        "tbma_feature_1_pc2",
        "tbma_feature_2_pc2",
    ]


def test_pca_fixed_components_without_mean(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    features = model.generate_features(
        X,
        feature_window=2,
        summary_method="pca",
        pca_components=3,
        pca_include_mean=False,
    )
    assert list(features.columns) == [
        "tbma_feature_1_pc1",
        "tbma_feature_2_pc1",
        "tbma_feature_1_pc2",
        "tbma_feature_2_pc2",
        "tbma_feature_1_pc3",
        "tbma_feature_2_pc3",
    ]


def test_variance_threshold_pca_returns_component_features(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    features = model.generate_features(
        X,
        feature_window=3,
        summary_method="pca",
        pca_components=0.75,
        pca_include_mean=False,
    )
    assert features.shape[0] == len(X)
    assert features.shape[1] >= 3
    assert all(column.startswith("tbma_feature_") for column in features.columns)
    assert all("_pc" in column for column in features.columns)


def test_pca_basis_is_reused_for_later_subsets(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    full = model.generate_features(X, feature_window=2, summary_method="pca")
    subset = model.generate_features(
        X.iloc[-20:], feature_window=2, summary_method="pca"
    )
    np.testing.assert_allclose(
        subset.to_numpy(),
        full.iloc[-20:].to_numpy(),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )


def test_pca_cache_is_reused(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    first = model.generate_features(X, feature_window=2, summary_method="pca")
    cache_size = len(model.core_._pca_cache)
    second = model.generate_features(X, feature_window=2, summary_method="pca")
    assert len(model.core_._pca_cache) == cache_size == 1
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_pca_raises_clear_error_when_component_count_is_too_large(sample_data):
    X, y, _ = sample_data
    model = make_model(n_estimators=2).fit(X, y, seasonal_period=7)
    with pytest.raises(ValueError, match="supports at most 2 components"):
        model.generate_features(
            X,
            summary_method="pca",
            pca_components=3,
            pca_include_mean=False,
        )


def test_pca_requested_before_available_history_returns_nan(sample_data):
    X, y, _ = sample_data
    model = make_model().fit(X, y, seasonal_period=7)
    early_dates = pd.date_range("1990-01-01", periods=4, freq="D")
    features = model.generate_features(
        X.iloc[:4],
        dates=early_dates,
        feature_window=2,
        summary_method="pca",
    )
    assert features.isna().all().all()


def test_variance_threshold_pca_handles_zero_variance_training_features():
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame({"x": np.zeros(n)}, index=dates)
    y = np.ones(n)
    model = TBMA(
        ma_order=2,
        n_estimators=4,
        rf_params={"max_depth": 2},
        random_state=3,
    ).fit(X, y)

    with pytest.warns(RuntimeWarning):
        features = model.generate_features(
            X,
            summary_method="pca",
            pca_components=0.75,
            pca_include_mean=False,
        )
    assert list(features.columns) == ["tbma_feature_1_pc1"]
