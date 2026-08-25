from __future__ import annotations

import pickle

import pandas as pd
import pytest

from tbma import TBMA


def test_save_load_round_trip(tmp_path, sample_data):
    X, y, _ = sample_data
    model = TBMA(
        ma_order=range(1, 5),
        n_estimators=7,
        rf_params={"max_depth": 4},
        random_state=9,
    ).fit(X, y, seasonal_period=7)
    expected = model.predict(X, horizon=3)

    path = model.save(tmp_path / "models" / "tbma.pkl")
    loaded = TBMA.load(path)
    actual = loaded.predict(X, horizon=3)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_save_requires_fitted_model(tmp_path):
    with pytest.raises(Exception, match="not fitted"):
        TBMA(ma_order=3).save(tmp_path / "model.pkl")


def test_load_rejects_other_pickle_object(tmp_path):
    path = tmp_path / "other.pkl"
    with path.open("wb") as handle:
        pickle.dump({"not": "tbma"}, handle)
    with pytest.raises(TypeError, match="not TBMA"):
        TBMA.load(path)


def test_save_load_preserves_cached_pca_summary(tmp_path, sample_data):
    X, y, _ = sample_data
    model = TBMA(
        ma_order=range(1, 5),
        n_estimators=7,
        rf_params={"max_depth": 4},
        random_state=9,
    ).fit(X, y, seasonal_period=7)
    expected = model.generate_features(
        X.iloc[-20:], feature_window=2, summary_method="pca"
    )

    loaded = TBMA.load(model.save(tmp_path / "tbma-with-pca.pkl"))
    actual = loaded.generate_features(
        X.iloc[-20:], feature_window=2, summary_method="pca"
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
