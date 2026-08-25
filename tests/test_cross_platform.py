from __future__ import annotations

import os

import pandas as pd

from tbma import TBMA


def test_persistence_supports_nested_unicode_and_space_paths(tmp_path, sample_data):
    X, y, _ = sample_data
    model = TBMA(
        ma_order=range(1, 5),
        n_estimators=7,
        rf_params={"max_depth": 4},
        random_state=21,
    ).fit(X, y, seasonal_period=7)
    expected = model.predict(X.iloc[-12:], horizon=2)

    model_path = tmp_path / "models with spaces" / "tëmporal 模型" / "tbma model.pkl"
    saved = model.save(model_path)

    assert saved == model_path
    assert saved.exists()
    loaded = TBMA.load(os.fspath(saved))
    actual = loaded.predict(X.iloc[-12:], horizon=2)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_pathlike_objects_are_accepted_for_persistence(tmp_path, sample_data):
    X, y, _ = sample_data
    model = TBMA(ma_order=2, n_estimators=5, random_state=5).fit(
        X, y, seasonal_period=7
    )

    path = tmp_path / "portable.pkl"
    model.save(path)
    loaded = TBMA.load(path)

    pd.testing.assert_frame_equal(
        loaded.predict(X.iloc[-8:], horizon=1),
        model.predict(X.iloc[-8:], horizon=1),
        check_exact=True,
    )
