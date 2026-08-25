from __future__ import annotations

import inspect

import tbma
from tbma import TBMA


def test_top_level_api_is_small():
    assert tbma.__all__ == ["TBMA", "__version__"]
    assert TBMA.__name__ == "TBMA"


def test_constructor_contains_only_model_hyperparameters():
    params = TBMA(ma_order=3).get_params(deep=False)
    assert set(params) == {"ma_order", "n_estimators", "rf_params", "random_state"}


def test_feature_summary_choices_are_call_time_options():
    signature = inspect.signature(TBMA.generate_features)
    assert signature.parameters["feature_window"].default == 1
    assert signature.parameters["summary_method"].default is None
    assert signature.parameters["pca_components"].default == 2
    assert signature.parameters["pca_include_mean"].default is True
