from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")
pytest.importorskip("openpyxl")
paper = pytest.importorskip("paper_reproduction.paper_tsf_workflow")
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_tsf(
    path: Path,
    *,
    frequency: str | None = "quarterly",
    horizon: int = 2,
    n_series: int = 5,
    n_periods: int = 40,
) -> Path:
    lines = ["@attribute series_name string"]
    if frequency is not None:
        lines.append(f"@frequency {frequency}")
    lines.extend(
        [
            f"@horizon {horizon}",
            "@missing false",
            "@equallength true",
            "@data",
        ]
    )
    seasonal = np.array([1.0, 4.0, 2.0, 5.0])
    for series_idx in range(n_series):
        values = (
            20.0
            + series_idx
            + np.resize(seasonal, n_periods)
            + np.arange(n_periods) * 0.05
        )
        encoded = ",".join(f"{value:.8f}" for value in values)
        lines.append(f"s{series_idx}:{encoded}")
    path.write_text("\n".join(lines), encoding="cp1252")
    return path


def test_repository_dataset_info_is_read_from_repo_data_sheet():
    configs = paper.load_dataset_info(REPO_ROOT / "dataset_info.xlsx")
    by_name = {config.name: config for config in configs}

    tourism = by_name["Tourism Quarterly"]
    assert tourism.file_name == "tourism_quarterly_dataset.tsf"
    assert tourism.horizon == 8
    assert tourism.lookback == 5
    assert tourism.integer_conversion is False
    assert tourism.evaluate is True

def test_tsf_parser_and_paper_preparation(tmp_path):
    path = _write_tsf(tmp_path / "quarterly.tsf")
    metadata = paper.read_tsf(path)

    assert metadata.frequency == "quarterly"
    assert metadata.horizon == 2
    assert metadata.contains_missing is False
    assert metadata.equal_length is True
    assert len(metadata.attributes) == 5

    data = paper.prepare_dataset(
        path,
        lookback=4,
        horizon=2,
        name="Synthetic Quarterly",
        integer_conversion=True,
        evaluate=False,
    )
    assert data.name == "Synthetic Quarterly"
    assert data.horizon == 2
    assert data.lookback == 4
    assert data.frequency == "QS"
    assert data.seasonal_period == 4
    assert data.seasonal_diff_lag == 4
    assert data.seasonal_differencing is True
    assert data.seasonal_dominance_ratio >= 0.5
    assert data.integer_conversion is True
    assert data.evaluate is False
    assert len(data.test) == 5
    assert len(data.target_columns) == 2
    assert len(data.feature_columns) == 4

    reconstructed = paper._reconstruct_predictions(
        np.full((len(data.test), data.horizon), 0.12345),
        data.test,
        data,
    )
    assert np.equal(reconstructed, np.rint(reconstructed)).all()


def test_missing_tsf_frequency_defaults_to_yearly(tmp_path):
    path = _write_tsf(tmp_path / "missing_frequency.tsf", frequency=None)
    metadata = paper.read_tsf(path)

    assert metadata.frequency is None

    data = paper.prepare_dataset(path, lookback=4)

    assert data.frequency == "YS"
    assert data.seasonal_period == 1
    dates = data.full["date"].drop_duplicates().sort_values().reset_index(drop=True)
    assert dates.iloc[0].month == 1
    assert dates.iloc[0].day == 1
    assert dates.iloc[1].year - dates.iloc[0].year == 1


def test_explicit_unsupported_tsf_frequency_still_raises(tmp_path):
    path = _write_tsf(tmp_path / "unsupported_frequency.tsf", frequency="fortnightly")

    with pytest.raises(ValueError, match="Unsupported TSF frequency 'fortnightly'"):
        paper.prepare_dataset(path, lookback=4)


def test_complete_paper_seed_uses_requested_estimators(tmp_path):
    path = _write_tsf(tmp_path / "quarterly.tsf")
    data = paper.prepare_dataset(path, lookback=4)
    settings = paper.PaperModelSettings(
        tbma_n_estimators=8,
        rf_n_estimators=8,
        catboost_iterations=8,
        men_n_alphas=4,
    )

    rows = paper.run_seed(data, 1, settings=settings, n_jobs=1)
    result = pd.DataFrame(rows).set_index("model")

    assert set(result.index) == set(paper.MODEL_ORDER)
    assert np.isfinite(result["mean_MASE"]).all()
    assert np.isfinite(result["median_MASE"]).all()
    assert (result["n_series"] == 5).all()


def test_workbook_config_drives_complete_dataset_run(tmp_path):
    datasets = tmp_path / "Datasets"
    datasets.mkdir()
    _write_tsf(
        datasets / "tourism_quarterly_dataset.tsf",
        horizon=8,
        n_periods=64,
    )
    settings = paper.PaperModelSettings(
        tbma_n_estimators=8,
        rf_n_estimators=8,
        catboost_iterations=8,
        men_n_alphas=4,
    )

    per_seed = paper.run_from_dataset_info(
        REPO_ROOT / "dataset_info.xlsx",
        datasets_dir=datasets,
        selected=["Tourism Quarterly"],
        seeds=[1],
        output_dir=tmp_path / "results",
        settings=settings,
        n_jobs=1,
    )
    assert set(per_seed["dataset"]) == {"Tourism Quarterly"}
    assert set(per_seed["model"]) == set(paper.MODEL_ORDER)
    assert set(per_seed["file_name"]) == {"tourism_quarterly_dataset.tsf"}
    assert set(per_seed["horizon"]) == {8}
    assert set(per_seed["lookback"]) == {5}


def test_summaries_respect_evaluate_flag(tmp_path):
    rows = []
    for dataset_idx, (dataset, evaluate) in enumerate(
        (("a", True), ("b", True), ("diagnostic", False)), start=1
    ):
        for model_idx, model in enumerate(paper.MODEL_ORDER, start=1):
            for seed in (1, 2):
                rows.append(
                    {
                        "dataset": dataset,
                        "evaluate": evaluate,
                        "seed": seed,
                        "model": model,
                        "mean_MASE": float(
                            dataset_idx + model_idx / 10 + seed / 100
                        ),
                    }
                )
    per_seed = pd.DataFrame(rows)
    output = tmp_path / "results"
    summaries = paper.write_results(per_seed, output)

    assert (output / "per_seed_mase.csv").exists()
    for name in summaries:
        assert (output / f"{name}.csv").exists()
    assert set(summaries["dataset_mean_mase"]["dataset"]) == {
        "a",
        "b",
        "diagnostic",
    }
    assert set(summaries["table2_mean_mase"]["dataset"]) == {"a", "b"}
    assert len(summaries["feature_spd"]) == 6
    assert len(summaries["standalone_spd"]) == 6
    assert len(summaries["wilcoxon_feature"]) == 3
    assert len(summaries["wilcoxon_standalone"]) == 3


def test_dataset_selection_uses_enabled_names_files_and_stems():
    first = paper.DatasetConfig("First", "first.tsf", 2, 4)
    second = paper.DatasetConfig("Second", "second_dataset.tsf", 3, 5)
    configs = [first, second]

    assert paper._select_configs(configs, ["First"]) == [first]
    assert paper._select_configs(configs, ["second_dataset.tsf"]) == [second]
    assert paper._select_configs(configs, ["second_dataset"]) == [second]
    assert paper._select_configs(configs, None) == configs

    with pytest.raises(ValueError, match="not enabled"):
        paper._select_configs(configs, ["missing"])


def test_integer_conversion_only_changes_final_original_scale_predictions(tmp_path):
    path = _write_tsf(tmp_path / "quarterly.tsf")
    base = paper.prepare_dataset(path, lookback=4, integer_conversion=False)
    rounded = replace(base, integer_conversion=True)
    model_values = np.full((len(base.test), base.horizon), 0.12345)

    unrounded = paper._reconstruct_predictions(model_values, base.test, base)
    integer = paper._reconstruct_predictions(model_values, rounded.test, rounded)

    assert np.equal(integer, np.rint(unrounded)).all()
    assert not np.array_equal(unrounded, integer)


def test_tsf_parser_accepts_crlf_and_unicode_paths(tmp_path):
    directory = tmp_path / "Datasets with spaces" / "données"
    directory.mkdir(parents=True)
    path = directory / "séries quarterly.tsf"
    source = _write_tsf(path)
    content = source.read_text(encoding="cp1252")
    source.write_bytes(content.replace("\n", "\r\n").encode("cp1252"))

    metadata = paper.read_tsf(source)
    data = paper.prepare_dataset(source, lookback=4, horizon=2)

    assert metadata.frequency == "quarterly"
    assert data.path == source
    assert len(data.test) == 5


def test_result_output_supports_unicode_and_space_paths(tmp_path):
    rows = []
    for dataset in ("a", "b"):
        for model_idx, model in enumerate(paper.MODEL_ORDER, start=1):
            for seed in (1, 2):
                rows.append(
                    {
                        "dataset": dataset,
                        "evaluate": True,
                        "seed": seed,
                        "model": model,
                        "mean_MASE": 1.0 + model_idx / 10 + seed / 100,
                    }
                )

    output = tmp_path / "paper results" / "résultats"
    paper.write_results(pd.DataFrame(rows), output)

    assert (output / "per_seed_mase.csv").is_file()
    assert (output / "table2_mean_mase.csv").is_file()


def test_run_seed_emits_each_model_as_soon_as_it_finishes(tmp_path):
    path = _write_tsf(tmp_path / "quarterly.tsf")
    data = paper.prepare_dataset(path, lookback=4)
    settings = paper.PaperModelSettings(
        tbma_n_estimators=8,
        rf_n_estimators=8,
        catboost_iterations=8,
        men_n_alphas=4,
    )
    emitted: list[str] = []

    rows = paper.run_seed(
        data,
        1,
        settings=settings,
        n_jobs=1,
        on_result=lambda row: emitted.append(row["model"]),
    )

    assert emitted == [
        "TBMA",
        "CB",
        "MEN",
        "RF",
        "CB_TBMA",
        "MEN_TBMA",
        "RF_TBMA",
    ]
    assert [row["model"] for row in rows] == emitted


def test_completed_models_remain_checkpointed_if_a_seed_is_interrupted(
    tmp_path, monkeypatch
):
    datasets = tmp_path / "Datasets"
    datasets.mkdir()
    _write_tsf(datasets / "synthetic.tsf")
    config = paper.DatasetConfig("Synthetic", "synthetic.tsf", 2, 4)
    output = tmp_path / "paper results"

    def interrupted_run_seed(data, seed, *, settings, n_jobs, on_result):
        del settings, n_jobs
        common = {
            "dataset": data.name,
            "file_name": data.path.name,
            "evaluate": data.evaluate,
            "seed": seed,
            "median_MASE": 1.0,
            "n_series": len(data.test),
            "horizon": data.horizon,
            "lookback": data.lookback,
            "seasonal_period": data.seasonal_period,
            "seasonal_differencing": data.seasonal_differencing,
            "seasonal_dominance_ratio": data.seasonal_dominance_ratio,
        }
        on_result({**common, "model": "TBMA", "mean_MASE": 1.1})
        on_result({**common, "model": "CB", "mean_MASE": 1.2})
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(paper, "run_seed", interrupted_run_seed)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        paper.run_dataset_configs(
            [config],
            datasets_dir=datasets,
            seeds=[1],
            output_dir=output,
            n_jobs=1,
        )

    checkpoint = pd.read_csv(output / "per_seed_mase.csv")
    assert list(checkpoint["model"]) == ["TBMA", "CB"]
    assert list(checkpoint["mean_MASE"]) == [1.1, 1.2]
    assert (output / "dataset_mean_mase.csv").is_file()
    assert (output / "table2_mean_mase.csv").is_file()
