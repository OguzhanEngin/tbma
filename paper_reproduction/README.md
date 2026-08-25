# Paper reproduction workflow

This directory reproduces the experimental workflow used in the TBMA paper. It is **repository-only research/reproducibility code**. It is deliberately excluded from both the PyPI wheel and source distribution and is not part of the public `tbma` API.

## Repository layout

Run the workflow from the repository root with the following layout:

```text
.
├── dataset_info.xlsx
├── Datasets/
│   ├── tourism_yearly_dataset.tsf
│   ├── tourism_quarterly_dataset.tsf
│   └── ...
├── paper_reproduction/
│   ├── paper_tsf_workflow.py
│   └── requirements.txt
└── src/tbma/
```

`dataset_info.xlsx` is read from the `repo_data` sheet by default. The workflow uses these columns:

| Column | Meaning |
| --- | --- |
| `Dataset` | Display name written to result tables. |
| `file_name` | `.tsf` filename under `./Datasets/`. |
| `horizon` | Forecast horizon `H`. |
| `predetermined_lag` | Autoregressive lookback `L`. |
| `run` | Run the row when set to `1`; skip it when `0`. |
| `integer_conversion` | Round reconstructed predictions to integers before scoring when `1`. |
| `evaluate` | Include the dataset in cross-dataset paper tables/statistical tests when `1`. |

Other workbook columns are intentionally ignored by this workflow. They may remain in the workbook for other repository analyses without becoming TBMA library settings.

## Install

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r paper_reproduction/requirements.txt
```

The downstream models are exactly:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNetCV
from catboost import CatBoostRegressor
```


## Platform support

This repository-only workflow is written with `pathlib` and Python file APIs, not
OS-specific shell/path conventions. GitHub CI runs its end-to-end synthetic TSF
tests on `ubuntu-latest`, `windows-latest`, and `macos-latest` with Python 3.13.
The tests include CRLF TSF input plus dataset/result paths containing spaces and
Unicode characters. CatBoost and OpenPyXL remain repository-only dependencies.

## Run

Run every row with `run = 1`:

```bash
python paper_reproduction/paper_tsf_workflow.py
```

Run only one or more enabled datasets without editing the workbook:

```bash
python paper_reproduction/paper_tsf_workflow.py \
  --dataset "FREDMD" \
  --dataset "M3 Quarterly"
```

Useful overrides:

```bash
python paper_reproduction/paper_tsf_workflow.py \
  --dataset-info ./dataset_info.xlsx \
  --datasets-dir ./Datasets \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --output-dir ./paper_results \
  --n-jobs -1
```

Use `--skip-missing` only when intentionally working with an incomplete local dataset directory. The default is to fail if a row has `run = 1` but its configured file is absent, which prevents silently omitting a configured experiment.

## Paper settings

The workflow uses the manuscript settings by default:

- TBMA: 256 trees, maximum depth 12, minimum leaf size 4, `max_features=1/3`, per-tree MA order sampled from `{1, 2, 3, 4, 5}`, and `feature_window=1`.
- Random forest: 256 trees, maximum depth 8, minimum leaf size 4, `max_features=1/3`.
- CatBoost: 512 iterations, learning rate 0.025, depth 5, minimum child size 32, `colsample_bylevel=1/3`, and `loss_function="MultiRMSE"`.
- Multi-task elastic net: `MultiTaskElasticNetCV` with `l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.9)`, `eps=1e-4`, 20 alpha values, and `max_iter=1000`.
- Seeds: 1 through 10.

The script performs training-only mean scaling, the paper's dataset-level seasonal-dominance test, optional whole-season differencing, pooled global lag construction, TBMA fitting, full tree-level TBMA feature generation, base and TBMA-augmented downstream fitting, standalone TBMA readout, reconstruction to the original scale, MASE calculation, and the paper-style SPD/win-count/Wilcoxon-Holm summaries.

The earliest seasonally aligned TBMA feature positions can be unavailable because their reference dates precede fitted PMA history. CatBoost and modern `RandomForestRegressor` can handle those missing predictors. For TBMA-augmented `MultiTaskElasticNetCV`, which cannot accept NaNs, incomplete training origins are removed; no imputation is introduced.

## Outputs

By default results are written to `./paper_results/`:

```text
paper_results/
├── per_seed_mase.csv
├── dataset_mean_mase.csv
├── table2_mean_mase.csv
├── feature_spd.csv
├── standalone_spd.csv
├── win_counts.csv
├── wilcoxon_feature.csv
└── wilcoxon_standalone.csv
```

`per_seed_mase.csv` and `dataset_mean_mase.csv` retain every executed (`run = 1`) dataset. Cross-dataset paper tables and statistical comparisons use only rows whose workbook setting has `evaluate = 1`.
