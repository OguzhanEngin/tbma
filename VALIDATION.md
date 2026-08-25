# Validation record for 0.1.0

The numerical implementation was compared with the untouched `tbma.py` supplied with the source code used to create this package.

Reference source SHA-256:

```text
2d4cb04c0e301563e9e795a29adfe53b1b583da606bbb157e11b48f93333f789
```

The reference source is not included in the release distribution.

## Numerical-equivalence checks

Using the same predictors, one-step targets, timestamps, forest parameters, seed, MA orders, frequency, and seasonal period, exact NumPy array equality passes for:

- standalone mean TBMA forecasts;
- the complete tree-level TBMA representation;
- 25th/50th/75th-percentile summaries;
- two-component PCA plus mean summaries on fitted training rows;
- three-component PCA-only summaries on fitted training rows;
- 75% explained-variance PCA summaries on fitted training rows;
- sequence-valued MA orders.

The default raw representation therefore preserves the tree-level feature values and ordering used by the source implementation, while the public column names use clearer library terminology.

PCA now fits its basis from fitted training TBMA features and reuses that basis for later calls. This preserves the source calculation when PCA is generated on the fitted training rows while making train/test PCA coordinates consistent for general use.

## Package test suite

The release suite covers, among other cases:

- DataFrame and NumPy inputs;
- timestamp handling and frequency inference;
- seconds, minutes, hours, days, business days, weeks, month starts/ends, quarter starts/ends, and year starts/ends;
- timezone-aware daily and weekly shifts across daylight-saving transitions;
- explicit seasonal periods;
- integer and sequence-valued MA orders;
- standalone multi-horizon forecasts;
- full tree-level feature generation;
- quantile summaries;
- fixed-component and variance-threshold PCA summaries;
- cached training-fitted PCA bases and later subset transformations;
- zero-variance PCA input;
- PCA component-count errors;
- repeated-fit determinism and parameter non-mutation;
- duplicate DataFrame indexes during bootstrap mapping;
- feature-column reordering and schema errors;
- malformed predictors, targets, dates, windows, summaries, seeds, and configuration;
- internal defensive guards normally protected by the public API;
- package-version fallback when distribution metadata is unavailable;
- save/load round trips;
- scikit-learn cloning and fitted-state checks.

The reusable-library suite passes **141 tests** with **100% statement coverage and 100% branch coverage** across package source. The separate repository-only paper workflow adds **9 tests**, for **150 passing tests** across the full GitHub checkout. No package lines or branches are excluded from coverage.

## Repository-only paper-workflow validation

`paper_reproduction/paper_tsf_workflow.py` implements the manuscript evaluation path outside the installable package. It reads dataset-specific settings from the `repo_data` sheet of the repository-root `dataset_info.xlsx`: `Dataset`, `file_name`, `horizon`, `predetermined_lag`, `run`, optional `integer_conversion`, and optional `evaluate`. Additional workbook columns are ignored by this workflow.

The repository-only test exercises the supplied workbook reader plus a complete generated TSF run with the exact downstream estimator classes `CatBoostRegressor`, `MultiTaskElasticNetCV`, and `RandomForestRegressor`. It covers training-only mean scaling, seasonal-dominance detection, optional seasonal differencing, global autoregressive construction, TBMA fitting, full tree-level feature augmentation, standalone readout, original-scale reconstruction, integer-output rounding, MASE, per-seed aggregation, evaluation filtering, SPD/win-count tables, and Wilcoxon/Holm output.

The smoke test substitutes smaller tree/iteration counts for speed but uses the same code path as the manuscript defaults. The default workflow itself uses 256-tree TBMA and RF models, 512 CatBoost iterations, MA orders `{1, 2, 3, 4, 5}`, `feature_window=1`, and seeds 1 through 10.

The paper reproduction code is intentionally excluded from both the wheel and sdist. `dataset_info.xlsx`, `Datasets/`, and generated `paper_results/` are also excluded. Consequently the PyPI library has no CatBoost/OpenPyXL dependency and no dataset-experiment API.

## Distribution checks

The wheel and source distribution are built with the configured setuptools PEP 517 backend. A repository check asserts that neither artifact contains `paper_reproduction/`, `dataset_info.xlsx`, `Datasets/`, or `paper_results/`, and that CatBoost/OpenPyXL do not appear as package requirements. The built wheel is installed separately and exercised for import, fitting, prediction, and default feature generation; the extracted sdist reruns the 141-test package suite at 100% statement and branch coverage. CI additionally runs `twine check` on both artifacts before publishing.

## Cross-platform release audit

The repository is designed to support the three major GitHub-hosted operating
systems: Linux, Windows, and macOS. The reusable package is pure Python and the
wheel is tagged `py3-none-any`; NumPy, pandas, SciPy, and scikit-learn remain
external dependencies whose own platform wheels determine installability on
less common systems.

The CI and release workflows now gate the **built wheel**, rather than only an
editable source checkout, on 15 core combinations:

- `ubuntu-latest` with Python 3.10, 3.11, 3.12, 3.13, and 3.14;
- `windows-latest` with Python 3.10, 3.11, 3.12, 3.13, and 3.14;
- `macos-latest` with Python 3.10, 3.11, 3.12, 3.13, and 3.14.

Each core job builds a wheel, installs that wheel with the active interpreter,
executes the complete package suite with the 100% statement/branch coverage
gate, and executes `examples/basic_forecasting.py` from the installed artifact.
Ruff 0.16.4 runs once in a separate Ubuntu lint job, so a lint failure is
distinguished from operating-system or Python-version test failures. The release
build cannot proceed unless lint, all core matrix jobs, and all paper-workflow
jobs succeed.

The repository-only paper reproduction workflow is independently tested on
`ubuntu-latest`, `windows-latest`, and `macos-latest` with Python 3.13 after
installing the built TBMA wheel plus `paper_reproduction/requirements.txt`.
This covers the CatBoost, MultiTaskElasticNetCV, RandomForestRegressor,
OpenPyXL, workbook, TSF, and result-writing path on each operating system.

Platform-focused tests additionally exercise persistence beneath nested paths
containing spaces and Unicode characters, `PathLike`/string persistence paths,
CRLF-terminated CP1252 TSF input, Unicode/space dataset paths, and Unicode/space
result directories. `.gitattributes` normalizes repository text files so Git
checkout line-ending behavior does not change source/configuration content.

### Validation that can be performed in this build environment

This artifact was assembled in a Linux container with Python 3.13.5. In that
environment, the reusable-library suite passes **141 tests** with **100% statement
coverage and 100% branch coverage** over `src/tbma`, and the separate paper
workflow passes **9 tests**, for **150 passing tests** in the checkout.

A Linux container cannot itself execute Windows or macOS binaries/runners. A
release must therefore not be described as *actually executed on all three
operating systems* until the repository's GitHub Actions matrix has completed
green on the corresponding hosted runners. Both `ci.yml` and `publish.yml`
enforce those real-runner checks before release artifacts can be published.
The previous GitHub run stopped at Ruff before pytest; the reported 24 lint
findings were corrected in this revision, and lint is now a dedicated job.
## Release metadata validation

The release metadata declares the MIT license and the confirmed authors Mustafa Baydoğan, Berk Görgülü, and Oğuzhan Engin. The source distribution includes `LICENSE` and `CITATION.cff`; the wheel includes the MIT license file under its distribution metadata. The canonical repository metadata points to `https://github.com/OguzhanEngin/tbma`, with issues at `https://github.com/OguzhanEngin/tbma/issues`.



## pandas 3 compatibility

The CI matrix intentionally installs current dependency versions. TBMA copies the
forward-filled node-PMA table into a writable NumPy buffer before parent-node
fallback assignment, and normalizes the legacy pandas aliases `T`, `H`, `M`, `Q`,
and `Y` to their current equivalents before frequency parsing.
