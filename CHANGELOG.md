# Changelog

## 0.1.0

- Added final project URLs for `https://github.com/OguzhanEngin/tbma` and its issue tracker to package and citation metadata.
- Initial Python package release of Tree-Based Moving Average (TBMA).
- Added the MIT license, confirmed author metadata (Mustafa Baydoğan, Berk Görgülü, and Oğuzhan Engin), and `CITATION.cff`.
- Added a scikit-learn-style `TBMA` estimator with explicit `fit`, `predict`, feature-generation, and persistence APIs.
- Exposed moving-average order(s) directly through `ma_order`; seasonal metadata never chooses MA orders automatically.
- Made the full tree-level TBMA representation the default output of `generate_features()`.
- Separated the feature window (`feature_window`) from the standalone forecast horizon (`horizon`).
- Added optional `summary_method="quantile"` and `summary_method="pca"` feature summaries.
- Made PCA summaries fit their basis on training TBMA features and reuse that basis for later transformations.
- Added explicit pandas-style calendar frequency support, including business days and month/quarter/year start and end frequencies.
- Made daily and weekly timezone-aware shifts preserve local wall-clock time across daylight-saving transitions.
- Added explicit random-seed range validation.
- Removed file-system side effects and dataset-specific preparation assumptions from fitting.
- Added deterministic repeated fitting, feature-schema validation, and duplicate-index-safe bootstrap mapping.
- Added a 100% statement-and-branch coverage gate to tests and CI.
- Added CI, distribution validation, and PyPI Trusted Publishing workflows.
- Fixed the first GitHub CI lint audit and separated Ruff 0.16.4 into a dedicated lint job so lint failures no longer appear as failures for every OS/Python matrix entry.
- Added cross-platform release gating: the built wheel is tested on Linux, Windows, and macOS for Python 3.10–3.14, and the repository-only paper workflow is tested on all three operating systems with Python 3.13.
- Added platform-focused regression tests for nested Unicode/space paths, CRLF TSF input, portable persistence paths, and result directories, plus repository line-ending normalization through `.gitattributes`.
- Added a GitHub-only paper reproduction workflow using `RandomForestRegressor`, `MultiTaskElasticNetCV`, and `CatBoostRegressor`. It is driven by repository-root `dataset_info.xlsx`, honors `run`, `horizon`, `predetermined_lag`, `integer_conversion`, and `evaluate`, and is explicitly excluded from PyPI artifacts and dependencies.
- Removed the former `paper` package extra; CatBoost and OpenPyXL are now listed only in `paper_reproduction/requirements.txt`, and CI verifies that repository-only reproduction material cannot leak into wheel/sdist artifacts.
