# Publishing TBMA

The project uses a PEP 517 setuptools build and GitHub Actions with PyPI Trusted Publishing.

## 1. Repository and release metadata

The release candidate already contains the confirmed project identity metadata:

- MIT license in `LICENSE` and `project.license = "MIT"` metadata;
- authors Mustafa Baydoğan, Berk Görgülü, and Oğuzhan Engin in `pyproject.toml`;
- `CITATION.cff` for GitHub/software citation metadata.

The canonical GitHub repository is `https://github.com/OguzhanEngin/tbma`. `pyproject.toml`, `CITATION.cff`, and the README already contain the canonical repository and issue-tracker URLs.

The configured distribution name is `tbma`. PyPI names are globally unique and can be unavailable even when no public release page exists, so confirm that the project name can be registered before the first upload.

For the initial push, the remote should be:

```bash
git remote add origin https://github.com/OguzhanEngin/tbma.git
git push -u origin main
```

## 2. Run release checks locally

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -r paper_reproduction/requirements.txt
ruff check .
pytest --cov=tbma --cov-branch --cov-report=term-missing --cov-fail-under=100
pytest -q paper_reproduction/tests/test_paper_tsf_workflow.py
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python tools/check_distribution.py
```

Then install the built wheel in a fresh environment and run a smoke test rather than relying only on an editable installation.

## 3. Configure Trusted Publishing

The release workflow is `.github/workflows/publish.yml`.

Create GitHub Environments named:

- `testpypi`
- `pypi`

Configure the production `pypi` environment with required manual approval before deployment. In PyPI and TestPyPI, register the GitHub repository/workflow as a Trusted Publisher with the matching environment name. Use these values:

- GitHub owner: `OguzhanEngin`
- Repository: `tbma`
- Workflow filename: `publish.yml`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`

The workflow uses OIDC and therefore does not require a long-lived PyPI token in GitHub Secrets.

## 4. Verify CI on every supported Python version

Before tagging a production release, require a successful CI run across Python 3.10, 3.11, 3.12, 3.13, and 3.14 on Linux, Windows, and macOS. Ruff 0.16.4 runs once in a dedicated Ubuntu lint job, while the OS/Python matrix independently enforces 100% statement and branch coverage against the installed wheel. Dedicated Python 3.13 jobs separately install the repository-only `paper_reproduction/requirements.txt` dependencies and run the complete synthetic `.tsf` workflow test on Linux, Windows, and macOS, including the workbook-driven configuration path, CatBoost, Multi-task Elastic Net, Random Forest, TBMA feature augmentation, and standalone TBMA.

The publishing workflow repeats both the core release tests and the repository-only paper-reproduction test before it builds distributions. The reproduction directory and `dataset_info.xlsx` are deliberately excluded from both PyPI artifacts.

## 5. TestPyPI

Run the `Publish distributions` workflow manually. A manual run publishes the built artifacts to TestPyPI through the `testpypi` environment.

Install that release in a clean environment and exercise the public API before creating the production tag.

## 6. PyPI

The production workflow checks that the Git tag version matches `project.version` in `pyproject.toml`.

For version 0.1.0:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Only a pushed `v*` tag can enter the production PyPI job. PyPI release files are immutable; publish fixes under a new version rather than attempting to overwrite an existing release.

## Repository-only reproduction material

Before publishing, verify that `paper_reproduction/`, `dataset_info.xlsx`, `Datasets/`, and `paper_results/` are present only in the GitHub/source checkout and are absent from both files under `dist/`. The PyPI metadata must not expose CatBoost or OpenPyXL as core or optional package dependencies.

## Cross-platform release gate

Do not tag or publish `0.1.0` until the GitHub CI run for the release commit is
green on the entire platform matrix. The required core jobs are Python
3.10–3.14 on each of `ubuntu-latest`, `windows-latest`, and `macos-latest`.
The repository-only paper workflow must also pass on all three operating systems
with Python 3.13.

The jobs build and install the wheel before testing, so the matrix validates the
artifact users will install rather than only an editable checkout. Production
publishing repeats the same cross-platform gates before the build/publish job.
