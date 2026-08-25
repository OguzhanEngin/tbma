"""Verify that PyPI artifacts contain only reusable-library material."""

from __future__ import annotations

import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path


FORBIDDEN_PARTS = {
    "paper_reproduction",
    "dataset_info.xlsx",
    "Datasets",
    "paper_results",
}
FORBIDDEN_REQUIREMENTS = {"catboost", "openpyxl"}
NATIVE_SUFFIXES = {".so", ".pyd", ".dll", ".dylib"}
EXPECTED_AUTHOR = "Mustafa Baydoğan, Berk Görgülü, Oğuzhan Engin"
EXPECTED_LICENSE = "MIT"
EXPECTED_PROJECT_URLS = {
    "Homepage": "https://github.com/OguzhanEngin/tbma",
    "Repository": "https://github.com/OguzhanEngin/tbma",
    "Issues": "https://github.com/OguzhanEngin/tbma/issues",
}


def _contains_forbidden(name: str) -> bool:
    return any(part in FORBIDDEN_PARTS for part in Path(name).parts)


def _wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _sdist_names(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(
                f"Expected exactly one wheel METADATA file, got {metadata_names}"
            )
        return archive.read(metadata_names[0]).decode("utf-8")


def _wheel_file_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_names) != 1:
            raise RuntimeError(
                f"Expected exactly one wheel WHEEL metadata file, got {wheel_names}"
            )
        return archive.read(wheel_names[0]).decode("utf-8")


def main() -> None:
    dist = Path("dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"Expected one wheel and one sdist, found {len(wheels)} and {len(sdists)}"
        )

    for archive_path, names in (
        (wheels[0], _wheel_names(wheels[0])),
        (sdists[0], _sdist_names(sdists[0])),
    ):
        forbidden = sorted(name for name in names if _contains_forbidden(name))
        if forbidden:
            raise SystemExit(
                f"Repository-only material leaked into {archive_path.name}: {forbidden}"
            )

    wheel_names = _wheel_names(wheels[0])
    native_files = sorted(
        name for name in wheel_names if Path(name).suffix.lower() in NATIVE_SUFFIXES
    )
    if native_files:
        raise SystemExit(
            f"Platform-specific native files found in wheel: {native_files}"
        )

    if not wheels[0].name.endswith("-py3-none-any.whl"):
        raise SystemExit(f"Wheel is not platform-independent: {wheels[0].name}")

    wheel_metadata = Parser().parsestr(_wheel_file_metadata(wheels[0]))
    if wheel_metadata.get("Root-Is-Purelib") != "true":
        raise SystemExit("Wheel does not declare Root-Is-Purelib: true")
    if "py3-none-any" not in wheel_metadata.get_all("Tag", []):
        raise SystemExit("Wheel metadata does not contain Tag: py3-none-any")

    metadata = Parser().parsestr(_wheel_metadata(wheels[0]))
    classifiers = metadata.get_all("Classifier", [])
    if "Operating System :: OS Independent" not in classifiers:
        raise SystemExit("Wheel metadata does not declare OS-independent support")

    if metadata.get("Author") != EXPECTED_AUTHOR:
        raise SystemExit(
            "Wheel author metadata does not match the confirmed project authors: "
            f"{metadata.get('Author')!r}"
        )
    if metadata.get("License-Expression") != EXPECTED_LICENSE:
        raise SystemExit(
            "Wheel license metadata is not MIT: "
            f"{metadata.get('License-Expression')!r}"
        )
    if "LICENSE" not in metadata.get_all("License-File", []):
        raise SystemExit("Wheel metadata does not declare LICENSE as a license file")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel_names):
        raise SystemExit("Wheel does not contain the MIT LICENSE file")

    project_urls = {}
    for item in metadata.get_all("Project-URL", []):
        label, separator, url = item.partition(", ")
        if separator:
            project_urls[label] = url
    if project_urls != EXPECTED_PROJECT_URLS:
        raise SystemExit(
            "Wheel project URLs do not match the canonical repository metadata: "
            f"{project_urls!r}"
        )

    sdist_names = _sdist_names(sdists[0])
    for required_name in ("LICENSE", "CITATION.cff"):
        if not any(name.endswith(f"/{required_name}") for name in sdist_names):
            raise SystemExit(f"Source distribution is missing {required_name}")

    requirements = metadata.get_all("Requires-Dist", [])
    leaked_dependencies = sorted(
        requirement
        for requirement in requirements
        if any(
            requirement.lower().startswith(package)
            for package in FORBIDDEN_REQUIREMENTS
        )
    )
    if leaked_dependencies:
        raise SystemExit(
            "Repository-only dependencies leaked into wheel metadata: "
            f"{leaked_dependencies}"
        )

    print("Distribution separation check passed.")


if __name__ == "__main__":
    main()
