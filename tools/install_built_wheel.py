"""Install the single wheel in ``dist`` using the active Python interpreter.

This helper exists so CI can install a freshly built wheel with the same command
on Linux, macOS, and Windows without relying on shell-specific glob expansion.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one wheel in dist/, found {wheels}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", str(wheels[0])]
    )


if __name__ == "__main__":
    main()
