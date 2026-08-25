"""Tree-Based Moving Average forecasting."""

from importlib.metadata import PackageNotFoundError, version

from .model import TBMA

try:
    __version__ = version("tbma")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0.1.0"

__all__ = ["TBMA", "__version__"]
