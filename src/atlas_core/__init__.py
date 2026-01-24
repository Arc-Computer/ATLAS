"""Atlas Core: offline training engine for adaptive agents."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("atlas_core")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
