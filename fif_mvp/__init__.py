"""FIF MVP package."""

from importlib import metadata


def get_version() -> str:
    """Return package version if available, else placeholder."""
    try:
        return metadata.version("fif_mvp")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]
