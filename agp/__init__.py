from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agp_tool")
except PackageNotFoundError:
    __version__ = "unknown"

from .api import ReportGenerator, generate_report  # noqa: F401

__all__ = ["generate_report", "ReportGenerator", "__version__"]
