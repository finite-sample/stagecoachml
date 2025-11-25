"""StagecoachML - A powerful machine learning pipeline orchestration framework."""

from importlib.metadata import PackageNotFoundError, version

from stagecoachml.pipeline import Pipeline
from stagecoachml.stage import Stage

try:
    __version__ = version("stagecoachml")
except PackageNotFoundError:
    # Package is not installed, use fallback version
    __version__ = "0.1.0"

__all__ = ["Pipeline", "Stage", "__version__"]
