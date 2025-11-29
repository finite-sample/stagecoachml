"""StagecoachML - A library for two-stage machine learning models."""

from importlib.metadata import PackageNotFoundError, version

from stagecoachml.classification import StagecoachClassifier
from stagecoachml.regression import StagecoachRegressor

try:
    __version__ = version("stagecoachml")
except PackageNotFoundError:
    # Package is not installed, use fallback version
    __version__ = "0.1.0"

__all__ = ["StagecoachClassifier", "StagecoachRegressor", "__version__"]
