"""Reusable upstream computation and independent online prediction."""

from importlib.metadata import version

from stagecoachml.classification import StagecoachClassifier
from stagecoachml.encoding import PredictionEncoder
from stagecoachml.regression import StagecoachRegressor
from stagecoachml.serving import OnlinePredictor
from stagecoachml.state import EncodedState

__version__ = version("stagecoachml")

__all__ = [
    "EncodedState",
    "OnlinePredictor",
    "PredictionEncoder",
    "StagecoachClassifier",
    "StagecoachRegressor",
    "__version__",
]
