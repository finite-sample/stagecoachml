"""StagecoachML - A library for two-stage machine learning models."""

import logging
from importlib.metadata import version

from stagecoachml.classification import StagecoachClassifier
from stagecoachml.regression import StagecoachRegressor

# Configure logging for the library
_logger = logging.getLogger(__name__)

# Don't configure handlers if they already exist (avoids interfering with user config)
if not _logger.handlers and not logging.getLogger().handlers:
    # Only set up basic config if no logging is configured
    logging.basicConfig(
        level=logging.WARNING,  # Default to WARNING to be quiet by default
        format="%(name)s - %(levelname)s - %(message)s",
    )

__version__ = version("stagecoachml")

__all__ = ["StagecoachClassifier", "StagecoachRegressor", "__version__"]
