"""Stage implementation for pipeline components."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Stage(BaseModel, ABC):
    """Abstract base class for pipeline stages."""

    name: str
    description: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(default=0, ge=0)
    timeout: Optional[float] = Field(default=None, gt=0)

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the stage with given context."""
        pass

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate stage inputs from context."""
        return True

    def validate_outputs(self, result: Any) -> bool:
        """Validate stage outputs."""
        return True


class FunctionStage(Stage):
    """Stage that wraps a Python function."""

    func: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.func is None:
            raise ValueError("FunctionStage requires a function")

    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the wrapped function."""
        if not self.validate_inputs(context):
            raise ValueError(f"Invalid inputs for stage '{self.name}'")

        logger.debug(f"Executing function stage '{self.name}'")
        result = self.func(context)

        if not self.validate_outputs(result):
            raise ValueError(f"Invalid outputs from stage '{self.name}'")

        return result


class TransformStage(Stage):
    """Stage for data transformations."""

    input_key: str
    output_key: str
    transform_func: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def execute(self, context: Dict[str, Any]) -> Any:
        """Apply transformation to data from context."""
        if self.input_key not in context:
            raise KeyError(f"Required input '{self.input_key}' not found in context")

        data = context[self.input_key]
        if self.transform_func:
            transformed = self.transform_func(data)
        else:
            transformed = self._default_transform(data)

        return {self.output_key: transformed}

    def _default_transform(self, data: Any) -> Any:
        """Default passthrough transformation."""
        return data


class DataLoaderStage(Stage):
    """Stage for loading data from various sources."""

    source_type: str  # 'csv', 'parquet', 'json', etc.
    source_path: Optional[str] = None
    output_key: str = "data"

    def execute(self, context: Dict[str, Any]) -> Any:
        """Load data from source."""
        import pandas as pd

        source_path = self.source_path or context.get("source_path")
        if not source_path:
            raise ValueError(f"No source path provided for stage '{self.name}'")

        logger.info(f"Loading data from {source_path}")

        if self.source_type == "csv":
            data = pd.read_csv(source_path)
        elif self.source_type == "parquet":
            data = pd.read_parquet(source_path)
        elif self.source_type == "json":
            data = pd.read_json(source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

        return {self.output_key: data}


class ModelStage(Stage):
    """Stage for model training or prediction."""

    model_type: str  # 'train' or 'predict'
    model_class: Optional[str] = None
    input_features: str = "features"
    input_target: Optional[str] = "target"
    output_key: str = "model"

    def execute(self, context: Dict[str, Any]) -> Any:
        """Train or apply a model."""
        if self.model_type == "train":
            return self._train_model(context)
        elif self.model_type == "predict":
            return self._predict(context)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _train_model(self, context: Dict[str, Any]) -> Any:
        """Train a model with data from context."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        X = context.get(self.input_features)
        y = context.get(self.input_target)

        if X is None or y is None:
            raise ValueError(f"Training requires both {self.input_features} and {self.input_target}")

        if self.model_class == "LogisticRegression":
            model = LogisticRegression(random_state=42)
        elif self.model_class == "RandomForest":
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)
        return {self.output_key: model}

    def _predict(self, context: Dict[str, Any]) -> Any:
        """Make predictions with a trained model."""
        model = context.get("model")
        X = context.get(self.input_features)

        if model is None or X is None:
            raise ValueError(f"Prediction requires both model and {self.input_features}")

        predictions = model.predict(X)
        return {"predictions": predictions}
