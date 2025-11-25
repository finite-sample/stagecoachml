"""Shared fixtures and configuration for tests."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample dataframe for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    return df


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """Create a simple pipeline for testing."""
    pipeline = Pipeline(name="test_pipeline")

    def stage1_func(context):
        return {"data": [1, 2, 3]}

    def stage2_func(context):
        data = context.get("stage1", {}).get("data", [])
        return {"processed": [x * 2 for x in data]}

    stage1 = FunctionStage(
        name="stage1",
        description="First stage",
        func=stage1_func,
    )

    stage2 = FunctionStage(
        name="stage2",
        description="Second stage",
        func=stage2_func,
    )

    pipeline.add_stage(stage1)
    pipeline.add_stage(stage2)
    pipeline.add_dependency("stage1", "stage2")

    return pipeline


@pytest.fixture
def pipeline_config() -> dict:
    """Sample pipeline configuration dictionary."""
    return {
        "pipeline": {
            "name": "test_pipeline",
            "description": "Test pipeline for ML workflow",
        },
        "stages": [
            {
                "name": "load_data",
                "type": "data_loader",
                "source_type": "csv",
                "source_path": "data.csv",
            },
            {
                "name": "preprocess",
                "type": "transform",
                "input_key": "data",
                "output_key": "features",
            },
            {
                "name": "train_model",
                "type": "model",
                "model_type": "train",
                "model_class": "RandomForest",
            },
        ],
        "dependencies": [
            ["load_data", "preprocess"],
            ["preprocess", "train_model"],
        ],
    }
