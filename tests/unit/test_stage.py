"""Unit tests for stage module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stagecoachml.stage import (
    DataLoaderStage,
    FunctionStage,
    ModelStage,
    Stage,
    TransformStage,
)


class ConcreteStage(Stage):
    """Concrete implementation of Stage for testing."""

    def execute(self, context):
        return {"result": "executed"}


class TestStageBase:
    """Test base Stage class."""

    def test_stage_abstract(self):
        """Test that Stage is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Stage(name="test")

    def test_concrete_stage(self):
        """Test concrete stage implementation."""
        stage = ConcreteStage(name="test", description="Test stage")
        assert stage.name == "test"
        assert stage.description == "Test stage"
        assert stage.retry_count == 0
        assert stage.timeout is None

    def test_stage_validation(self):
        """Test stage validation methods."""
        stage = ConcreteStage(name="test")
        assert stage.validate_inputs({})
        assert stage.validate_outputs("anything")


class TestFunctionStage:
    """Test FunctionStage class."""

    def test_function_stage_requires_func(self):
        """Test that FunctionStage requires a function."""
        with pytest.raises(ValueError, match="requires a function"):
            FunctionStage(name="test", func=None)

    def test_function_stage_execution(self):
        """Test FunctionStage executes the wrapped function."""

        def test_func(context):
            return {"value": context.get("input", 0) * 2}

        stage = FunctionStage(name="test", func=test_func)
        result = stage.execute({"input": 5})
        assert result == {"value": 10}

    def test_function_stage_with_validation(self):
        """Test FunctionStage with custom validation."""

        def test_func(context):
            return {"value": 42}

        stage = FunctionStage(name="test", func=test_func)
        stage.validate_inputs = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Invalid inputs"):
            stage.execute({})

    def test_function_stage_output_validation(self):
        """Test FunctionStage output validation."""

        def test_func(context):
            return None

        stage = FunctionStage(name="test", func=test_func)
        stage.validate_outputs = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Invalid outputs"):
            stage.execute({})


class TestTransformStage:
    """Test TransformStage class."""

    def test_transform_stage_basic(self):
        """Test basic TransformStage functionality."""
        stage = TransformStage(
            name="test",
            input_key="data",
            output_key="transformed",
        )
        result = stage.execute({"data": [1, 2, 3]})
        assert result == {"transformed": [1, 2, 3]}

    def test_transform_stage_with_function(self):
        """Test TransformStage with custom transformation."""

        def transform(data):
            return [x * 2 for x in data]

        stage = TransformStage(
            name="test",
            input_key="data",
            output_key="transformed",
            transform_func=transform,
        )
        result = stage.execute({"data": [1, 2, 3]})
        assert result == {"transformed": [2, 4, 6]}

    def test_transform_stage_missing_input(self):
        """Test TransformStage with missing input."""
        stage = TransformStage(
            name="test",
            input_key="data",
            output_key="transformed",
        )
        with pytest.raises(KeyError, match="Required input"):
            stage.execute({"wrong_key": [1, 2, 3]})


class TestDataLoaderStage:
    """Test DataLoaderStage class."""

    def test_data_loader_csv(self, sample_csv_file):
        """Test loading CSV data."""
        stage = DataLoaderStage(
            name="test",
            source_type="csv",
            source_path=str(sample_csv_file),
        )
        result = stage.execute({})
        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 100

    def test_data_loader_from_context(self, sample_csv_file):
        """Test loading data with path from context."""
        stage = DataLoaderStage(
            name="test",
            source_type="csv",
        )
        result = stage.execute({"source_path": str(sample_csv_file)})
        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)

    def test_data_loader_no_path(self):
        """Test data loader with no path provided."""
        stage = DataLoaderStage(
            name="test",
            source_type="csv",
        )
        with pytest.raises(ValueError, match="No source path"):
            stage.execute({})

    def test_data_loader_unsupported_type(self):
        """Test data loader with unsupported source type."""
        stage = DataLoaderStage(
            name="test",
            source_type="unknown",
            source_path="dummy.txt",
        )
        with pytest.raises(ValueError, match="Unsupported source type"):
            stage.execute({})

    @patch("pandas.read_parquet")
    def test_data_loader_parquet(self, mock_read):
        """Test loading Parquet data."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_read.return_value = mock_df

        stage = DataLoaderStage(
            name="test",
            source_type="parquet",
            source_path="data.parquet",
        )
        result = stage.execute({})
        assert "data" in result
        mock_read.assert_called_once_with("data.parquet")

    @patch("pandas.read_json")
    def test_data_loader_json(self, mock_read):
        """Test loading JSON data."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_read.return_value = mock_df

        stage = DataLoaderStage(
            name="test",
            source_type="json",
            source_path="data.json",
        )
        result = stage.execute({})
        assert "data" in result
        mock_read.assert_called_once_with("data.json")


class TestModelStage:
    """Test ModelStage class."""

    def test_model_train_stage(self):
        """Test model training stage."""
        import numpy as np

        stage = ModelStage(
            name="test",
            model_type="train",
            model_class="RandomForest",
        )

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        result = stage.execute({"features": X, "target": y})
        assert "model" in result
        assert hasattr(result["model"], "predict")

    def test_model_predict_stage(self):
        """Test model prediction stage."""
        import numpy as np

        # Create and train a model first
        train_stage = ModelStage(
            name="train",
            model_type="train",
        )
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        train_result = train_stage.execute({"features": X_train, "target": y_train})

        # Use the model for prediction
        predict_stage = ModelStage(
            name="predict",
            model_type="predict",
        )
        X_test = np.array([[2, 3], [4, 5]])
        predict_result = predict_stage.execute(
            {
                "model": train_result["model"],
                "features": X_test,
            }
        )

        assert "predictions" in predict_result
        assert len(predict_result["predictions"]) == 2

    def test_model_train_missing_data(self):
        """Test model training with missing data."""
        stage = ModelStage(
            name="test",
            model_type="train",
        )
        with pytest.raises(ValueError, match="Training requires"):
            stage.execute({"features": [[1, 2]]})

    def test_model_predict_missing_model(self):
        """Test prediction with missing model."""
        stage = ModelStage(
            name="test",
            model_type="predict",
        )
        with pytest.raises(ValueError, match="Prediction requires"):
            stage.execute({"features": [[1, 2]]})

    def test_model_invalid_type(self):
        """Test model stage with invalid type."""
        stage = ModelStage(
            name="test",
            model_type="invalid",
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            stage.execute({})

    def test_model_logistic_regression(self):
        """Test training with LogisticRegression."""
        import numpy as np

        stage = ModelStage(
            name="test",
            model_type="train",
            model_class="LogisticRegression",
        )

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        result = stage.execute({"features": X, "target": y})
        assert "model" in result
        assert result["model"].__class__.__name__ == "LogisticRegression"
