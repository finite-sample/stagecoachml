# Stage API

```{eval-rst}
.. automodule:: stagecoachml.stage
   :members:
   :undoc-members:
   :show-inheritance:
```

## Base Stage Class

```{eval-rst}
.. autoclass:: stagecoachml.stage.Stage
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Built-in Stages

### FunctionStage

```{eval-rst}
.. autoclass:: stagecoachml.stage.FunctionStage
   :members:
   :special-members: __init__
   :show-inheritance:
```

### DataLoaderStage

```{eval-rst}
.. autoclass:: stagecoachml.stage.DataLoaderStage
   :members:
   :special-members: __init__
   :show-inheritance:
```

### TransformStage

```{eval-rst}
.. autoclass:: stagecoachml.stage.TransformStage
   :members:
   :special-members: __init__
   :show-inheritance:
```

### ModelStage

```{eval-rst}
.. autoclass:: stagecoachml.stage.ModelStage
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Usage Examples

### Function Stage

```python
from stagecoachml.stage import FunctionStage

def my_processing_function(context):
    data = context.get("input_data", [])
    processed = [x * 2 for x in data]
    return {"processed_data": processed}

stage = FunctionStage(
    name="process_data",
    func=my_processing_function,
    description="Double all input values"
)
```

### Data Loader Stage

```python
from stagecoachml.stage import DataLoaderStage

# Load CSV data
loader = DataLoaderStage(
    name="load_csv",
    source_type="csv",
    source_path="data.csv",
    output_key="raw_data"
)

# Load Parquet data
loader = DataLoaderStage(
    name="load_parquet",
    source_type="parquet",
    source_path="data.parquet"
)
```

### Transform Stage

```python
from stagecoachml.stage import TransformStage

def normalize_data(data):
    # Custom normalization logic
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

transformer = TransformStage(
    name="normalize",
    input_key="raw_data",
    output_key="normalized_data",
    transform_func=normalize_data
)
```

### Model Stage

```python
from stagecoachml.stage import ModelStage

# Training stage
trainer = ModelStage(
    name="train_classifier",
    model_type="train",
    model_class="RandomForest",
    input_features="X_train",
    input_target="y_train"
)

# Prediction stage
predictor = ModelStage(
    name="make_predictions",
    model_type="predict",
    input_features="X_test"
)
```

## Creating Custom Stages

```python
from stagecoachml.stage import Stage
from typing import Any, Dict

class CustomStage(Stage):
    """A custom processing stage."""
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute custom logic."""
        # Your custom implementation here
        input_data = context.get("input_key")
        
        # Perform custom processing
        result = self.custom_processing(input_data)
        
        return {"output_key": result}
    
    def custom_processing(self, data):
        """Custom processing logic."""
        # Implement your processing here
        return data
    
    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate stage inputs."""
        return "input_key" in context
    
    def validate_outputs(self, result: Any) -> bool:
        """Validate stage outputs."""
        return isinstance(result, dict) and "output_key" in result
```