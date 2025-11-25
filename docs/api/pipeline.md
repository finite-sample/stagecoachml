# Pipeline API

```{eval-rst}
.. automodule:: stagecoachml.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

## Pipeline Class

```{eval-rst}
.. autoclass:: stagecoachml.pipeline.Pipeline
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Usage Examples

### Creating a Pipeline

```python
from stagecoachml import Pipeline

# Create an empty pipeline
pipeline = Pipeline(name="my_pipeline")

# Create with configuration
pipeline = Pipeline(
    name="configured_pipeline",
    config={
        "max_retries": 3,
        "timeout": 300
    }
)
```

### Adding Stages

```python
from stagecoachml.stage import FunctionStage

def my_function(context):
    return {"result": "success"}

stage = FunctionStage(name="my_stage", func=my_function)
pipeline.add_stage(stage)
```

### Defining Dependencies

```python
# Add a dependency between stages
pipeline.add_dependency("stage1", "stage2")

# This ensures stage1 runs before stage2
```

### Running the Pipeline

```python
# Run with default context
results = pipeline.run()

# Run with custom context
results = pipeline.run({"custom_param": "value"})
```

### Validation

```python
# Validate pipeline structure
is_valid = pipeline.validate()

# Get execution order
order = pipeline.get_execution_order()
print(f"Stages will execute in order: {order}")
```