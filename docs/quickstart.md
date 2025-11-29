# Quick Start Guide

This guide will get you up and running with StagecoachML in just a few minutes.

## Installation

First, install StagecoachML:

```bash
# Using pip
pip install stagecoachml

# Using uv (recommended)
uv pip install stagecoachml
```

## Your First Pipeline

Let's create a simple ML pipeline that loads data, preprocesses it, and trains a model:

```python
from stagecoachml import Pipeline
from stagecoachml.stage import DataLoaderStage, FunctionStage, ModelStage

# Create a new pipeline
pipeline = Pipeline(name="my_first_pipeline")

# Stage 1: Load some data
data_loader = DataLoaderStage(
    name="load_data",
    source_type="csv",
    source_path="data.csv"
)
pipeline.add_stage(data_loader)

# Stage 2: Preprocess the data
def preprocess_data(context):
    df = context["load_data"]["data"]
    # Simple preprocessing - drop null values
    df_clean = df.dropna()
    return {"clean_data": df_clean}

preprocessor = FunctionStage(
    name="preprocess",
    func=preprocess_data
)
pipeline.add_stage(preprocessor)

# Stage 3: Train a model
trainer = ModelStage(
    name="train_model",
    model_type="train",
    model_class="RandomForest"
)
pipeline.add_stage(trainer)

# Define the execution order
pipeline.add_dependency("load_data", "preprocess")
pipeline.add_dependency("preprocess", "train_model")

# Run the pipeline
results = pipeline.run()

print("Pipeline completed!")
print(f"Trained model: {results['train_model']['model']}")
```

## Understanding the Output

When you run a pipeline, StagecoachML returns a dictionary containing the outputs from each stage:

```python
{
    "load_data": {"data": <pandas.DataFrame>},
    "preprocess": {"clean_data": <pandas.DataFrame>},
    "train_model": {"model": <sklearn.ensemble.RandomForestClassifier>}
}
```

## Visualizing Your Pipeline

See the structure of your pipeline:

```python
print(pipeline.visualize())
```

Output:
```
Pipeline: my_first_pipeline
========================================
Stage: load_data
  Leads to: preprocess

Stage: preprocess
  Dependencies: load_data
  Leads to: train_model

Stage: train_model
  Dependencies: preprocess
```

## Next Steps

- Learn about [different types of stages](user_guide/stages.md)
- Explore [pipeline configuration](user_guide/configuration.md)
- Check out [advanced examples](examples/index.md)
- Read the [API reference](api/index.md)