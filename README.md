# StagecoachML

[![PyPI - Version](https://img.shields.io/pypi/v/stagecoachml.svg)](https://pypi.org/project/stagecoachml)
[![Tests](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml)
[![Documentation](https://github.com/finite-sample/stagecoachml/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/stagecoachml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful and intuitive machine learning pipeline orchestration framework that makes building, managing, and deploying ML workflows as easy as riding a stagecoach.

## Features

✨ **Simple & Intuitive** - Build complex ML pipelines with a clean, Pythonic API

🚀 **Fast & Efficient** - Optimized DAG execution with intelligent caching

🔧 **Flexible** - Support for various data sources, transformations, and models

📊 **Comprehensive** - Built-in stages for common ML tasks

🎯 **Type-Safe** - Full type hints and runtime validation with Pydantic

🛠️ **Extensible** - Easy to create custom stages and integrate with any ML framework

## Installation

```bash
# Using pip
pip install stagecoachml

# Using uv (recommended)
uv pip install stagecoachml

# With visualization support
pip install stagecoachml[viz]

# For development
pip install stagecoachml[dev]
```

## Quick Start

```python
from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a pipeline
pipeline = Pipeline(name="iris_classifier")

# Stage 1: Load iris dataset
def load_iris_data(context):
    iris = load_iris()
    return {"X": iris.data, "y": iris.target}

# Stage 2: Split data
def split_data(context):
    X, y = context["load_data"]["X"], context["load_data"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

# Stage 3: Train model
def train_model(context):
    X_train = context["split_data"]["X_train"]
    y_train = context["split_data"]["y_train"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return {"model": model}

# Stage 4: Evaluate model
def evaluate_model(context):
    model = context["train_model"]["model"]
    X_test = context["split_data"]["X_test"]
    y_test = context["split_data"]["y_test"]
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {"accuracy": accuracy, "predictions": predictions}

# Add stages to pipeline
pipeline.add_stage(FunctionStage(name="load_data", func=load_iris_data))
pipeline.add_stage(FunctionStage(name="split_data", func=split_data))
pipeline.add_stage(FunctionStage(name="train_model", func=train_model))
pipeline.add_stage(FunctionStage(name="evaluate", func=evaluate_model))

# Define dependencies
pipeline.add_dependency("load_data", "split_data")
pipeline.add_dependency("split_data", "train_model")
pipeline.add_dependency("train_model", "evaluate")

# Run the pipeline
results = pipeline.run()
print(f"Model accuracy: {results['evaluate']['accuracy']:.4f}")
```

## Core Concepts

### Pipelines

Pipelines are directed acyclic graphs (DAGs) of stages that define your ML workflow:

```python
from stagecoachml import Pipeline

pipeline = Pipeline(name="my_ml_workflow")
```

### Stages

Stages are the building blocks of pipelines. Use built-in stages or create custom ones:

```python
from stagecoachml.stage import FunctionStage

def preprocess_data(context):
    df = context["raw_data"]
    # Perform preprocessing
    return {"processed_data": df}

stage = FunctionStage(
    name="preprocess",
    func=preprocess_data
)
```

### Dependencies

Define execution order by specifying dependencies between stages:

```python
pipeline.add_dependency("load_data", "preprocess")
pipeline.add_dependency("preprocess", "train_model")
```

## Real Examples

StagecoachML comes with comprehensive examples using real datasets:

### 🌸 Iris Classification
```bash
python examples/iris_classification/iris_pipeline.py
```
Complete classification workflow with multiple models, feature importance, and sample predictions.

### 🏠 California Housing Regression  
```bash
python examples/boston_housing/housing_pipeline.py
```
Regression pipeline comparing Linear, Ridge, Random Forest, and Gradient Boosting models on California housing dataset.

### 🔢 Handwritten Digits Recognition
```bash
python examples/digits_recognition/digits_pipeline.py
```
Computer vision pipeline with SVM, Logistic Regression, and Random Forest on 8×8 pixel images.

### 🔧 Custom Stages
```bash
python examples/custom_stages/custom_pipeline.py
```
Advanced example showing how to create custom stages with validation, retry logic, and configuration.

## Advanced Example

```python
from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline(name="housing_regression")

# Stage 1: Load California housing data
def load_housing_data(context):
    california = fetch_california_housing()
    return {"X": california.data, "y": california.target}

# Stage 2: Split data
def split_data(context):
    X, y = context["load_data"]["X"], context["load_data"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train, "X_test": X_test, 
        "y_train": y_train, "y_test": y_test
    }

pipeline.add_stage(FunctionStage(name="load_data", func=load_housing_data))
pipeline.add_stage(FunctionStage(name="split", func=split_data))

# Stage 3: Scale features  
def scale_features(context):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(context["split"]["X_train"])
    X_test = scaler.transform(context["split"]["X_test"])
    return {"X_train_scaled": X_train, "X_test_scaled": X_test, "scaler": scaler}

# Stage 4: Train model
def train_model(context):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(context["scale"]["X_train_scaled"], context["split"]["y_train"])
    return {"model": model}

# Stage 5: Evaluate
def evaluate(context):
    from sklearn.metrics import mean_squared_error, r2_score
    model = context["train"]["model"]
    X_test, y_test = context["scale"]["X_test_scaled"], context["split"]["y_test"]
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"rmse": rmse, "r2": r2, "predictions": predictions}

# Build pipeline
for stage in [
    FunctionStage(name="scale", func=scale_features),
    FunctionStage(name="train", func=train_model),
    FunctionStage(name="evaluate", func=evaluate)
]:
    pipeline.add_stage(stage)

# Define dependencies
for dep in [("load_data", "split"), ("split", "scale"), ("scale", "train"), ("train", "evaluate")]:
    pipeline.add_dependency(dep[0], dep[1])

# Run pipeline
results = pipeline.run()
print(f"Model RMSE: ${results['evaluate']['rmse']:.2f}k")
print(f"R² Score: {results['evaluate']['r2']:.4f}")
```

## CLI Usage

StagecoachML comes with a powerful CLI for managing pipelines:

```bash
# Show version
stagecoach version

# Run a pipeline from YAML config
stagecoach run pipeline.yaml

# Validate pipeline configuration
stagecoach validate pipeline.yaml

# List stages in a pipeline
stagecoach list-stages pipeline.yaml

# Dry run to see execution plan
stagecoach run pipeline.yaml --dry-run
```

## Configuration Files

Define pipelines in YAML for reusability:

```yaml
pipeline:
  name: iris_classifier
  description: Classify iris species

stages:
  - name: load_data
    type: data_loader
    source_type: csv
    source_path: iris.csv
    
  - name: preprocess
    type: transform
    input_key: data
    output_key: features
    
  - name: train_model
    type: model
    model_type: train
    model_class: RandomForest
    
dependencies:
  - [load_data, preprocess]
  - [preprocess, train_model]
```

## Built-in Stages

- **DataLoaderStage** - Load data from CSV, Parquet, JSON
- **TransformStage** - Apply transformations to data
- **ModelStage** - Train and evaluate ML models
- **FunctionStage** - Wrap any Python function as a stage

## Creating Custom Stages

```python
from stagecoachml.stage import Stage

class CustomProcessingStage(Stage):
    def execute(self, context):
        # Your custom logic here
        data = context.get("input_data")
        processed = self.custom_process(data)
        return {"output": processed}
    
    def custom_process(self, data):
        # Implementation
        return data
```

## Development

```bash
# Clone the repository
git clone https://github.com/finite-sample/stagecoachml.git
cd stagecoachml

# Install with development dependencies using uv
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .

# Run type checking
mypy src/stagecoachml
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 [Documentation](https://finite-sample.github.io/stagecoachml/)
- 🐛 [Issue Tracker](https://github.com/finite-sample/stagecoachml/issues)
- 💬 [Discussions](https://github.com/finite-sample/stagecoachml/discussions)

## Acknowledgments

Built with ❤️ using:
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [NetworkX](https://networkx.org/) for graph operations
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [Typer](https://typer.tiangolo.com/) for the CLI

## Citation

If you use StagecoachML in your research, please cite:

```bibtex
@software{stagecoachml,
  author = {Sood, Gaurav},
  title = {StagecoachML: Machine Learning Pipeline Orchestration Framework},
  year = {2024},
  url = {https://github.com/finite-sample/stagecoachml}
}
```
