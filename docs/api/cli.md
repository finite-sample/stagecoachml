# CLI API

```{eval-rst}
.. automodule:: stagecoachml.cli
   :members:
   :undoc-members:
   :show-inheritance:
```

## Command Reference

### version

Show version information.

```bash
stagecoach version
```

### run

Run a pipeline from a configuration file.

```bash
stagecoach run pipeline.yaml [OPTIONS]
```

**Options:**
- `--verbose, -v`: Enable verbose output
- `--dry-run`: Show execution plan without running

**Examples:**

```bash
# Run a pipeline
stagecoach run my_pipeline.yaml

# Verbose output
stagecoach run my_pipeline.yaml --verbose

# Dry run to see execution plan
stagecoach run my_pipeline.yaml --dry-run
```

### validate

Validate a pipeline configuration file.

```bash
stagecoach validate pipeline.yaml
```

**Examples:**

```bash
# Validate configuration
stagecoach validate my_pipeline.yaml
```

### list-stages

List all stages in a pipeline configuration.

```bash
stagecoach list-stages pipeline.yaml
```

**Examples:**

```bash
# List stages
stagecoach list-stages my_pipeline.yaml
```

## Configuration File Format

The CLI accepts YAML configuration files with the following structure:

```yaml
pipeline:
  name: my_pipeline
  description: Description of the pipeline

stages:
  - name: stage1
    type: data_loader
    source_type: csv
    source_path: data.csv
    
  - name: stage2
    type: transform
    input_key: data
    output_key: features
    
  - name: stage3
    type: model
    model_type: train
    model_class: RandomForest

dependencies:
  - [stage1, stage2]
  - [stage2, stage3]
```

## Usage Examples

### Running a Simple Pipeline

Create a file `iris_pipeline.yaml`:

```yaml
pipeline:
  name: iris_classifier
  description: Classify iris species

stages:
  - name: load_data
    type: data_loader
    source_type: csv
    source_path: iris.csv
    
  - name: train_model
    type: model
    model_type: train
    model_class: RandomForest
    
dependencies:
  - [load_data, train_model]
```

Run it:

```bash
stagecoach run iris_pipeline.yaml
```

### Validation Workflow

```bash
# First validate the configuration
stagecoach validate iris_pipeline.yaml

# Check the stages
stagecoach list-stages iris_pipeline.yaml

# Do a dry run
stagecoach run iris_pipeline.yaml --dry-run

# Actually run it
stagecoach run iris_pipeline.yaml
```