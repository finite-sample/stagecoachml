# Custom Stages Example

This example demonstrates how to create custom stages and extend StagecoachML with specialized functionality.

## What This Example Does

1. **Custom Stage Classes**: Shows how to inherit from the base `Stage` class
2. **Input/Output Validation**: Implements custom validation logic
3. **Configuration Parameters**: Uses Pydantic fields for stage configuration
4. **Error Handling**: Demonstrates retry logic and timeout capabilities
5. **Complex Workflows**: Builds a sophisticated ML pipeline using custom components

## Custom Stages Demonstrated

### 1. DataGeneratorStage
- **Purpose**: Generates synthetic classification datasets
- **Features**: Configurable sample size, features, and classes
- **Validation**: Ensures output data meets expected format
- **Configuration**: `n_samples`, `n_features`, `n_classes`, `random_state`

### 2. StatisticalAnalyzerStage  
- **Purpose**: Performs comprehensive statistical analysis
- **Features**: Calculates means, correlations, class distributions
- **Output**: Detailed statistics and feature summaries
- **Validation**: Requires 'X' and 'y' data in context

### 3. CrossValidatorStage
- **Purpose**: Cross-validates multiple models with retry capability
- **Features**: Configurable CV folds, automatic retry on failure
- **Configuration**: `cv_folds`, `retry_count`, `timeout`
- **Error Handling**: Graceful failure handling for individual models

### 4. ModelExplainerStage
- **Purpose**: Provides model interpretability and insights
- **Features**: Feature importance, model-specific analysis
- **Output**: Explanations and top contributing features
- **Intelligence**: Adapts to different model types

### 5. PerformanceBenchmarkStage
- **Purpose**: Analyzes pipeline performance metrics
- **Features**: Memory usage, processing speed, recommendations
- **Output**: Performance insights and optimization suggestions
- **Timing**: Benchmarks various pipeline aspects

## Running the Example

### Method 1: Python Script

```bash
# From the project root directory
python examples/custom_stages/custom_pipeline.py
```

### Method 2: Interactive Usage

```python
import sys
sys.path.insert(0, "src")

from examples.custom_stages.custom_pipeline import main, DataGeneratorStage
results = main()

# Use individual custom stages
generator = DataGeneratorStage(
    name="my_generator",
    n_samples=500,
    n_features=20
)
```

## Expected Output

The pipeline demonstrates:

1. **Synthetic Data Generation**: Creates configurable datasets
2. **Statistical Analysis**: Comprehensive data exploration
3. **Model Validation**: Cross-validation across multiple algorithms
4. **Feature Insights**: Automatic feature importance analysis
5. **Performance Metrics**: Pipeline efficiency benchmarking
6. **Smart Recommendations**: Optimization suggestions

## Key Concepts Demonstrated

### Custom Stage Creation

```python
class MyCustomStage(Stage):
    """Custom stage example."""
    
    # Configuration parameters (Pydantic fields)
    param1: int = 10
    param2: str = "default"
    
    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate stage inputs."""
        return "required_key" in context
    
    def validate_outputs(self, result: Any) -> bool:
        """Validate stage outputs."""
        return isinstance(result, dict)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Main stage logic."""
        # Your custom implementation
        return {"output": "result"}
```

### Configuration and Validation

- **Type Safety**: Pydantic provides automatic type validation
- **Default Values**: Set sensible defaults for parameters
- **Input Validation**: Check context requirements before execution
- **Output Validation**: Ensure results meet expected format

### Error Handling and Retry

- **Retry Logic**: Automatic retry on transient failures
- **Timeout Protection**: Prevent hanging operations
- **Graceful Degradation**: Continue pipeline on partial failures
- **Informative Errors**: Clear error messages for debugging

## Architecture Benefits

### 1. Reusability
- Custom stages can be used across different pipelines
- Configuration makes stages adaptable to various scenarios
- Validation ensures robust operation

### 2. Maintainability
- Clear separation of concerns
- Type hints improve code clarity
- Validation catches errors early

### 3. Extensibility
- Easy to add new functionality
- Inherit from base classes for consistency
- Plugin-like architecture

### 4. Robustness
- Built-in retry and timeout mechanisms
- Comprehensive error handling
- Validation at multiple levels

## Advanced Features

### Context Management
```python
def execute(self, context: Dict[str, Any]) -> Any:
    # Access outputs from previous stages
    data = context["previous_stage"]["output_key"]
    
    # Use stage configuration
    param_value = self.my_parameter
    
    # Return results for next stages
    return {"my_output": processed_data}
```

### Stage Configuration
```python
# Configure stages with custom parameters
stage = MyCustomStage(
    name="configured_stage",
    param1=100,
    param2="custom_value",
    retry_count=3,
    timeout=60.0
)
```

### Validation Patterns
```python
def validate_inputs(self, context: Dict[str, Any]) -> bool:
    """Common validation patterns."""
    # Check required keys
    if "data" not in context:
        return False
    
    # Validate data types
    data = context["data"]
    if not isinstance(data, np.ndarray):
        return False
    
    # Check data properties
    if data.shape[0] == 0:
        return False
    
    return True
```

## Best Practices

1. **Clear Naming**: Use descriptive stage and parameter names
2. **Documentation**: Add docstrings explaining stage purpose
3. **Validation**: Implement both input and output validation
4. **Error Handling**: Provide meaningful error messages
5. **Configuration**: Make stages configurable for reusability
6. **Testing**: Test custom stages independently
7. **Type Hints**: Use proper type annotations

## Common Patterns

### Data Processing Stages
- Validation of input data format
- Transformation with configurable parameters
- Output in standard format

### Model Training Stages
- Support for different algorithms
- Hyperparameter configuration
- Model serialization capabilities

### Analysis Stages
- Statistical computations
- Visualization generation
- Report creation

## Files

- `custom_pipeline.py`: Complete example with 5 custom stages
- `README.md`: This documentation

## Next Steps

After understanding custom stages:

1. **Create Your Own**: Build stages for your specific domain
2. **Share Components**: Package reusable stages
3. **Advanced Features**: Explore async execution, parallel processing
4. **Integration**: Connect with external services and APIs