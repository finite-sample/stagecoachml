# Examples

StagecoachML comes with comprehensive examples demonstrating real-world machine learning workflows. All examples use datasets from scikit-learn for easy reproduction.

## Quick Start Examples

### 🌸 Iris Classification
**File**: `examples/iris_classification/iris_pipeline.py`

A complete classification pipeline demonstrating:
- Multi-class classification (3 species)
- Multiple algorithms (Random Forest, Logistic Regression)
- Model comparison and selection
- Feature importance analysis
- Sample predictions with probabilities

```bash
python examples/iris_classification/iris_pipeline.py
```

**Key Features**:
- Loads famous iris dataset (150 samples, 4 features)
- Trains and compares two different classifiers
- Provides detailed performance metrics
- Shows feature importance for decision trees
- Makes predictions on sample flower measurements

### 🏠 Boston Housing Regression
**File**: `examples/boston_housing/housing_pipeline.py`

A comprehensive regression workflow featuring:
- Multiple regression algorithms
- Feature scaling for linear models
- Advanced performance metrics
- Real estate price predictions

```bash
python examples/boston_housing/housing_pipeline.py
```

**Key Features**:
- Boston housing dataset (506 samples, 13 features)
- Four different regressors (Linear, Ridge, Random Forest, Gradient Boosting)
- Automatic model comparison with RMSE, MAE, and R²
- Feature importance analysis
- Predictions on realistic house characteristics

### 🔢 Handwritten Digits Recognition
**File**: `examples/digits_recognition/digits_pipeline.py`

A computer vision pipeline showcasing:
- Image classification (8×8 pixel digits)
- Multiple classifiers (SVM, Logistic Regression, Random Forest)
- ASCII visualization of digits
- Detailed error analysis

```bash
python examples/digits_recognition/digits_pipeline.py
```

**Key Features**:
- Handwritten digits dataset (1,797 samples, 64 features)
- Image preprocessing and normalization
- Three different classification algorithms
- Confusion matrix and error analysis
- Visual representation of digit patterns

## Advanced Examples

### 🔧 Custom Stages
**File**: `examples/custom_stages/custom_pipeline.py`

Advanced example demonstrating how to extend StagecoachML:
- Custom stage creation
- Input/output validation
- Configuration and retry logic
- Performance benchmarking

```bash
python examples/custom_stages/custom_pipeline.py
```

**Custom Stages Demonstrated**:
- `DataGeneratorStage`: Synthetic dataset generation
- `StatisticalAnalyzerStage`: Advanced data analysis
- `CrossValidatorStage`: Model validation with retry capability
- `ModelExplainerStage`: Feature importance and insights
- `PerformanceBenchmarkStage`: Pipeline performance metrics

## Example Structure

Each example follows a consistent structure:

```
examples/
├── iris_classification/
│   ├── iris_pipeline.py      # Complete runnable script
│   ├── iris_config.yaml      # CLI configuration
│   └── README.md             # Detailed documentation
├── boston_housing/
│   ├── housing_pipeline.py
│   └── README.md
├── digits_recognition/
│   ├── digits_pipeline.py
│   └── README.md
└── custom_stages/
    ├── custom_pipeline.py
    └── README.md
```

## Running Examples

### Method 1: Direct Python Execution
```bash
# From the project root directory
python examples/iris_classification/iris_pipeline.py
python examples/boston_housing/housing_pipeline.py
python examples/digits_recognition/digits_pipeline.py
python examples/custom_stages/custom_pipeline.py
```

### Method 2: Interactive Exploration
```python
import sys
sys.path.insert(0, "src")

# Run any example
from examples.iris_classification.iris_pipeline import main
results = main()

# Access specific results
accuracy = results['evaluate']['best_accuracy']
model = results['evaluate']['best_model']
```

### Method 3: CLI (where available)
```bash
stagecoach run examples/iris_classification/iris_config.yaml
```

## Common Patterns Demonstrated

### 1. **Data Loading and Preprocessing**
- Loading sklearn datasets
- Data normalization and scaling
- Train/test splitting with stratification

### 2. **Model Training and Comparison**
- Multiple algorithm comparison
- Hyperparameter configuration
- Cross-validation techniques

### 3. **Evaluation and Analysis**
- Performance metrics (accuracy, RMSE, R²)
- Confusion matrices
- Feature importance analysis
- Error pattern identification

### 4. **Pipeline Management**
- Stage dependencies and execution order
- Context passing between stages
- Result aggregation and reporting

### 5. **Visualization and Reporting**
- ASCII art for simple visualizations
- Detailed classification reports
- Performance summaries

## Expected Performance

### Iris Classification
- **Best Model**: Usually Random Forest or Logistic Regression
- **Accuracy**: 95-100% (high-quality, separable dataset)
- **Runtime**: <5 seconds

### Boston Housing Regression
- **Best Model**: Typically Random Forest or Gradient Boosting
- **RMSE**: ~$3-4k (good performance for housing prices)
- **R² Score**: 0.85-0.95
- **Runtime**: <10 seconds

### Digits Recognition
- **Best Model**: Usually SVM with RBF kernel
- **Accuracy**: 95-99% (excellent for 8×8 images)
- **Common Errors**: 8↔9, 4↔9, 5↔6 (similar visual patterns)
- **Runtime**: <15 seconds

### Custom Stages
- **Demonstrates**: Advanced StagecoachML features
- **Synthetic Data**: Configurable dataset generation
- **Analysis**: Comprehensive pipeline insights
- **Runtime**: <20 seconds

## Customization Ideas

### Extend Existing Examples
1. **Add More Models**: Include XGBoost, Neural Networks, or ensemble methods
2. **Feature Engineering**: Create polynomial features or interactions
3. **Hyperparameter Tuning**: Add grid search or Bayesian optimization
4. **Cross-Validation**: Implement k-fold validation for robust evaluation
5. **Visualization**: Add matplotlib plots for better insights

### Create New Examples
1. **Time Series**: Stock price or weather prediction
2. **NLP**: Text classification or sentiment analysis (using sklearn's text datasets)
3. **Clustering**: Customer segmentation or image clustering
4. **Dimensionality Reduction**: PCA or t-SNE visualization
5. **Anomaly Detection**: Outlier detection in various domains

## Best Practices Shown

### Code Organization
- Clear function separation for each stage
- Descriptive variable names and comments
- Consistent error handling
- Modular design for reusability

### Pipeline Design
- Logical stage dependencies
- Minimal coupling between stages
- Clear context passing
- Comprehensive result collection

### Machine Learning
- Proper data splitting
- Feature preprocessing
- Model validation techniques
- Performance evaluation
- Result interpretation

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure src/ is in Python path
2. **Missing Dependencies**: Install sklearn, numpy, pandas
3. **Dataset Warnings**: sklearn may show deprecation warnings (normal)
4. **Performance Variation**: Results may vary slightly due to randomness

### Solutions
```python
# Fix import issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Install missing packages
pip install scikit-learn numpy pandas

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
```

## Next Steps

After exploring the examples:

1. **Modify Parameters**: Change model hyperparameters and see the impact
2. **Add Stages**: Implement additional preprocessing or analysis stages
3. **Create Custom Examples**: Apply StagecoachML to your own datasets
4. **Build Production Pipelines**: Scale examples for real-world applications
5. **Contribute**: Share your examples with the community

These examples provide a solid foundation for understanding StagecoachML capabilities and serve as templates for building your own machine learning pipelines.