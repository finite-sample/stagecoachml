# Iris Classification Example

This example demonstrates a complete machine learning pipeline using StagecoachML to classify iris species.

## What This Example Does

1. **Data Loading**: Loads the famous iris dataset from scikit-learn
2. **Data Splitting**: Splits data into training and testing sets with stratification
3. **Feature Scaling**: Standardizes features using StandardScaler
4. **Model Training**: Trains two different classifiers (Random Forest and Logistic Regression)
5. **Model Evaluation**: Compares model performance and selects the best one
6. **Prediction**: Makes predictions on sample data points

## Dataset

The iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length
- Sepal width  
- Petal length
- Petal width

The goal is to classify flowers into 3 species:
- Setosa
- Versicolor
- Virginica

## Running the Example

### Method 1: Python Script

```bash
# From the project root directory
python examples/iris_classification/iris_pipeline.py
```

### Method 2: StagecoachML CLI

```bash
# From the project root directory
stagecoach run examples/iris_classification/iris_config.yaml
```

## Expected Output

The pipeline will output:

1. **Dataset Information**: Size, features, classes
2. **Train/Test Split**: Sample counts for each set
3. **Feature Scaling**: Confirmation of standardization
4. **Model Training**: Training progress for each algorithm
5. **Feature Importance**: Most important features (for Random Forest)
6. **Model Comparison**: Accuracy scores for both models
7. **Best Model Selection**: Winner with detailed metrics
8. **Classification Report**: Precision, recall, F1-score for each class
9. **Confusion Matrix**: Prediction accuracy breakdown
10. **Sample Predictions**: Predictions on new data points

## Key Features Demonstrated

- **Pipeline Orchestration**: Sequential and parallel stage execution
- **Multiple Models**: Training and comparing different algorithms
- **Feature Engineering**: Data preprocessing and scaling
- **Model Evaluation**: Comprehensive performance assessment
- **Real Predictions**: Making predictions on new samples

## Pipeline Visualization

```
Pipeline: iris_classification
========================================
Stage: load_data
  Leads to: split

Stage: split
  Dependencies: load_data
  Leads to: scale

Stage: scale
  Dependencies: split
  Leads to: train_rf, train_lr

Stage: train_rf
  Dependencies: scale
  Leads to: evaluate

Stage: train_lr
  Dependencies: scale
  Leads to: evaluate

Stage: evaluate
  Dependencies: train_rf, train_lr
  Leads to: predict

Stage: predict
  Dependencies: evaluate
```

## Customization

You can modify the pipeline by:

1. **Adding new models**: Include SVM, KNN, or other classifiers
2. **Changing parameters**: Adjust model hyperparameters
3. **Adding feature selection**: Include feature importance filtering
4. **Cross-validation**: Add k-fold validation stages
5. **Hyperparameter tuning**: Include grid search optimization

## Files

- `iris_pipeline.py`: Complete runnable Python script
- `iris_config.yaml`: Pipeline configuration for CLI usage
- `README.md`: This documentation