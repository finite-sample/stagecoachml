# Boston Housing Price Prediction

This example demonstrates a comprehensive regression pipeline using StagecoachML to predict housing prices in Boston.

## What This Example Does

1. **Data Loading**: Loads the Boston housing dataset from scikit-learn
2. **Data Exploration**: Analyzes dataset characteristics and feature statistics
3. **Data Splitting**: Creates training and testing sets
4. **Feature Scaling**: Standardizes features for linear models
5. **Model Training**: Trains four different regression models:
   - Linear Regression
   - Ridge Regression  
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. **Model Evaluation**: Compares performance using multiple metrics (RMSE, MAE, R²)
7. **Prediction**: Makes predictions on sample house characteristics

## Dataset

The Boston housing dataset contains 506 samples with 13 features:

**Key Features:**
- `CRIM`: Crime rate per capita
- `RM`: Average number of rooms per dwelling
- `DIS`: Distances to employment centers
- `LSTAT`: % lower status of the population
- `TAX`: Property tax rate
- `PTRATIO`: Pupil-teacher ratio

**Target:** Median home value in $1000s

## Running the Example

### Method 1: Python Script

```bash
# From the project root directory
python examples/boston_housing/housing_pipeline.py
```

### Method 2: Explore individual components

```python
import sys
sys.path.insert(0, "src")

from examples.boston_housing.housing_pipeline import main
results = main()

# Access results
best_model = results['evaluate']['best_model']
predictions = results['predict']['sample_predictions']
```

## Expected Output

The pipeline provides comprehensive output including:

1. **Dataset Overview**: Size, target range, and average prices
2. **Data Exploration**: Feature statistics and characteristics
3. **Training Progress**: Model training confirmations
4. **Feature Importance**: Most predictive features (for tree-based models)
5. **Model Comparison**: Performance metrics for all models:
   - **RMSE** (Root Mean Square Error): Lower is better
   - **MAE** (Mean Absolute Error): Average prediction error
   - **R²** (R-squared): Proportion of variance explained (higher is better)
6. **Best Model Selection**: Automatically selects highest performing model
7. **Sample Predictions**: Price predictions for different house types

## Model Comparison

Typically, the models perform as follows:

- **Linear Regression**: Good baseline, interpretable
- **Ridge Regression**: Handles multicollinearity better
- **Random Forest**: Usually best performance, handles non-linearity
- **Gradient Boosting**: Often competitive with Random Forest

## Key Features Demonstrated

- **Multiple Model Training**: Parallel training of different algorithms
- **Feature Scaling**: Proper preprocessing for different model types
- **Comprehensive Evaluation**: Multiple regression metrics
- **Feature Importance**: Understanding model decisions
- **Real-world Predictions**: Practical use of trained models

## Pipeline Visualization

```
Pipeline: boston_housing_regression
========================================
Stage: load_data
  Leads to: explore

Stage: explore  
  Dependencies: load_data
  Leads to: split

Stage: split
  Dependencies: explore
  Leads to: scale, train_rf, train_gb

Stage: scale
  Dependencies: split
  Leads to: train_lr, train_ridge

Stage: train_lr
  Dependencies: scale
  Leads to: evaluate

Stage: train_ridge
  Dependencies: scale  
  Leads to: evaluate

Stage: train_rf
  Dependencies: split
  Leads to: evaluate

Stage: train_gb
  Dependencies: split
  Leads to: evaluate

Stage: evaluate
  Dependencies: train_lr, train_ridge, train_rf, train_gb
  Leads to: predict

Stage: predict
  Dependencies: evaluate
```

## Customization Ideas

1. **Add More Models**: Include SVR, XGBoost, or neural networks
2. **Feature Engineering**: Create polynomial features or interactions
3. **Cross-Validation**: Add k-fold validation for robust evaluation
4. **Hyperparameter Tuning**: Include grid search or Bayesian optimization
5. **Ensemble Methods**: Combine multiple models for better performance
6. **Visualization**: Add plots for predictions vs. actual values

## Performance Notes

- Random Forest typically performs best on this dataset
- Linear models benefit significantly from feature scaling
- Gradient Boosting is often close second to Random Forest
- RMSE values around $3-4k are considered good for this dataset

## Files

- `housing_pipeline.py`: Complete runnable Python script
- `README.md`: This documentation