# Handwritten Digits Recognition

This example demonstrates a computer vision pipeline using StagecoachML to recognize handwritten digits (0-9).

## What This Example Does

1. **Data Loading**: Loads the handwritten digits dataset from scikit-learn (8×8 pixel images)
2. **Data Visualization**: Shows sample digits with ASCII art representation
3. **Data Preprocessing**: Normalizes pixel values and splits into train/test sets
4. **Feature Scaling**: Standardizes features for linear models
5. **Model Training**: Trains three different classifiers:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest Classifier
6. **Model Evaluation**: Compares performance with accuracy and detailed metrics
7. **Individual Testing**: Tests predictions on specific digit examples with visualization

## Dataset

The digits dataset contains 1,797 samples of handwritten digits:

**Characteristics:**
- **Images**: 8×8 pixel grayscale images
- **Features**: 64 pixel values (flattened from 8×8)
- **Classes**: 10 digits (0-9)
- **Pixel values**: 0-16 (grayscale intensity)
- **Balance**: ~180 samples per digit class

## Running the Example

### Method 1: Python Script

```bash
# From the project root directory
python examples/digits_recognition/digits_pipeline.py
```

### Method 2: Interactive exploration

```python
import sys
sys.path.insert(0, "src")

from examples.digits_recognition.digits_pipeline import main
results = main()

# Access specific results
best_model = results['evaluate']['best_model']
accuracy = results['evaluate']['best_accuracy']
confusion_matrix = results['evaluate']['confusion_matrix']
```

## Expected Output

The pipeline provides comprehensive output including:

1. **Dataset Overview**: Number of images, features, and class distribution
2. **Sample Visualization**: ASCII art representation of digit examples
3. **Preprocessing Details**: Normalization and train/test split information
4. **Training Progress**: Confirmation of each model training
5. **Model Comparison**: Accuracy scores for all three models
6. **Best Model Selection**: Highest performing algorithm
7. **Detailed Classification Report**: Precision, recall, F1-score per digit
8. **Confusion Matrix**: Visual breakdown of classification errors
9. **Individual Predictions**: Examples with correct/incorrect predictions
10. **Error Analysis**: Most common classification mistakes

## Model Performance

Typical performance on this dataset:

- **SVM**: Usually achieves highest accuracy (~98-99%)
- **Logistic Regression**: Strong performance (~95-97%)
- **Random Forest**: Good performance (~95-97%)

The SVM with RBF kernel typically performs best due to the non-linear nature of pixel patterns.

## Key Features Demonstrated

- **Computer Vision**: Working with image data (pixel arrays)
- **Multi-class Classification**: 10-class problem
- **Model Comparison**: Different algorithm types (linear, kernel, tree-based)
- **Feature Preprocessing**: Normalization and standardization
- **Visualization**: ASCII art representation of images
- **Error Analysis**: Understanding model mistakes

## Pipeline Visualization

```
Pipeline: digits_recognition
========================================
Stage: load_data
  Leads to: visualize

Stage: visualize
  Dependencies: load_data
  Leads to: preprocess

Stage: preprocess
  Dependencies: visualize
  Leads to: scale, train_rf

Stage: scale
  Dependencies: preprocess
  Leads to: train_lr, train_svm

Stage: train_lr
  Dependencies: scale
  Leads to: evaluate

Stage: train_svm
  Dependencies: scale
  Leads to: evaluate

Stage: train_rf
  Dependencies: preprocess
  Leads to: evaluate

Stage: evaluate
  Dependencies: train_lr, train_svm, train_rf
  Leads to: test_individual

Stage: test_individual
  Dependencies: evaluate
```

## Common Confusion Patterns

The model typically confuses digits that look similar:
- **8 ↔ 9**: Similar curves and loops
- **4 ↔ 9**: Similar vertical strokes
- **5 ↔ 6**: Similar upper portions
- **1 ↔ 7**: Similar vertical orientation

## Customization Ideas

1. **Add More Models**: Include Neural Networks, KNN, or Naive Bayes
2. **Feature Engineering**: Add edge detection or geometric features
3. **Data Augmentation**: Rotate, shift, or add noise to training images
4. **Hyperparameter Tuning**: Optimize model parameters with grid search
5. **Ensemble Methods**: Combine predictions from multiple models
6. **Visualization**: Add matplotlib plots for better digit visualization
7. **Real-time Prediction**: Add functionality to draw and predict digits

## Performance Optimization

- **SVM**: Tune C and gamma parameters for better performance
- **Random Forest**: Adjust n_estimators and max_depth
- **Feature Selection**: Use PCA for dimensionality reduction
- **Cross-Validation**: Add k-fold validation for robust evaluation

## Files

- `digits_pipeline.py`: Complete runnable Python script
- `README.md`: This documentation

## Technical Notes

- Pixel values are normalized to [0,1] range for better model performance
- StandardScaler is applied for linear models (LR, SVM) but not Random Forest
- ASCII visualization provides a quick way to see digit patterns without matplotlib
- Stratified split ensures balanced representation of all digits in train/test sets