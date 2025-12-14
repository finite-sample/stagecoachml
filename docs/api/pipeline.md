# Regression API

```{eval-rst}
.. automodule:: stagecoachml.regression
   :members:
   :undoc-members:
   :show-inheritance:
```

## StagecoachRegressor

```{eval-rst}
.. autoclass:: stagecoachml.regression.StagecoachRegressor
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Usage Examples

### Basic Usage

```python
from stagecoachml import StagecoachRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
diabetes = load_diabetes(as_frame=True)
X = diabetes.frame.drop(columns=["target"])
y = diabetes.frame["target"]

# Split features
features = list(X.columns)
mid = len(features) // 2
early_features = features[:mid]
late_features = features[mid:]

# Create model
model = StagecoachRegressor(
    stage1_estimator=LinearRegression(),
    stage2_estimator=RandomForestRegressor(),
    early_features=early_features,
    late_features=late_features,
    residual=True,
    use_stage1_pred_as_feature=True,
)

# Train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# Get stage-1 predictions (early features only)
stage1_pred = model.predict_stage1(X_test)

# Get final predictions (all features)
final_pred = model.predict(X_test)
```