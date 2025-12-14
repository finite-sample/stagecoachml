# Classification API

```{eval-rst}
.. automodule:: stagecoachml.classification
   :members:
   :undoc-members:
   :show-inheritance:
```

## StagecoachClassifier

```{eval-rst}
.. autoclass:: stagecoachml.classification.StagecoachClassifier
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Usage Examples

### Basic Usage

```python
from stagecoachml import StagecoachClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Split features
features = list(X.columns)
mid = len(features) // 2
early_features = features[:mid]
late_features = features[mid:]

# Create model
model = StagecoachClassifier(
    stage1_estimator=LogisticRegression(max_iter=1000),
    stage2_estimator=RandomForestClassifier(),
    early_features=early_features,
    late_features=late_features,
    use_stage1_pred_as_feature=True,
)

# Train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model.fit(X_train, y_train)

# Get stage-1 probabilities (early features only)
stage1_proba = model.predict_stage1_proba(X_test)

# Get final predictions (all features)
final_pred = model.predict(X_test)
final_proba = model.predict_proba(X_test)
```