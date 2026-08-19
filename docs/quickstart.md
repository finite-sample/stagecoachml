# Quick Start Guide

This guide gets you running with StagecoachML in a few minutes.

## Installation

```bash
# Using pip
pip install stagecoachml

# Using uv (recommended)
uv pip install stagecoachml
```

## The idea in one paragraph

StagecoachML is for the case where your features do not all arrive at once.
You nominate some columns as **early** (available immediately) and the rest as
**late**. A stage-1 estimator is fitted on the early columns alone; a stage-2
estimator is fitted on the late columns plus, optionally, the stage-1
prediction. The pair behaves as a single scikit-learn estimator, so you can
`fit`, cross-validate, and grid-search it the way you would any other — while
still asking for an early-only score when that is all you can afford.

## Your first two-stage model

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from stagecoachml import StagecoachRegressor

diabetes = load_diabetes(as_frame=True)
X = diabetes.frame.drop(columns=["target"])
y = diabetes.frame["target"]

# Pretend the first half of the columns arrives before the second half.
features = list(X.columns)
mid = len(features) // 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = StagecoachRegressor(
    stage1_estimator=LinearRegression(),
    stage2_estimator=RandomForestRegressor(n_estimators=200, random_state=0),
    early_features=features[:mid],
    late_features=features[mid:],
    residual=True,
    use_stage1_pred_as_feature=True,
)
model.fit(X_train, y_train)

print("Stage-1 R2:", r2_score(y_test, model.predict_stage1(X_test)))
print("Final   R2:", r2_score(y_test, model.predict(X_test)))
```

`predict_stage1` uses only the early columns, so it is what you would call at
the point in a request where the late features do not exist yet. `predict`
uses both stages.

## Understanding the output

With `residual=True` and `use_stage1_pred_as_feature=True`, stage 2 is fitted
on `y - ŷ₁` rather than on `y`, and `predict` returns `ŷ₁ + ŷ₂`. With
`residual=False`, stage 2 predicts the target directly and `predict` returns
`ŷ₂` alone. Everything else is unchanged, which makes the two settings
directly comparable on the same split.

## Classification

`StagecoachClassifier` follows the same shape. Its stage-1 estimator must
implement `predict_proba` or `decision_function`, and its stage-2 estimator
must implement `predict_proba`:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from stagecoachml import StagecoachClassifier

data = load_breast_cancer(as_frame=True)
features = list(data.data.columns)
mid = len(features) // 2

model = StagecoachClassifier(
    stage1_estimator=LogisticRegression(max_iter=1000),
    stage2_estimator=RandomForestClassifier(n_estimators=200, random_state=0),
    early_features=features[:mid],
    late_features=features[mid:],
)
model.fit(data.data, data.target)

provisional = model.predict_stage1_proba(data.data)  # early features only
final = model.predict_proba(data.data)  # both stages
```

## Guarding against leakage

If the stage-1 prediction is used as a stage-2 feature, fitting stage 2 on
in-sample stage-1 predictions lets stage 2 learn from stage 1's overfitting.
Set `inner_cv` to a fold count to have the stage-1 feature generated
out-of-fold instead:

```python
model = StagecoachRegressor(
    stage1_estimator=LinearRegression(),
    stage2_estimator=RandomForestRegressor(random_state=0),
    inner_cv=5,
)
```

## Next steps

- Work through the [examples](examples/index.md)
- Read the API reference for [`StagecoachRegressor`](api/regression.md) and
  [`StagecoachClassifier`](api/classification.md)
