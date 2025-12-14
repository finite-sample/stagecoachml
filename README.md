# StagecoachML

[![PyPI - Version](https://img.shields.io/pypi/v/stagecoachml.svg)](https://pypi.org/project/stagecoachml)
[![Tests](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml)
[![Documentation](https://github.com/finite-sample/stagecoachml/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/stagecoachml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StagecoachML** is a tiny library for building two-stage models when your features arrive in two batches at different times.

Think:

- Ad serving and recommendation:  
  first score on **user + context**, then refine on **creative/item + real-time signals**.
- Per-customer privacy:  
  a shared **non-sensitive trunk**, plus a **per-customer head** that uses private fields inside their own environment.
- Latency-sensitive inference:  
  run a fast **stage-1** model early in the request, and only run the heavier **stage-2** model when needed.

StagecoachML encodes that pattern directly in the model interface instead of leaving it buried in infra and notebooks.

---

## When should you use StagecoachML?

Use StagecoachML when:

- You **can’t wait** for all features before you have to start making decisions.
- Some features live in a different **silo** (e.g. customer’s infra) and must never
  hit the central model.
- You want to **tune and evaluate** the whole two-stage system as a *single* estimator
  (train/test/CV), while still being able to:
  - get stage-1 scores from early features, and  
  - get refined scores once late features arrive.

If you have all your features at once and a single model is fine, this library is
probably overkill. But if you live with staggered features, StagecoachML keeps the
logic honest.

---

## Core idea

A StagecoachML model splits features into two groups:

- **Early features**: available at stage 1 (e.g. user, context).
- **Late features**: only available at stage 2 (e.g. ad/creative/item, customer-side data).

You choose:

- a **stage-1 estimator** that sees only early features, and
- a **stage-2 estimator** that sees late features plus (optionally) the stage-1
  prediction, and either:
  - learns to predict the **residual** `y − ŷ₁`, or
  - learns the final target directly.

At inference time you can:

- call `predict_stage1(...)` / `predict_stage1_proba(...)` when you only have
  early features; and
- call `predict(...)` / `predict_proba(...)` later when you have both.

Under the hood, you still train and cross-validate it like any other sklearn estimator.

---

## Installation

StagecoachML is a pure Python package that depends on NumPy, pandas, and scikit-learn.

```bash
pip install stagecoachml
```

Or install from source:

```bash
git clone https://github.com/finite-sample/stagecoachml.git
cd stagecoachml
pip install -e .
```

Import the estimators:

```python
from stagecoachml import StagecoachRegressor, StagecoachClassifier
```

---

## Quick start

### Regression example (diabetes dataset)

```python
from stagecoachml import StagecoachRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data as a DataFrame
diabetes = load_diabetes(as_frame=True)
X = diabetes.frame.drop(columns=["target"])
y = diabetes.frame["target"]

# Split columns into "early" and "late" features
features = list(X.columns)
mid = len(features) // 2
early_features = features[:mid]   # pretend these arrive early
late_features  = features[mid:]   # pretend these arrive later

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Stage-1: fast global model on early features
stage1 = LinearRegression()

# Stage-2: more flexible model on late features + stage-1 prediction
stage2 = RandomForestRegressor(n_estimators=200, random_state=0)

model = StagecoachRegressor(
    stage1_estimator=stage1,
    stage2_estimator=stage2,
    early_features=early_features,
    late_features=late_features,
    residual=True,
    use_stage1_pred_as_feature=True,
    inner_cv=None,            # set >1 to cross-fit stage-1 preds if you care
)

# Hyper-parameter search over both stages
param_grid = {
    "stage1_estimator__fit_intercept": [True, False],
    "stage2_estimator__max_depth": [None, 5, 10],
}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

best = grid.best_estimator_

print("Stage-1 test R²: ", r2_score(y_test, best.predict_stage1(X_test)))
print("Final   test R²: ", r2_score(y_test, best.predict(X_test)))
```

### Classification example (breast cancer dataset)

```python
from stagecoachml import StagecoachClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

features = list(X.columns)
mid = len(features) // 2
early = features[:mid]
late  = features[mid:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

stage1_clf = LogisticRegression(max_iter=1000)
stage2_clf = RandomForestClassifier(n_estimators=200, random_state=2)

model = StagecoachClassifier(
    stage1_estimator=stage1_clf,
    stage2_estimator=stage2_clf,
    early_features=early,
    late_features=late,
    use_stage1_pred_as_feature=True,
)

model.fit(X_train, y_train)

def metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)

# Provisional scores from early features only
stage1_test_proba = model.predict_stage1_proba(X_test)
stage1_acc, stage1_f1 = metrics(y_test, (stage1_test_proba >= 0.5).astype(int))

# Final scores with all features
final_acc, final_f1 = metrics(y_test, model.predict(X_test))

print("Stage-1  test accuracy/F1:", f"{stage1_acc:.3f}/{stage1_f1:.3f}")
print("Final    test accuracy/F1:", f"{final_acc:.3f}/{final_f1:.3f}")
```

---

## API overview

### `StagecoachRegressor`

```python
StagecoachRegressor(
    stage1_estimator,
    stage2_estimator,
    early_features,
    late_features,
    residual=True,
    use_stage1_pred_as_feature=True,
    inner_cv=None,
    random_state=None,
)
```

Key points:

* `stage1_estimator`: any sklearn regressor (`RandomForestRegressor`, `LinearRegression`, etc.).
* `stage2_estimator`: another regressor for the late features (often more flexible).
* `early_features` / `late_features`: column names defining feature arrival.
* `residual=True`: stage 2 learns `y − ŷ₁` and we add it back at prediction time.
* `use_stage1_pred_as_feature=True`: stage-1 prediction becomes an extra input to stage 2.
* `inner_cv`: optional K-fold cross-fitting to generate out-of-fold stage-1 predictions for stage-2 training.

Methods:

* `fit(X, y)`
* `predict_stage1(X)` – early-only predictions.
* `predict(X)` – final predictions.

### `StagecoachClassifier`

```python
StagecoachClassifier(
    stage1_estimator,
    stage2_estimator,
    early_features,
    late_features,
    use_stage1_pred_as_feature=True,
    inner_cv=None,
    random_state=None,
)
```

* Stage-1 classifier must implement `predict_proba` or `decision_function`.
* Stage-2 classifier must implement `predict_proba`.
* `predict_stage1_proba(X)` returns a provisional probability for the positive class
  using early features only.
* `predict_proba(X)` / `predict(X)` use both stages.

---

## Business-level use cases

### 1. Ad serving & recommendation

* **Stage 1 (trunk):**
  user, session, page/context features. Run for every candidate to
  do rough scoring / candidate pruning.
* **Stage 2 (head):**
  ad/creative/item-side features (embeddings, textual features, sponsorship info),
  plus stage-1 scores. Run only on the smaller candidate set.

This lets you:

* keep the expensive features and models off the critical path where possible,
* cross-validate the *whole* two-stage scoring process as one estimator, and
* reason explicitly about which features are actually available at each stage.

### 2. Per-customer models with private fields

* **Shared trunk:** trained on non-sensitive features across all customers.
* **Per-customer head (stage 2):** trained only on that customer’s private fields
  (GDP data, custom risk scores, internal labels) inside their environment.

You can:

* ship the trunk once,
* let each customer fit their own stage-2 model locally,
* still evaluate how “global trunk + local head” behaves on held-out data.

### 3. Latency and staged inference

If your system has a front-door budget (say ~10 ms) and a back-end budget per
selected candidate, StagecoachML gives you a clean way to:

* do rough scoring at T₁ using a small, cheap stage-1 model;
* hydrate more features or call heavier services; and
* refine scores at T₂ with stage-2.

Because the whole pipeline is an sklearn estimator, you don’t have to guess whether
this staging actually helps: you can compare two-stage vs single-stage models on
the same train/test splits.

---

## Examples

The `examples/` directory contains runnable scripts:

* `examples/regression_example.py`
  Uses the diabetes dataset, splits features into early/late, trains a
  `StagecoachRegressor`, and compares it to a one-stage baseline.

* `examples/classification_example.py`
  Uses the breast cancer dataset, trains a `StagecoachClassifier`, and compares
  provisional vs final predictions and a one-stage logistic baseline.

Run them with:

```bash
python -m examples.regression_example
python -m examples.classification_example
```

---

## Design notes & non-goals

* Treat `Stagecoach*` as **one model** for train/validation/test; don’t hand-tune
  stages in isolation and then try to glue them.
* `inner_cv` is an optional extra for robustness, not a replacement for normal
  cross-validation.
* This library is *not* a general DAG/workflow engine. If you want full pipeline
  orchestration (scheduling, retries, monitoring, etc.), you probably want
  Airflow/Prefect/etc. StagecoachML is about one very specific modeling pattern:
  staged feature arrival.

