"""Two-stage regression example using the diabetes dataset.

This example demonstrates basic StagecoachRegressor usage with feature splitting,
hyperparameter tuning, and performance comparison.
"""

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

print(f"Dataset: {len(X)} samples, {len(features)} features")
print(f"Early features: {early_features}")
print(f"Late features: {late_features}")

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

print("\nResults:")
print("Stage-1 test R²: ", r2_score(y_test, best.predict_stage1(X_test)))
print("Final   test R²: ", r2_score(y_test, best.predict(X_test)))

# Compare with single-stage baseline
baseline = RandomForestRegressor(n_estimators=200, random_state=0)
baseline.fit(X_train, y_train)
print("Baseline test R²:", r2_score(y_test, baseline.predict(X_test)))