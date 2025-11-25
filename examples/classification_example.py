"""Two-stage classification example using the breast cancer dataset.

This example demonstrates basic StagecoachClassifier usage with probability
estimation and performance comparison.
"""

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

print(f"Dataset: {len(X)} samples, {len(features)} features")
print(f"Early features: {len(early)} features")
print(f"Late features: {len(late)} features")

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

print("\nResults:")
print("Stage-1  test accuracy/F1:", f"{stage1_acc:.3f}/{stage1_f1:.3f}")
print("Final    test accuracy/F1:", f"{final_acc:.3f}/{final_f1:.3f}")

# Compare with single-stage baseline
baseline = RandomForestClassifier(n_estimators=200, random_state=2)
baseline.fit(X_train, y_train)
baseline_acc, baseline_f1 = metrics(y_test, baseline.predict(X_test))
print("Baseline test accuracy/F1:", f"{baseline_acc:.3f}/{baseline_f1:.3f}")