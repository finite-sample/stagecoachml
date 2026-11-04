"""Residual regression with explicit preparation and online serving."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from stagecoachml import PredictionEncoder, StagecoachRegressor


def main():
    data = load_diabetes(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
    upstream = ["age", "sex", "bmi", "bp"]
    online = [name for name in data.data.columns if name not in upstream]
    model = StagecoachRegressor(
        PredictionEncoder(Ridge()), Ridge(), upstream_features=upstream,
        online_features=online, residual=True, cv=5,
    )
    search = GridSearchCV(model, {"encoder__estimator__alpha": [0.1, 1.0], "estimator__alpha": [0.1, 1.0]}, cv=3).fit(X_train, y_train)
    model = search.best_estimator_
    state = model.encode(X_test[upstream])
    predicted = model.predictor_.predict(X_test[online], state)
    np.testing.assert_allclose(predicted, model.predict(X_test))
    print(f"Upstream R2: {r2_score(y_test, state.values[:, 0]):.3f}")
    print(f"Final R2: {r2_score(y_test, predicted):.3f}")


if __name__ == "__main__":
    main()
