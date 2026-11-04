"""Classification with a frozen encoder and a separate downstream training set."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stagecoachml import StagecoachClassifier


def main():
    data = load_breast_cancer(as_frame=True)
    upstream = [name for name in data.data.columns if name.startswith("mean ")]
    online = [name for name in data.data.columns if name not in upstream]
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0, stratify=data.target)
    X_encoder, X_head, _, y_head = train_test_split(X_train, y_train, random_state=1, stratify=y_train)
    encoder = make_pipeline(StandardScaler(), PCA(n_components=4)).fit(X_encoder[upstream])
    model = StagecoachClassifier(
        FrozenEstimator(encoder), make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        upstream_features=upstream, online_features=online,
    ).fit(X_head, y_head)
    state = model.encode(X_test[upstream])
    probability = model.predictor_.predict_proba(X_test[online], state)
    np.testing.assert_allclose(probability, model.predict_proba(X_test))
    print(f"Held-out log loss: {log_loss(y_test, probability):.3f}")
    print(f"Representation width: {state.values.shape[1]}")


if __name__ == "__main__":
    main()
