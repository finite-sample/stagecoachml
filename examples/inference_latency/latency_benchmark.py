"""Temporal candidate-scoring benchmark with explicit representation reuse.

Run with ``python -m examples.inference_latency.latency_benchmark``. All data is
synthetic; timings are local CPU measurements, not production serving estimates.
"""

import argparse
import json
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, ndcg_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from stagecoachml import PredictionEncoder, StagecoachClassifier

UPSTREAM = [f"history_{i}" for i in range(8)]
ONLINE = [f"item_{i}" for i in range(3)] + [f"session_{i}" for i in range(3)]


def make_data(n_users=60, n_days=24, n_candidates=8, seed=42):
    """Simulate user drift, semantic item attributes, and fresh session intent."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_users, 3))
    direction = rng.normal(size=(n_users, 3))
    phase = rng.uniform(0, 2 * np.pi, size=(n_users, 1))
    day = np.arange(n_days)[:, None, None]
    preference = base[None] + np.sin(day / 4 + phase[None]) * direction[None]
    projection = rng.normal(size=(3, 8))
    history = preference @ projection + rng.normal(scale=0.15, size=(n_days, n_users, 8))
    session = rng.normal(scale=0.7, size=(n_days, n_users, 3))
    items = rng.normal(size=(n_days, n_users, n_candidates, 3))
    intent = preference + session
    logits = (items * intent[:, :, None, :]).sum(axis=-1) / 1.5 - 1.0
    probability = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probability).ravel()
    repeated_history = np.repeat(history[:, :, None, :], n_candidates, axis=2)
    repeated_session = np.repeat(session[:, :, None, :], n_candidates, axis=2)
    X = pd.DataFrame(
        np.concatenate([repeated_history, items, repeated_session], axis=-1).reshape(-1, 14),
        columns=UPSTREAM + ONLINE,
    )
    days = np.repeat(np.arange(n_days), n_users * n_candidates)
    return X, y, days, history


def ranker():
    """Use the same interaction-capable head for each comparison."""
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=500),
    )


def quality(y, probability, n_candidates):
    """Evaluate probability quality and ranking within each request."""
    prevalence = np.mean(y)
    entropy = -(prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence))
    bins = np.minimum((probability * 10).astype(int), 9)
    calibration_error = sum(
        np.mean(bins == b) * abs(np.mean(probability[bins == b]) - np.mean(y[bins == b]))
        for b in range(10) if np.any(bins == b)
    )
    return {
        "log_loss": float(log_loss(y, probability)),
        "normalized_entropy": float(log_loss(y, probability) / entropy),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_error_10_bins": float(calibration_error),
        "ndcg": float(ndcg_score(y.reshape(-1, n_candidates), probability.reshape(-1, n_candidates))),
    }


def latency(predict_request, n_requests):
    """Measure complete request calls after warmup, reporting CPU milliseconds."""
    for request in range(min(10, n_requests)):
        predict_request(request)
    elapsed = []
    for request in range(n_requests):
        start = perf_counter()
        predict_request(request)
        elapsed.append((perf_counter() - start) * 1000)
    return dict(zip(["p50_ms", "p95_ms", "p99_ms"], np.percentile(elapsed, [50, 95, 99]).tolist(), strict=True))


def benchmark(n_users=60, n_candidates=8):
    """Compare frozen scalar/vector bridges across representation widths and ages."""
    X, y, days, history = make_data(n_users=n_users, n_candidates=n_candidates)
    pretrain = days < 6
    train = (days >= 6) & (days < 20)
    test = days >= 20
    X_test, y_test = X.loc[test].reset_index(drop=True), y[test]
    online = X_test[ONLINE]
    n_requests = len(X_test) // n_candidates
    results = []
    for name, columns in [("online_only", ONLINE), ("all_features_online", UPSTREAM + ONLINE)]:
        model = ranker().fit(X.loc[train, columns], y[train])
        probability = model.predict_proba(X_test[columns])[:, 1]

        def predict_request(request, model=model, columns=columns):
            start = request * n_candidates
            return model.predict_proba(X_test.iloc[start:start + n_candidates][columns])

        results.append({"model": name, **quality(y_test, probability, n_candidates), **latency(predict_request, n_requests)})

    encoders = [("scalar", PredictionEncoder(LogisticRegression(max_iter=500)).fit(X.loc[pretrain, UPSTREAM], y[pretrain]))]
    history_pretrain = pd.DataFrame(history[:6].reshape(-1, 8), columns=UPSTREAM)
    encoders.extend((f"vector_{width}", PCA(n_components=width).fit(history_pretrain)) for width in [2, 4, 8])
    candidate_ids = np.repeat(np.arange(n_requests), n_candidates)
    request_times = np.repeat(np.repeat(np.arange(20, 24), n_users), n_candidates) * 86400.0
    for name, encoder in encoders:
        model = StagecoachClassifier(
            FrozenEstimator(encoder), ranker(),
            upstream_features=UPSTREAM, online_features=ONLINE,
        ).fit(X.loc[train], y[train])
        for age in [0, 2, 7]:
            snapshots = pd.DataFrame(history[20 - age:24 - age].reshape(-1, 8), columns=UPSTREAM)
            start = perf_counter()
            state = model.encode(
                snapshots, entity_ids=np.arange(n_requests),
                feature_times=np.repeat(np.arange(20 - age, 24 - age), n_users) * 86400.0,
            )
            preparation_ms = (perf_counter() - start) * 1000
            with patch.object(model.encoder_, "transform", side_effect=AssertionError("Online scoring executed the encoder")) as upstream:
                probability = model.predict_proba_from_state(
                    online, state, state_indices=state.rows_for(candidate_ids),
                    as_of=request_times, max_age=age * 86400.0,
                )[:, 1]

                def predict_request(request):
                    start = request * n_candidates
                    stop = start + n_candidates
                    return model.predictor_.predict_proba(
                        online.iloc[start:stop], state,
                        state_indices=state.rows_for(candidate_ids[start:stop]),
                        as_of=request_times[start:stop], max_age=age * 86400.0,
                    )

                timing = latency(predict_request, n_requests)
                assert upstream.call_count == 0
            if age == 0:
                np.testing.assert_allclose(probability, model.predict_proba(X_test)[:, 1])
            results.append({
                "model": name,
                "state_age_days": age,
                "representation_width": state.values.shape[1],
                "representation_values_bytes": state.values.nbytes,
                "state_preparation_ms": preparation_ms,
                "preparation_ms_per_candidate": preparation_ms / len(X_test),
                "candidates_per_state": n_candidates,
                "online_encoder_calls": upstream.call_count,
                **quality(y_test, probability, n_candidates), **timing,
            })
    return {
        "description": "Synthetic temporal experiment; illustrative quality and local CPU latency, no network or feature-store cost.",
        "training": "Encoder: days 0-5; downstream: days 6-19; evaluation: days 20-23. No future observations enter training.",
        "freshness": "Downstream training uses fresh snapshots; age sweeps measure degradation without retraining.",
        "n_users": n_users, "candidates_per_request": n_candidates, "n_test_requests": n_requests,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()
    if args.users < 2 or args.candidates < 2:
        parser.error("--users and --candidates must both be at least 2")
    print(json.dumps(benchmark(args.users, args.candidates), indent=2))
