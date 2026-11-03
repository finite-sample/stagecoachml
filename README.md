# stagecoachml: reusable representations for online prediction

[![PyPI - Version](https://img.shields.io/pypi/v/stagecoachml.svg)](https://pypi.org/project/stagecoachml)
[![Tests](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/stagecoachml/actions/workflows/ci.yml)
[![Documentation](https://github.com/finite-sample/stagecoachml/actions/workflows/docs.yml/badge.svg)](https://finite-sample.github.io/stagecoachml/)
[![Try in Browser](https://img.shields.io/badge/Try%20in%20Browser-JupyterLite-orange)](https://finite-sample.github.io/stagecoachml/lite/lab/index.html?path=quickstart_interactive.ipynb)

**StagecoachML separates reusable upstream computation from online prediction.**
An encoder prepares a vector from user history, document content, or other reusable
inputs. An online estimator combines that vector with fresh context and candidate
features. The pair remains a scikit-learn estimator for training and evaluation.

```text
Upstream inputs → encoder → EncodedState ───────┐
                                              ├→ online estimator → prediction
Fresh context + candidate features ────────────┘
```

Use it when an expensive representation can be reused across candidates or requests,
when features arrive at different times, or when the stages run in different
processes. An upstream probability is one possible representation; vectors preserve
information that a single score can discard. A user-only score cannot rank different
items for the same user without candidate information downstream.

The upstream encoder can be larger than the online estimator. Its cost is amortized
over the predictions that reuse its output. Benefits depend on actual reuse,
representation quality, freshness, and lookup overhead.

## Installation

```bash
pip install stagecoachml
```

Requires Python 3.12+ and scikit-learn 1.6+. To work from a checkout, use
`pip install -e .`.

## Prepare once, score many candidates

This small example learns interactions between user preferences and item attributes.
It uses an identity encoder so the serving boundary is visible without extra training.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from stagecoachml import StagecoachClassifier

rng = np.random.default_rng(7)
X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["interest", "item", "fresh"])
y = (X.interest * X.item + 0.3 * X.fresh > 0).astype(int)

model = StagecoachClassifier(
    encoder=FunctionTransformer(),
    estimator=make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(),
    ),
    upstream_features=["interest"],
    online_features=["item", "fresh"],
).fit(X, y)

users = pd.DataFrame({"interest": [-1.0, 1.0]})
state = model.encode(users, entity_ids=["alice", "bob"], feature_times=1000.0)

candidates = pd.DataFrame({"item": [-1.0, 1.0, 1.0], "fresh": [0.1, 0.1, 0.2]})
rows = state.rows_for(["alice", "alice", "bob"])
probabilities = model.predict_proba_from_state(
    candidates,
    state,
    state_indices=rows,
    as_of=1010.0,
    max_age=60.0,
)
```

`encode` needs only upstream features. `predict_from_state` and
`predict_proba_from_state` need only online features and prepared state, and never
execute the encoder. `predict(X)` and `predict_proba(X)` run both stages for ordinary
scikit-learn evaluation. Named DataFrame columns are reordered to the fitted schema;
array inputs to the separate stages must already be in the corresponding stage order.
Selectors are explicit lists of names or integer positions. `online_features=[]` is
supported. A feature may be routed to both stages when a raw upstream input should
also be available online.

For a separate serving process, serialize **`model.predictor_`** and the state.
The predictor contains the fitted online estimator and schema, with no reference to
the encoder. Its methods are `predict(X_online, state, ...)` and, for classifiers,
`predict_proba(X_online, state, ...)`. The probability columns follow
`model.classes_` (also available as `model.predictor_.estimator.classes_`).

## Choose a training contract

### Frozen representations

Train an encoder on an earlier or independent dataset, freeze it, then fit the
online estimator against that fixed representation:

```python
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge

from stagecoachml import StagecoachRegressor

encoder = PCA(n_components=4).fit(X_pretrain[upstream_columns])
model = StagecoachRegressor(
    encoder=FrozenEstimator(encoder),
    estimator=Ridge(),
    upstream_features=upstream_columns,
    online_features=online_columns,
).fit(X_train, y_train)
```

The variable names in this snippet represent your own disjoint datasets and column
lists. A complete runnable version is in `examples/classification_example.py`.
Keep the final evaluation set separate from both encoder and downstream training.
Freezing an encoder does not undo leakage from how it was originally trained.

### Prediction stacking and residual regression

`PredictionEncoder` turns a predictor into a transformer. Its default method is
`predict_proba` for classifiers and `predict` for regressors. Explicit
`method="decision_function"` passes raw margins through without claiming calibration.
Binary probabilities retain both class columns.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from stagecoachml import PredictionEncoder, StagecoachRegressor

model = StagecoachRegressor(
    encoder=PredictionEncoder(Ridge()),
    estimator=Ridge(),
    upstream_features=["history"],
    online_features=["current_signal"],
    cv=TimeSeriesSplit(n_splits=4),
    residual=True,
)
```

For chronological rows, each downstream training representation comes from a model
trained on earlier rows. The initial window is excluded from downstream fitting;
`training_indices_` records the rows actually used. Integer `cv` uses standard
K-fold/stratified folds and does **not** imply temporal safety. Group splitters accept
`fit(X, y, groups=...)`. Train/test overlap and repeated test coverage are rejected;
every classification encoder fold must retain the same classes. Splits must reflect
your data dependencies: group separation and chronological separation are different
constraints. Sequence features must also be constructed without future events.

Cross-fitting is restricted to `PredictionEncoder`. Independently fitted latent
encoders may rotate or permute their coordinates, so their vectors cannot generally
be pooled into one downstream training matrix. With `cv=None`, components fit on
the supplied data; supervised encoders can produce optimistic in-sample features.
The deployment encoder is refitted on all supplied training rows, so evaluate the
complete fitted pair on a separate holdout.

`sample_weight` is passed to both trainable components, sliced within folds, and
restricted to the downstream training rows. Components must support that keyword;
unsupported weights raise an error. Frozen encoders ignore fitting arguments.

With `residual=True`, the encoder must return one prediction column. The online
regressor learns `y - prediction` and serving adds that prediction back. This option
is independent of feature routing; the representation is always available online.

## Representation lifecycle

`EncodedState` contains a dense numeric matrix, a fitted encoder identifier,
optional unique entity IDs, and optional feature cutoff timestamps. Timestamps are
Unix seconds. They describe when the input information was current, not when a
cache write happened. Arrays are copied and made read-only on construction.

- `state.rows_for(candidate_entity_ids)` preserves candidate order and allows repeats.
  Integer and string identifiers remain distinct; IDs are not converted to text.
- Missing entities raise an error; there is no implicit recomputation or zero filling.
- `as_of` rejects future feature snapshots; `max_age` additionally rejects stale state.
  Both operate after candidate mapping. Freshness checks require stored timestamps.
- Every successful model fit creates a new identifier. State from an older fit is
  rejected by the new predictor; an older predictor/state pair remains usable.
- Save and deploy matching encoder/predictor versions together. Do not mutate fitted
  components in place. The identifier checks compatibility, not data provenance.

State storage, refresh scheduling, retries, eviction, and fallback models belong to
application infrastructure. The library supplies an explicit boundary and validation.

## Evaluation

Run the examples from a checkout:

```bash
python -m examples.regression_example
python -m examples.classification_example
python -m examples.inference_latency.latency_benchmark
```

The temporal benchmark compares online-only prediction, all features computed online,
a scalar probability bridge, and vector bridges of widths 2, 4, and 8. It evaluates
fresh, two-day-old, and seven-day-old state. Encoders train on days 0–5, downstream
models on days 6–19, and evaluation uses days 20–23.

Its JSON output reports log loss, normalized entropy, Brier score, binned calibration
error, within-request NDCG, p50/p95/p99 request latency, representation bytes, preparation
time, and reuse count. Encoder call counts are asserted to be zero during prepared
serving, and fresh prepared predictions must equal combined predictions. Use
`--users` and `--candidates` to vary the workload. These are synthetic quality results
and local CPU timings; lookup measurements exclude a remote feature store.

## Scope and architectural background

The separation is motivated by systems such as Meta's
[multi-stage ads-ranking architecture](https://engineering.fb.com/2026/08/05/ml-applications/from-user-sequences-to-scaling-laws-a-multi-stage-architecture-for-metas-ads-ranking/):
user representations can be prepared asynchronously and consumed alongside fresh,
candidate-dependent signals. StagecoachML implements the composition and serving
contract. It does not implement LLaTTE, tokenization, attention, joint neural training,
a retrieval cascade, or a feature store. Its benchmarks do not establish scaling laws
or reproduce Meta's production results.

## Migration from the prediction-only API

This is a breaking redesign; there are no compatibility aliases.

| Previous API | Replacement |
|---|---|
| `stage1_estimator=regressor` | `encoder=PredictionEncoder(regressor)` |
| `stage2_estimator=...` | `estimator=...` |
| `early_features`, `late_features` | `upstream_features`, `online_features` |
| `inner_cv` | `cv`, now accepting splitters as well as integers |
| `predict_stage1`, `predict_stage1_proba` | `encode(X_upstream).values` |
| `set_stage1_cache`, `clear_stage1_cache` | Store/discard explicit `EncodedState` objects |
| `use_stage1_pred_as_feature` | Removed; the online estimator always receives the representation |
| Regression's default residual correction | Set `residual=True` explicitly; default is direct prediction |

Grid-search parameter paths now use `encoder__...` and `estimator__...`; a prediction
encoder's wrapped model is addressed as `encoder__estimator__...`.

## Development

```bash
uv sync --group dev --group docs
uv run pytest --cov --cov-report=term-missing --cov-fail-under=86
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pydoclint stagecoachml
uv run sphinx-build -W -b html docs _site
uv build
```

The documentation includes this README directly. Its browser notebook installs the
wheel built from the same checkout as the documentation.
