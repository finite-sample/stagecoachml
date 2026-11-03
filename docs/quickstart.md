# Quick start

The [README on the home page](index.md) contains the installation instructions,
training contracts, and a complete candidate-scoring example.

## Independent deployment

After fitting a model, prepare state from upstream columns and export the online
predictor. The online artifact contains no upstream encoder:

```python
import joblib

state = model.encode(users, entity_ids=user_ids, feature_times=cutoff_times)
joblib.dump(model.predictor_, "predictor.joblib")
joblib.dump(state, "state.joblib")
```

In the serving process:

```python
predictor = joblib.load("predictor.joblib")
state = joblib.load("state.joblib")
rows = state.rows_for(candidate_user_ids)
probability = predictor.predict_proba(
    candidate_features,
    state,
    state_indices=rows,
    as_of=request_time,
    max_age=3600,
)
```

The variables here are your own fitted model, input tables, identifiers, and Unix
second timestamps. Load serialized artifacts only from trusted sources. Use the
candidate's actual user identifier to construct the mapping; a valid positional
mapping alone cannot establish that the state belongs to the intended user.

## Choosing feature schemas

Use explicit DataFrame column names for readability. Arrays use integer selectors
at fit time. `encode` then expects only upstream columns; downstream serving expects
only online columns. Arrays must keep each stage's fitted column order. DataFrames
are selected and reordered by name. All encoded values and online features must be
finite numeric matrices. Upstream transformers may process categorical DataFrames;
raw variable-length event sequences need an adapter before entering this interface.

The online design matrix is `[representation columns, online columns]`. It is passed
to the downstream estimator as an array. Put numeric preprocessing and interaction
modeling inside that estimator's pipeline as needed.

## Training without leakage

Use frozen encoders trained on independent or earlier data, or a prediction encoder
with an appropriate cross-fitting splitter. Ordinary cross-validation of the whole
model still supplies the outer evaluation boundary. A pretrained encoder used in
outer CV must itself have been trained independently of the outer holdouts.

For temporal evaluation, sort rows chronologically and construct historical features
at the prediction timestamp. Request/user boundaries may require a custom splitter:
`TimeSeriesSplit` splits row positions, so it can split a multi-candidate request.
Freshness checks validate supplied timestamps; they cannot detect future information
hidden inside your feature construction.

See the [examples](examples/index.md) for runnable training and evaluation workflows.
