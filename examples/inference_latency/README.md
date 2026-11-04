# Temporal candidate-scoring benchmark

From an installed source checkout:

```bash
python -m examples.inference_latency.latency_benchmark > benchmark.json
```

The benchmark uses NumPy, pandas, and scikit-learn. It needs no extra dependencies or
external data. `--users 120 --candidates 16` varies the workload and reuse count.

User preferences drift over 24 simulated days. Eight historical measurements encode
three preference dimensions. Candidate attributes and fresh session intent are online
inputs; labels are sampled from their interaction with current preferences.

- Encoders train on days 0–5; downstream models train on days 6–19.
- Evaluation covers days 20–23, using fresh, two-day-old, or seven-day-old snapshots.
- A scalar classifier bridge and PCA bridges of widths 2, 4, and 8 use frozen encoders.
- Online-only and all-features-online models provide comparisons with the same head family.

Every prepared serving path asserts zero encoder calls. Fresh prepared predictions
must match combined predictions. Candidate entity lookup, input validation, and the
online model are timed together after warmup. Preparation time and representation
values' byte size are reported separately; bytes exclude identifiers, timestamps,
serialization overhead, and model parameters.

Quality metrics are log loss, normalized entropy (log loss divided by the held-out
label entropy), Brier score, ten-bin calibration error, and NDCG over all candidates
within each request. Requests with no positive labels contribute zero NDCG.

Timings are local CPU measurements, exclude network/remote-store costs, and have no
fixed speedup assertion. The upstream encoders here are deliberately small: this
example establishes correct reuse and measures its overhead, rather than simulating
expensive inference with sleeps. The generated data favors informative user vectors;
results are illustrative and are not a claim about Meta's scaling laws.
