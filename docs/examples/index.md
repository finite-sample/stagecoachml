# Examples

The `examples/` directory in the repository holds runnable scripts. They use
scikit-learn's bundled datasets, so they need no data download and reproduce
exactly.

## Regression

**File:** `examples/regression_example.py`

Loads the diabetes dataset, splits its ten columns into an early and a late
half, and fits a `StagecoachRegressor` with a `LinearRegression` trunk and a
`RandomForestRegressor` head. It then runs a `GridSearchCV` over parameters of
*both* stages at once — the point being that the two-stage model is a single
estimator as far as scikit-learn is concerned — and reports stage-1 and final
R² on the held-out split alongside a one-stage baseline.

```bash
python -m examples.regression_example
```

## Classification

**File:** `examples/classification_example.py`

Loads the breast cancer dataset and fits a `StagecoachClassifier` with a
logistic trunk and a random-forest head. It compares the provisional
probabilities available from early features alone
(`predict_stage1_proba`) against the final two-stage probabilities, and
against a one-stage logistic baseline, using accuracy and F1.

```bash
python -m examples.classification_example
```

## Inference latency

**Directory:** `examples/inference_latency/`

A benchmark of the case the library exists for: a request budget that cannot
wait for every feature. Using the California housing dataset with a
location-first feature split, it times three arrangements — a single-stage
model that waits for everything, a two-stage model that scores early and
refines late, and a two-stage model whose stage-1 predictions are cached via
`set_stage1_cache`.

`profiler.py` in that directory is a small standalone timing and
memory-tracking helper, kept out of the installed package because its
dependency on `psutil` does not work in the browser (Pyodide/JupyterLite)
environment the interactive docs run in.

```bash
pip install -r examples/inference_latency/requirements.txt
python examples/inference_latency/latency_benchmark.py
```

## Try it without installing anything

The [interactive notebook](../index.md) runs the quickstart in your browser
through JupyterLite — no local Python required.
