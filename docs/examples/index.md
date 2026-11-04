# Examples

All examples use bundled or synthetic data and need no dataset downloads.

| Command | Demonstrates |
|---|---|
| `python -m examples.regression_example` | Cross-fitted prediction encoding, residual learning, and grid search |
| `python -m examples.classification_example` | Frozen PCA encoding with separate encoder, downstream, and test data |
| `python -m examples.inference_latency.latency_benchmark` | Temporal ranking evaluation, candidate reuse, representation width, and staleness |

The regression and classification examples assert agreement between combined and
prepared inference. The temporal benchmark additionally fails if prepared serving
executes the encoder. It reports measurements as JSON; redirect stdout to save a run.

The benchmark fits every encoder before the downstream training period and evaluates
on later days. It compares all models on the same labels and requests. Its synthetic
history contains semantic preference information by construction, so the experiment
demonstrates the representation tradeoff without claiming that a vector always wins
on real data. Downstream training uses fresh snapshots; the age sweep measures the
resulting distribution shift without retraining.

<a href="../lite/lab/index.html?path=quickstart_interactive.ipynb">Open the interactive notebook</a>
to exercise the same serving contract in a browser. The documentation build bundles
its own package wheel, keeping the notebook and package API synchronized.
