# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

Entries for 0.1.0 and 0.1.2, and the description of 0.2.0, were reconstructed
from the diffs between the released commits; there was no changelog while those
versions shipped.

## [Unreleased]

### Added

- Explicit `EncodedState` with entity mappings, fitted encoder identifiers, and
  optional feature cutoff times and freshness checks.
- `PredictionEncoder`, vector transformer support, frozen encoder training, and an
  encoder-free `OnlinePredictor` artifact at `model.predictor_`.
- Independent `encode`, `predict_from_state`, and `predict_proba_from_state` APIs.
- Grouped and forward-only prediction cross-fitting, with consistent sample weights
  and explicit exclusion of rows never held out by a splitter.
- A synthetic temporal candidate-scoring benchmark covering representation width,
  staleness, quality, preparation cost, and request latency.

### Maintenance

- Updated audited development/documentation dependency pins and aligned citation
  version metadata with the published release. CI now includes Python 3.11.
- Browser documentation bundles the checkout wheel for its executable notebook.

### Breaking changes

- Estimators now take `encoder`, `estimator`, `upstream_features`, `online_features`,
  and optional `cv`. Automatic half-column splitting is removed.
- The stage-1 prediction methods and implicit batch cache are replaced by explicit
  prepared state. Regression now defaults to direct prediction; residual learning
  must be enabled explicitly.
- Requires scikit-learn 1.6 or newer for frozen encoder support.


### Changed

- Adopted the py-canon fleet standard: reusable CI, docs and release
  workflows, canon's ruff/pyright/pydoclint configuration, and a shared Sphinx
  configuration.
- Estimator docstrings converted from NumPy to Google style, and public
  methods given type hints.
- `predict`/`fit` on both estimators now assemble the stage-2 design matrix
  through one shared helper instead of two copies of the same branching.

### Fixed

- The quickstart and examples pages documented a `Pipeline`/`Stage` API that
  the package does not have; both now describe the real estimators.
- `check_consistent_length_features` no longer takes an unused `X` argument.

### Removed

- `[project.optional-dependencies]`; development and documentation
  dependencies now live in `[dependency-groups]`.

## [0.2.0] - 2025-12-14

### Removed

- **Breaking.** `stagecoachml.utils`, including `LatencyProfiler`, is no longer
  part of the installed package; it moved to
  `examples/inference_latency/profiler.py` in the repository. `import
  stagecoachml.utils` fails on 0.2.0.
- `psutil` as a runtime dependency — only the profiler used it.

### Added

- An interactive quickstart notebook that runs in the browser through
  JupyterLite/Pyodide, so the documentation is executable without a Binder
  round-trip.

### Changed

- Documentation build moved from `nbsphinx` + `myst-parser` to `myst-nb` +
  `jupyterlite-sphinx`.
- `uv.lock` is now committed.

## [0.1.2] - 2025-12-14

### Fixed

- `__version__` is read from the installed package metadata in all cases. The
  previous fallback hardcoded `"0.1.0"`, so a source checkout reported the
  wrong version.

### Changed

- `LatencyProfiler.print_summary()` emits through `logging` rather than
  `print`, so it no longer writes to stdout unbidden.
- Dropped the `viz` extra (matplotlib, seaborn, plotly, graphviz), which
  nothing in the package used, along with the unused `pytest-timeout` and
  `pytest-benchmark` development dependencies.
- Added `deptry` configuration.

No 0.1.1 was published; the version went 0.1.0 to 0.1.2.

## [0.1.0] - 2025-11-29

### Added

- `StagecoachRegressor` and `StagecoachClassifier` for two-stage models over
  early and late feature groups, with optional residual learning, stage-1
  prediction as a stage-2 feature, cross-fitted stage-1 predictions, and a
  stage-1 prediction cache for latency-sensitive inference.
- `stagecoachml.utils.LatencyProfiler` for measuring inference latency and
  memory use.
- Regression and classification examples, and a latency benchmark.

[Unreleased]: https://github.com/finite-sample/stagecoachml/compare/v0.2.0...main
[0.2.0]: https://github.com/finite-sample/stagecoachml/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/finite-sample/stagecoachml/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/finite-sample/stagecoachml/releases/tag/v0.1.0
