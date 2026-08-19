# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

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

## 0.2.0

### Added

- `StagecoachRegressor` and `StagecoachClassifier` for two-stage models over
  early and late feature groups, with optional residual learning, stage-1
  prediction as a stage-2 feature, cross-fitted stage-1 predictions, and a
  stage-1 prediction cache for latency-sensitive inference.

[Unreleased]: https://github.com/finite-sample/stagecoachml/commits/main
