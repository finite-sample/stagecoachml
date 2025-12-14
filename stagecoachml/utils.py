"""Utility functions for StagecoachML."""

import logging
import time
from typing import Any

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class LatencyProfiler:
    """Utility class for measuring inference latency and memory usage."""

    def __init__(self):
        self.times: dict[str, list] = {}
        self.memory_usage: dict[str, list] = {}

    def profile(self, name: str):
        """Context manager for timing code blocks."""
        return _TimingContext(self, name)

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(duration)

    def add_memory(self, name: str, memory_mb: float):
        """Add a memory usage measurement."""
        if name not in self.memory_usage:
            self.memory_usage[name] = []
        self.memory_usage[name].append(memory_mb)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get timing statistics for a named operation."""
        if name not in self.times:
            return {}

        times = np.array(self.times[name])
        return {
            "mean_ms": np.mean(times) * 1000,
            "median_ms": np.median(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "count": len(times),
        }

    def get_memory_stats(self, name: str) -> dict[str, float]:
        """Get memory statistics for a named operation."""
        if name not in self.memory_usage:
            return {}

        memory = np.array(self.memory_usage[name])
        return {
            "mean_mb": np.mean(memory),
            "median_mb": np.median(memory),
            "std_mb": np.std(memory),
            "min_mb": np.min(memory),
            "max_mb": np.max(memory),
            "count": len(memory),
        }

    def print_summary(self):
        """Print a summary of all measurements."""
        logger.info("=== Latency Profile Summary ===")
        for name in self.times:
            stats = self.get_stats(name)
            logger.info("%s:", name)
            logger.info("  Mean: %.2fms", stats['mean_ms'])
            logger.info("  Median: %.2fms", stats['median_ms'])
            logger.info("  Std: %.2fms", stats['std_ms'])
            logger.info("  Range: %.2fms - %.2fms", stats['min_ms'], stats['max_ms'])
            logger.info("  Count: %d", stats['count'])
            logger.info("")


class _TimingContext:
    """Context manager for timing operations."""

    def __init__(self, profiler: LatencyProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            self.start_memory = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.profiler.add_timing(self.name, duration)

        if self.start_memory is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - self.start_memory
                self.profiler.add_memory(self.name, memory_delta)
            except Exception:
                pass


def benchmark_predictions(
    models: dict[str, Any],
    X_test: np.ndarray,
    n_runs: int = 100,
    individual_predictions: bool = True,
) -> dict[str, dict[str, float]]:
    """Benchmark prediction latency for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of {name: fitted_model} pairs to benchmark
    X_test : array-like
        Test data for predictions
    n_runs : int, default=100
        Number of timing runs
    individual_predictions : bool, default=True
        If True, time individual predictions. If False, time batch predictions.

    Returns
    -------
    results : dict
        Dictionary of timing results for each model

    """
    results = {}
    profiler = LatencyProfiler()

    for model_name, model in models.items():
        if individual_predictions:
            # Time individual predictions
            for i in range(min(n_runs, len(X_test))):
                with profiler.profile(f"{model_name}_individual"):
                    _ = model.predict(X_test[i : i + 1])
        else:
            # Time batch predictions
            for _ in range(n_runs):
                with profiler.profile(f"{model_name}_batch"):
                    _ = model.predict(X_test)

        timing_key = f"{model_name}_{'individual' if individual_predictions else 'batch'}"
        results[model_name] = profiler.get_stats(timing_key)

    return results


def compare_stage_performance(
    stagecoach_model,
    single_stage_model,
    X_test: np.ndarray,
    cache_stage1: bool = False,
    n_runs: int = 100,
) -> dict[str, Any]:
    """Compare performance between staged and single-stage models.

    Parameters
    ----------
    stagecoach_model : StagecoachRegressor or StagecoachClassifier
        Fitted two-stage model
    single_stage_model : sklearn estimator
        Fitted single-stage baseline model
    X_test : array-like
        Test data
    cache_stage1 : bool, default=False
        Whether to cache stage1 predictions for stagecoach model
    n_runs : int, default=100
        Number of timing runs

    Returns
    -------
    results : dict
        Comprehensive comparison results

    """
    profiler = LatencyProfiler()

    # Optionally cache stage1 predictions
    if cache_stage1:
        X_early, _ = stagecoach_model._split_features(X_test)
        stage1_pred = stagecoach_model.predict_stage1(X_test)
        stagecoach_model.set_stage1_cache(X_early, stage1_pred)

    # Benchmark different prediction modes
    models = {
        "single_stage": single_stage_model,
        "stagecoach_stage1_only": lambda X: stagecoach_model.predict_stage1(X),
        "stagecoach_full": stagecoach_model,
    }

    # Time predictions
    for i in range(min(n_runs, len(X_test))):
        sample = X_test[i : i + 1]

        with profiler.profile("single_stage"):
            _ = single_stage_model.predict(sample)

        with profiler.profile("stagecoach_stage1_only"):
            _ = stagecoach_model.predict_stage1(sample)

        with profiler.profile("stagecoach_full"):
            _ = stagecoach_model.predict(sample)

    return {
        "single_stage": profiler.get_stats("single_stage"),
        "stagecoach_stage1_only": profiler.get_stats("stagecoach_stage1_only"),
        "stagecoach_full": profiler.get_stats("stagecoach_full"),
    }
