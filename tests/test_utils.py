"""Tests for utility functions."""

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from stagecoachml import StagecoachRegressor
from stagecoachml.utils import (
    LatencyProfiler,
    benchmark_predictions,
    compare_stage_performance,
)


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


@pytest.fixture
def fitted_models(sample_regression_data):
    """Create fitted models for testing."""
    X_df, y = sample_regression_data

    # Single stage model
    single_model = RandomForestRegressor(n_estimators=10, random_state=42)
    single_model.fit(X_df, y)

    # Two-stage model
    stage1 = LinearRegression()
    stage2 = RandomForestRegressor(n_estimators=5, random_state=42)

    early_features = ["feature_0", "feature_1", "feature_2"]
    late_features = ["feature_3", "feature_4", "feature_5", "feature_6", "feature_7"]

    stagecoach_model = StagecoachRegressor(
        stage1, stage2, early_features=early_features, late_features=late_features
    )
    stagecoach_model.fit(X_df, y)

    return single_model, stagecoach_model, X_df


class TestLatencyProfiler:
    """Tests for LatencyProfiler class."""

    def test_init(self):
        """Test LatencyProfiler initialization."""
        profiler = LatencyProfiler()

        assert hasattr(profiler, "times")
        assert isinstance(profiler.times, dict)
        assert len(profiler.times) == 0
        assert hasattr(profiler, "memory_usage")
        assert isinstance(profiler.memory_usage, dict)
        assert len(profiler.memory_usage) == 0

    def test_profile_context_manager(self):
        """Test profiler as context manager."""
        profiler = LatencyProfiler()

        with profiler.profile("test_operation"):
            # Simulate some work
            sum(range(1000))

        assert "test_operation" in profiler.times
        assert len(profiler.times["test_operation"]) == 1
        assert profiler.times["test_operation"][0] > 0

    def test_multiple_profiles_same_name(self):
        """Test multiple profiles with the same name."""
        profiler = LatencyProfiler()

        with profiler.profile("operation"):
            sum(range(100))

        with profiler.profile("operation"):
            sum(range(200))

        assert "operation" in profiler.times
        assert len(profiler.times["operation"]) == 2
        assert all(t > 0 for t in profiler.times["operation"])

    def test_get_summary(self):
        """Test getting timing summary."""
        profiler = LatencyProfiler()

        # Add some timings
        with profiler.profile("fast_op"):
            pass  # Very fast

        with profiler.profile("fast_op"):
            pass

        with profiler.profile("slow_op"):
            sum(range(10000))  # Slower

        fast_stats = profiler.get_stats("fast_op")
        slow_stats = profiler.get_stats("slow_op")

        assert isinstance(fast_stats, dict)
        assert isinstance(slow_stats, dict)

        # Check structure of stats
        assert "count" in fast_stats
        assert "mean_ms" in fast_stats
        assert "std_ms" in fast_stats
        assert "min_ms" in fast_stats
        assert "max_ms" in fast_stats

        assert fast_stats["count"] == 2
        assert fast_stats["mean_ms"] >= 0
        assert fast_stats["std_ms"] >= 0

    def test_clear(self):
        """Test clearing profiler data."""
        profiler = LatencyProfiler()

        with profiler.profile("test"):
            pass

        assert len(profiler.times) == 1

        # Manual clearing since there's no clear method
        profiler.times.clear()
        profiler.memory_usage.clear()
        assert len(profiler.times) == 0

    def test_empty_summary(self):
        """Test summary with no timings."""
        profiler = LatencyProfiler()
        stats = profiler.get_stats("nonexistent")

        assert isinstance(stats, dict)
        assert len(stats) == 0


class TestBenchmarkPredictions:
    """Tests for benchmark_predictions function."""

    def test_benchmark_single_model(self, fitted_models):
        """Test benchmarking a single model."""
        single_model, _, X_df = fitted_models

        models = {"single_model": single_model}
        results = benchmark_predictions(models, X_df.values[:10], n_runs=3)

        assert isinstance(results, dict)
        assert "single_model" in results
        assert "count" in results["single_model"]
        assert results["single_model"]["count"] == 3

    def test_benchmark_multiple_models(self, fitted_models):
        """Test benchmarking multiple models."""
        single_model, stagecoach_model, X_df = fitted_models

        models = {"single": single_model, "stagecoach": stagecoach_model}
        results = benchmark_predictions(models, X_df.values[:5], n_runs=2)

        assert "single" in results
        assert "stagecoach" in results
        assert results["single"]["count"] == 2
        assert results["stagecoach"]["count"] == 2

    def test_benchmark_batch_predictions(self, fitted_models):
        """Test benchmarking batch predictions."""
        single_model, _, X_df = fitted_models

        models = {"model": single_model}
        results = benchmark_predictions(
            models, X_df.values[:10], n_runs=3, individual_predictions=False
        )

        assert "model" in results
        assert results["model"]["count"] == 3

    def test_benchmark_custom_parameters(self, fitted_models):
        """Test with custom parameters."""
        single_model, _, X_df = fitted_models

        models = {"test_model": single_model}
        results = benchmark_predictions(
            models, X_df.values[:15], n_runs=5, individual_predictions=True
        )

        assert results["test_model"]["count"] == 5


class TestCompareStagePerformance:
    """Tests for compare_stage_performance function."""

    def test_basic_comparison(self, fitted_models):
        """Test basic performance comparison."""
        single_model, stagecoach_model, X_df = fitted_models

        results = compare_stage_performance(
            stagecoach_model, single_model, X_df.values[:20], cache_stage1=False, n_runs=5
        )

        assert isinstance(results, dict)

        # Check expected keys from actual API
        expected_keys = ["single_stage", "stagecoach_stage1_only", "stagecoach_full"]
        for key in expected_keys:
            assert key in results
            assert "count" in results[key]
            assert "mean_ms" in results[key]
            assert results[key]["count"] == 5

    def test_comparison_with_cache(self, fitted_models):
        """Test comparison with stage1 caching enabled."""
        single_model, stagecoach_model, X_df = fitted_models

        results = compare_stage_performance(
            stagecoach_model, single_model, X_df.values[:10], cache_stage1=True, n_runs=3
        )

        # Should still work with caching enabled
        assert isinstance(results, dict)
        for key in ["single_stage", "stagecoach_stage1_only", "stagecoach_full"]:
            assert key in results
            assert results[key]["count"] == 3

    def test_comparison_different_sample_sizes(self, fitted_models):
        """Test comparison with different sample sizes."""
        single_model, stagecoach_model, X_df = fitted_models

        # Test with small sample size
        results_small = compare_stage_performance(
            stagecoach_model, single_model, X_df.values[:5], n_runs=2
        )

        # Test with larger sample size
        results_large = compare_stage_performance(
            stagecoach_model, single_model, X_df.values[:10], n_runs=3
        )

        # Both should complete successfully
        assert len(results_small) == 3
        assert len(results_large) == 3
