"""Inference latency benchmark comparing staged vs single-stage models.

This example demonstrates the latency benefits of two-stage models when:
1. Stage-1 predictions can be precomputed/cached
2. Early decisions need to be made before all features are available
3. Different stages have different computational requirements

Use case: Ad serving where user/context features arrive first, then
item/creative features are retrieved based on initial filtering.
"""

import numpy as np
from profiler import LatencyProfiler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from stagecoachml import StagecoachRegressor


def create_realistic_feature_split():
    """Create a realistic early/late feature split for housing data.

    Early features: Basic location and structural info (available immediately)
    Late features: Derived stats and computed features (require more processing)
    """
    # Load California housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data  # Features are separate from target
    y = housing.target

    # Split into early vs late features based on realistic availability
    early_features = [
        'Latitude',           # GPS coordinate (immediate)
        'Longitude',          # GPS coordinate (immediate)
        'HouseAge',           # Basic property info (immediate)
        'AveRooms',          # Basic room count (immediate)
    ]

    late_features = [
        'AveBedrms',         # Detailed room analysis (requires processing)
        'Population',        # Census/demographic lookup (slower)
        'AveOccup',         # Occupancy analysis (computed)
        'MedInc',           # Income analysis (external API call)
    ]

    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Early features ({len(early_features)}): {early_features}")
    print(f"Late features ({len(late_features)}): {late_features}")
    print()

    return X, y, early_features, late_features


def train_models(X_train, X_test, y_train, y_test, early_features, late_features):
    """Train single-stage baseline and two-stage StagecoachML models."""
    print("Training models...")

    # Single-stage baseline: one model using all features
    single_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    single_model.fit(X_train, y_train)

    # Two-stage model: fast stage1 + more complex stage2
    stage1_estimator = LinearRegression()  # Fast linear model for early filtering
    stage2_estimator = RandomForestRegressor(  # More complex model for refinement
        n_estimators=50,
        max_depth=8,
        random_state=42
    )

    stagecoach_model = StagecoachRegressor(
        stage1_estimator=stage1_estimator,
        stage2_estimator=stage2_estimator,
        early_features=early_features,
        late_features=late_features,
        residual=True,
        use_stage1_pred_as_feature=True,
        inner_cv=5,  # Cross-fit to avoid overfitting
        random_state=42
    )
    stagecoach_model.fit(X_train, y_train)

    # Evaluate accuracy
    single_pred = single_model.predict(X_test)
    stage1_pred = stagecoach_model.predict_stage1(X_test)
    stagecoach_pred = stagecoach_model.predict(X_test)

    print("Model Performance:")
    print(f"Single-stage R²:     {r2_score(y_test, single_pred):.4f}")
    print(f"Stage-1 only R²:     {r2_score(y_test, stage1_pred):.4f}")
    print(f"Two-stage R²:        {r2_score(y_test, stagecoach_pred):.4f}")
    print()

    return single_model, stagecoach_model


def benchmark_individual_predictions(single_model, stagecoach_model, X_test, n_samples=100):
    """Benchmark individual prediction latency - the key use case."""
    print(f"Benchmarking individual predictions ({n_samples} samples)...")

    # Select random samples for timing
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    test_samples = X_test.iloc[indices]

    profiler = LatencyProfiler()

    # Time individual predictions
    for i in range(n_samples):
        sample = test_samples.iloc[i:i+1]

        # Single-stage model (all features required)
        with profiler.profile('single_stage'):
            _ = single_model.predict(sample)

        # Stage-1 only (early features only - for initial filtering)
        with profiler.profile('stage1_only'):
            _ = stagecoach_model.predict_stage1(sample)

        # Two-stage full prediction (both stages)
        with profiler.profile('two_stage_fresh'):
            _ = stagecoach_model.predict(sample)

    # Now test with cached stage-1 predictions
    print("Testing with precomputed stage-1 cache...")

    # Precompute stage-1 for all test samples (simulation: batch processing)
    X_early, _ = stagecoach_model._split_features(test_samples)
    stage1_predictions = stagecoach_model.predict_stage1(test_samples)
    stagecoach_model.set_stage1_cache(X_early, stage1_predictions)

    # Time predictions with cached stage-1
    for i in range(n_samples):
        sample = test_samples.iloc[i:i+1]

        # Two-stage with cached stage-1
        with profiler.profile('two_stage_cached'):
            _ = stagecoach_model.predict(sample)

    # Clear cache
    stagecoach_model.clear_stage1_cache()

    return profiler


def benchmark_batch_predictions(single_model, stagecoach_model, X_test, batch_sizes=None):
    """Benchmark batch prediction performance."""
    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000]
    print("Benchmarking batch predictions...")

    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue

        # Select batch
        batch = X_test.iloc[:batch_size]

        profiler = LatencyProfiler()

        # Time batch predictions (10 runs each)
        for _ in range(10):
            with profiler.profile('single_stage'):
                _ = single_model.predict(batch)

            with profiler.profile('two_stage'):
                _ = stagecoach_model.predict(batch)

        results[batch_size] = {
            'single_stage': profiler.get_stats('single_stage'),
            'two_stage': profiler.get_stats('two_stage')
        }

    return results


def print_latency_results(profiler):
    """Print formatted latency comparison results."""
    print("=== Individual Prediction Latency Results ===")

    scenarios = [
        ('single_stage', 'Single-stage (all features)'),
        ('stage1_only', 'Stage-1 only (early features)'),
        ('two_stage_fresh', 'Two-stage (fresh computation)'),
        ('two_stage_cached', 'Two-stage (cached stage-1)')
    ]

    print(f"{'Scenario':<35} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std (ms)':<12}")
    print("-" * 71)

    for key, description in scenarios:
        stats = profiler.get_stats(key)
        if stats:
            print(f"{description:<35} {stats['mean_ms']:<12.2f} {stats['median_ms']:<12.2f} {stats['std_ms']:<12.2f}")

    print()

    # Calculate speedup ratios
    single_mean = profiler.get_stats('single_stage')['mean_ms']
    stage1_mean = profiler.get_stats('stage1_only')['mean_ms']
    cached_mean = profiler.get_stats('two_stage_cached')['mean_ms']

    print("=== Speedup Analysis ===")
    print(f"Stage-1 vs Single-stage:      {single_mean/stage1_mean:.1f}x faster")
    print(f"Cached vs Single-stage:       {single_mean/cached_mean:.1f}x faster")
    print(f"Cached vs Fresh two-stage:    {profiler.get_stats('two_stage_fresh')['mean_ms']/cached_mean:.1f}x faster")
    print()


def print_batch_results(batch_results):
    """Print batch prediction results."""
    print("=== Batch Prediction Results ===")
    print(f"{'Batch Size':<12} {'Single (ms)':<12} {'Two-stage (ms)':<15} {'Speedup':<10}")
    print("-" * 51)

    for batch_size, results in batch_results.items():
        single_ms = results['single_stage']['mean_ms']
        two_stage_ms = results['two_stage']['mean_ms']
        speedup = single_ms / two_stage_ms

        print(f"{batch_size:<12} {single_ms:<12.2f} {two_stage_ms:<15.2f} {speedup:<10.2f}x")

    print()


def demonstrate_use_cases():
    """Demonstrate practical use cases for two-stage models."""
    print("=== Practical Use Cases ===")
    print()

    print("1. Real-time Ad Serving:")
    print("   • Stage-1: Score user+context features (1-2ms budget)")
    print("   • Filter to top 100 candidates")
    print("   • Stage-2: Score with ad creative features (5-10ms budget)")
    print("   • Benefit: Only run expensive features on promising candidates")
    print()

    print("2. Progressive Model Serving:")
    print("   • Stage-1: Fast initial prediction when early features arrive")
    print("   • User sees provisional result immediately")
    print("   • Stage-2: Refine prediction when late features become available")
    print("   • Benefit: Better user experience with progressive enhancement")
    print()

    print("3. Distributed Privacy-Preserving ML:")
    print("   • Stage-1: Global model on non-sensitive features")
    print("   • Stage-2: Local refinement on sensitive features (customer side)")
    print("   • Benefit: Keep sensitive data local while leveraging global patterns")
    print()


def main():
    """Run the complete inference latency benchmark."""
    print("StagecoachML Inference Latency Benchmark")
    print("=" * 50)
    print()

    # Load and prepare data
    X, y, early_features, late_features = create_realistic_feature_split()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    single_model, stagecoach_model = train_models(
        X_train, X_test, y_train, y_test, early_features, late_features
    )

    # Benchmark individual predictions (key use case)
    profiler = benchmark_individual_predictions(single_model, stagecoach_model, X_test)
    print_latency_results(profiler)

    # Benchmark batch predictions
    batch_results = benchmark_batch_predictions(single_model, stagecoach_model, X_test)
    print_batch_results(batch_results)

    # Show practical applications
    demonstrate_use_cases()

    print("=== Key Takeaways ===")
    print("• Stage-1 predictions are much faster for early filtering/decisions")
    print("• Caching stage-1 predictions provides significant speedup")
    print("• Two-stage models enable progressive enhancement of predictions")
    print("• Best suited for scenarios with staggered feature availability")


if __name__ == "__main__":
    main()
