# Inference Latency Benchmark

This example demonstrates the latency benefits of StagecoachML's two-stage approach, particularly when stage-1 predictions can be precomputed or cached.

## Use Case: Real-time Ad Serving

Imagine an ad serving system where:

1. **Stage 1** (1-2ms budget): User + context features arrive first
   - Geographic location, device type, time of day
   - User demographic info, browsing history
   - Quick filtering to promising ad candidates

2. **Stage 2** (5-10ms budget): Ad creative features retrieved for top candidates
   - Ad embeddings, campaign metadata  
   - Real-time bid info, creative performance history
   - Final ranking and selection

## The Benchmark

We use the California housing dataset with a realistic feature split:

- **Early features** (immediately available): Location, basic property info
- **Late features** (require processing): Demographics, computed statistics

We compare three approaches:

1. **Single-stage model**: Waits for all features, then predicts
2. **Two-stage model**: Early prediction → refinement with late features  
3. **Cached two-stage**: Pre-computed stage-1 predictions for repeated inference

## Installation

Install the profiling dependencies:

```bash
cd examples/inference_latency
pip install -r requirements.txt
```

## Running the Benchmark

```bash
python latency_benchmark.py
```

## Profiling Utilities

This directory also contains `profiler.py` - a complete performance profiling utility that was moved from the core StagecoachML package to maintain browser compatibility. You can copy this profiler to your own projects if you need performance measurement capabilities.

## Expected Results

The benchmark shows:

- **Stage-1 only**: ~2-5x faster than single-stage (fewer features)
- **Cached two-stage**: ~3-8x faster than fresh computation
- **Progressive enhancement**: Initial prediction available immediately

## Key Insights

1. **Early filtering**: Use stage-1 for rapid candidate scoring
2. **Caching benefits**: Precompute stage-1 for repeated queries
3. **Accuracy trade-offs**: Stage-1 provides good initial estimates
4. **Memory vs latency**: Cache size vs. lookup speed considerations

## Real-world Applications

- **Ad serving**: User features → creative features
- **Recommendation**: User profile → item features  
- **Risk scoring**: Basic info → detailed financial data
- **Fraud detection**: Transaction basics → network analysis

The key insight: when features arrive at different times or have different computational costs, two-stage models can provide both better user experience (progressive results) and better system efficiency (targeted computation).