# Performance Profiling

The performance profiling functionality has been moved to the examples directory to avoid dependencies that are incompatible with browser environments (Pyodide/JupyterLite).

## Profiling Example

See the complete profiling implementation in:
- **`examples/inference_latency/profiler.py`** - Full profiling utilities
- **`examples/inference_latency/latency_benchmark.py`** - Benchmarking example

## Using the Profiler

If you need performance profiling in your own project, you can copy the profiler from the examples:

```python
# Copy profiler.py from examples/inference_latency/ to your project
from profiler import LatencyProfiler
import time

profiler = LatencyProfiler()

# Profile some operations
with profiler.profile("fast_operation"):
    time.sleep(0.01)

with profiler.profile("slow_operation"):
    time.sleep(0.1)

# Get statistics
stats = profiler.get_stats("fast_operation")
print(f"Mean time: {stats['mean_ms']:.2f}ms")

# Print summary
profiler.print_summary()
```

## Requirements

The profiler requires additional dependencies:
```bash
pip install psutil>=5.8.0  # For memory tracking
```

## Complete Example

For a complete benchmarking example, see `examples/inference_latency/latency_benchmark.py` which demonstrates:
- Comparing single-stage vs two-stage model performance
- Memory usage tracking
- Realistic feature arrival scenarios
- Performance analysis and reporting