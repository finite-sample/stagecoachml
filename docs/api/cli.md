# Utilities API

```{eval-rst}
.. automodule:: stagecoachml.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## LatencyProfiler

```{eval-rst}
.. autoclass:: stagecoachml.utils.LatencyProfiler
   :members:
   :special-members: __init__
   :show-inheritance:
```

## Utility Functions

```{eval-rst}
.. autofunction:: stagecoachml.utils.benchmark_predictions
```

```{eval-rst}
.. autofunction:: stagecoachml.utils.compare_stage_performance
```

## Usage Examples

### Profiling Latency

```python
from stagecoachml.utils import LatencyProfiler
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

### Benchmarking Model Performance

```python
from stagecoachml.utils import benchmark_predictions
from stagecoachml import StagecoachRegressor
from sklearn.datasets import load_diabetes

# Set up data and model
diabetes = load_diabetes(as_frame=True)
X = diabetes.frame.drop(columns=["target"])
# ... (model setup)

# Benchmark predictions
results = benchmark_predictions(
    models={"stagecoach": model},
    X=X.values[:100],
    n_runs=10
)

for model_name, stats in results.items():
    print(f"{model_name}: {stats['mean_ms']:.2f}ms avg")
```