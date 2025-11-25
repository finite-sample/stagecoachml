"""Custom stages example showing how to extend StagecoachML.

This example demonstrates:
1. Creating custom stage classes
2. Implementing data validation
3. Building reusable components
4. Using custom stages in pipelines
5. Advanced stage features (retry, timeout, config)

Run with: python examples/custom_stages/custom_pipeline.py
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from stagecoachml import Pipeline
from stagecoachml.stage import Stage, FunctionStage


# Custom Stage 1: Data Generator
class DataGeneratorStage(Stage):
    """Custom stage that generates synthetic datasets."""
    
    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    random_state: int = 42
    
    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate that we have the right configuration."""
        if self.n_samples < 100:
            print(f"⚠️  Warning: Small sample size ({self.n_samples})")
        return True
    
    def validate_outputs(self, result: Any) -> bool:
        """Validate the generated data."""
        if not isinstance(result, dict):
            return False
        if "X" not in result or "y" not in result:
            return False
        
        X, y = result["X"], result["y"]
        if X.shape[0] != self.n_samples:
            return False
        if X.shape[1] != self.n_features:
            return False
        if len(y) != self.n_samples:
            return False
        
        return True
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Generate synthetic classification dataset."""
        print(f"🎲 Generating synthetic data:")
        print(f"   Samples: {self.n_samples}")
        print(f"   Features: {self.n_features}")
        print(f"   Classes: {self.n_classes}")
        
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=max(2, self.n_features // 2),
            n_redundant=max(0, self.n_features // 4),
            n_classes=self.n_classes,
            random_state=self.random_state
        )
        
        print(f"✅ Generated {X.shape[0]} samples with {X.shape[1]} features")
        
        return {
            "X": X,
            "y": y,
            "feature_names": [f"feature_{i}" for i in range(self.n_features)],
            "n_samples": self.n_samples,
            "n_features": self.n_features
        }


# Custom Stage 2: Statistical Analyzer
class StatisticalAnalyzerStage(Stage):
    """Custom stage for detailed statistical analysis."""
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Perform statistical analysis on the data."""
        print("📊 Performing statistical analysis...")
        
        # Get data from previous stage
        gen_data = context.get("generate_data", {})
        X = gen_data.get("X")
        y = gen_data.get("y")
        
        if X is None or y is None:
            raise ValueError("No data found in context. Need 'X' and 'y' keys.")
        
        # Calculate statistics
        stats = {
            "feature_means": X.mean(axis=0),
            "feature_stds": X.std(axis=0),
            "feature_mins": X.min(axis=0),
            "feature_maxs": X.max(axis=0),
            "class_distribution": np.bincount(y),
            "class_balance": np.bincount(y) / len(y),
            "correlation_matrix": np.corrcoef(X.T),
            "total_samples": len(y),
            "n_features": X.shape[1]
        }
        
        # Print summary
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Features: {stats['n_features']}")
        print(f"   Class distribution: {dict(enumerate(stats['class_distribution']))}")
        print(f"   Class balance: {[f'{p:.3f}' for p in stats['class_balance']]}")
        
        # Feature analysis
        print("   Feature statistics (first 5):")
        for i in range(min(5, len(stats['feature_means']))):
            mean = stats['feature_means'][i]
            std = stats['feature_stds'][i]
            print(f"     Feature {i}: μ={mean:.3f}, σ={std:.3f}")
        
        return stats


# Custom Stage 3: Cross Validator with Retry Logic
class CrossValidatorStage(Stage):
    """Custom stage that performs cross-validation with retry capability."""
    
    cv_folds: int = 5
    retry_count: int = 2
    timeout: float = 30.0
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Perform cross-validation on multiple models."""
        print(f"🔄 Performing {self.cv_folds}-fold cross-validation...")
        
        gen_data = context.get("generate_data", {})
        X = gen_data.get("X")
        y = gen_data.get("y")
        
        if X is None or y is None:
            raise ValueError("Missing data for cross-validation")
        
        # Test multiple models
        models = {
            "Random Forest (10)": RandomForestClassifier(n_estimators=10, random_state=42),
            "Random Forest (50)": RandomForestClassifier(n_estimators=50, random_state=42),
            "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Testing {name}...")
            
            try:
                # Simulate potential failure (removed for actual demo)
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
                
                results[name] = {
                    "scores": scores,
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "model": model
                }
                
                print(f"     Mean CV score: {scores.mean():.4f} (±{scores.std():.4f})")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                results[name] = {"error": str(e)}
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]["mean_score"])
            best_score = valid_results[best_model_name]["mean_score"]
            print(f"🏆 Best model: {best_model_name} (CV score: {best_score:.4f})")
        else:
            best_model_name = None
            print("❌ No models succeeded")
        
        return {
            "cv_results": results,
            "best_model_name": best_model_name,
            "cv_folds": self.cv_folds
        }


# Custom Stage 4: Model Explainer
class ModelExplainerStage(Stage):
    """Custom stage that provides model explanations and insights."""
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Generate model explanations and feature importance."""
        print("🔍 Generating model explanations...")
        
        # Get best model from cross-validation
        cv_data = context.get("cross_validate", {})
        cv_results = cv_data.get("cv_results", {})
        best_model_name = cv_data.get("best_model_name")
        
        if not best_model_name or best_model_name not in cv_results:
            return {"explanation": "No valid model to explain"}
        
        # Get the best model and retrain on full data
        best_model = cv_results[best_model_name]["model"]
        gen_data = context.get("generate_data", {})
        X = gen_data.get("X")
        y = gen_data.get("y")
        feature_names = gen_data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])
        
        # Retrain on full dataset
        best_model.fit(X, y)
        
        # Get feature importance (if available)
        explanations = {}
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("   Feature Importance:")
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"     {i+1}. {feature}: {importance:.4f}")
            
            explanations["feature_importance"] = feature_importance
            explanations["top_features"] = sorted_features[:10]
        
        # Model-specific insights
        if hasattr(best_model, 'n_estimators'):
            explanations["n_trees"] = best_model.n_estimators
            print(f"   Model uses {best_model.n_estimators} decision trees")
        
        # Performance insights
        train_score = best_model.score(X, y)
        explanations["train_accuracy"] = train_score
        print(f"   Training accuracy: {train_score:.4f}")
        
        return {
            "model_explanation": explanations,
            "trained_model": best_model,
            "model_name": best_model_name
        }


# Custom Stage 5: Performance Benchmarker
class PerformanceBenchmarkStage(Stage):
    """Custom stage that benchmarks pipeline performance."""
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Benchmark various aspects of the pipeline."""
        print("⏱️  Benchmarking pipeline performance...")
        
        # Get data size info
        gen_data = context.get("generate_data", {})
        n_samples = gen_data.get("n_samples", 0)
        n_features = gen_data.get("n_features", 0)
        
        # Simulate timing analysis
        start_time = time.time()
        
        # Mock some performance metrics
        metrics = {
            "data_size": f"{n_samples} samples × {n_features} features",
            "memory_usage": f"~{(n_samples * n_features * 8) / 1024:.1f} KB",
            "pipeline_stages": len([k for k in context.keys() if not k.startswith("_")]),
            "total_features": n_features
        }
        
        # Calculate processing time per sample
        if n_samples > 0:
            processing_time = (time.time() - start_time) * 1000  # ms
            time_per_sample = processing_time / n_samples
            metrics["processing_speed"] = f"{time_per_sample:.3f} ms/sample"
        
        print(f"   Data processed: {metrics['data_size']}")
        print(f"   Estimated memory: {metrics['memory_usage']}")
        print(f"   Pipeline stages: {metrics['pipeline_stages']}")
        
        # Performance recommendations
        recommendations = []
        if n_samples < 500:
            recommendations.append("Consider using more training data for better model performance")
        if n_features > 50:
            recommendations.append("Consider feature selection or dimensionality reduction")
        if metrics["pipeline_stages"] > 8:
            recommendations.append("Pipeline has many stages - consider optimization")
        
        return {
            "performance_metrics": metrics,
            "recommendations": recommendations,
            "benchmark_timestamp": time.time()
        }


def main():
    """Run the custom stages pipeline demonstration."""
    print("🔧 Custom Stages Pipeline with StagecoachML")
    print("=" * 50)

    # Create pipeline
    pipeline = Pipeline(name="custom_stages_demo")

    # Stage 1: Generate data using custom stage
    data_generator = DataGeneratorStage(
        name="generate_data",
        description="Generate synthetic classification dataset",
        n_samples=800,
        n_features=15,
        n_classes=3,
        random_state=42
    )
    pipeline.add_stage(data_generator)

    # Stage 2: Analyze data using custom stage
    stats_analyzer = StatisticalAnalyzerStage(
        name="analyze_stats",
        description="Perform statistical analysis"
    )
    pipeline.add_stage(stats_analyzer)

    # Stage 3: Cross-validation using custom stage with retry
    cross_validator = CrossValidatorStage(
        name="cross_validate",
        description="Cross-validate multiple models",
        cv_folds=5,
        retry_count=2,
        timeout=30.0
    )
    pipeline.add_stage(cross_validator)

    # Stage 4: Model explanation using custom stage
    explainer = ModelExplainerStage(
        name="explain_model",
        description="Generate model explanations and insights"
    )
    pipeline.add_stage(explainer)

    # Stage 5: Performance benchmarking
    benchmarker = PerformanceBenchmarkStage(
        name="benchmark",
        description="Benchmark pipeline performance"
    )
    pipeline.add_stage(benchmarker)

    # Stage 6: Final summary using function stage
    def create_summary(context):
        """Create a final summary of all results."""
        print("\n📋 Pipeline Summary:")
        print("-" * 30)
        
        # Data info
        gen_data = context.get("generate_data", {})
        n_samples = gen_data.get("n_samples", 0)
        n_features = gen_data.get("n_features", 0)
        print(f"Dataset: {n_samples} samples, {n_features} features")
        
        # Best model info
        cv_data = context.get("cross_validate", {})
        best_model = cv_data.get("best_model_name", "Unknown")
        cv_results = cv_data.get("cv_results", {})
        if best_model and best_model in cv_results:
            cv_score = cv_results[best_model]["mean_score"]
            print(f"Best Model: {best_model} (CV: {cv_score:.4f})")
        
        # Top features
        explain_data = context.get("explain_model", {})
        explanation = explain_data.get("model_explanation", {})
        if "top_features" in explanation:
            top_3 = explanation["top_features"][:3]
            print("Top Features:")
            for i, (feature, importance) in enumerate(top_3):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Recommendations
        bench_data = context.get("benchmark", {})
        recommendations = bench_data.get("recommendations", [])
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        return {
            "summary_complete": True,
            "pipeline_successful": True
        }

    summary_stage = FunctionStage(
        name="summarize",
        func=create_summary,
        description="Create pipeline summary"
    )
    pipeline.add_stage(summary_stage)

    # Define dependencies
    pipeline.add_dependency("generate_data", "analyze_stats")
    pipeline.add_dependency("generate_data", "cross_validate")
    pipeline.add_dependency("cross_validate", "explain_model")
    pipeline.add_dependency("generate_data", "benchmark")
    pipeline.add_dependency("explain_model", "summarize")
    pipeline.add_dependency("benchmark", "summarize")

    # Show pipeline structure
    print("\n🔄 Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Run the pipeline
    print("🚀 Running custom stages pipeline...")
    results = pipeline.run()

    # Final summary
    print("\n" + "="*50)
    print("✅ Custom Stages Pipeline completed!")
    
    # Show what custom stages accomplished
    print("\n🔧 Custom Stages Demonstrated:")
    print("  ✓ DataGeneratorStage - Synthetic data generation")
    print("  ✓ StatisticalAnalyzerStage - Advanced data analysis")
    print("  ✓ CrossValidatorStage - Model validation with retry logic")
    print("  ✓ ModelExplainerStage - Feature importance and insights")
    print("  ✓ PerformanceBenchmarkStage - Pipeline performance metrics")
    
    print("\n📊 Key Features Showcased:")
    print("  • Custom stage inheritance from base Stage class")
    print("  • Input and output validation")
    print("  • Configuration parameters")
    print("  • Retry and timeout capabilities")
    print("  • Rich contextual information flow")
    print("  • Error handling and robustness")
    
    print("="*50)

    return results


if __name__ == "__main__":
    results = main()