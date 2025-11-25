"""Integration tests for end-to-end pipeline execution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from stagecoachml import Pipeline
from stagecoachml.stage import (
    DataLoaderStage,
    FunctionStage,
)


class TestEndToEndPipeline:
    """Test complete pipeline workflows."""

    def test_ml_pipeline_workflow(self, sample_csv_file):
        """Test a complete ML pipeline from data loading to prediction."""
        pipeline = Pipeline(name="ml_pipeline")

        # Stage 1: Load data
        load_stage = DataLoaderStage(
            name="load_data",
            source_type="csv",
            source_path=str(sample_csv_file),
        )
        pipeline.add_stage(load_stage)

        # Stage 2: Split features and target
        def split_features(context):
            df = context["load_data"]["data"]
            X = df.drop("target", axis=1).values
            y = df["target"].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        split_stage = FunctionStage(
            name="split_data",
            func=split_features,
        )
        pipeline.add_stage(split_stage)
        pipeline.add_dependency("load_data", "split_data")

        # Stage 3: Train model
        def train_model(context):
            from sklearn.ensemble import RandomForestClassifier

            X_train = context["split_data"]["X_train"]
            y_train = context["split_data"]["y_train"]

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            return {"model": model}

        train_stage = FunctionStage(
            name="train_model",
            func=train_model,
        )
        pipeline.add_stage(train_stage)
        pipeline.add_dependency("split_data", "train_model")

        # Stage 4: Evaluate model
        def evaluate_model(context):
            from sklearn.metrics import accuracy_score

            model = context["train_model"]["model"]
            X_test = context["split_data"]["X_test"]
            y_test = context["split_data"]["y_test"]

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            return {"accuracy": accuracy, "predictions": predictions}

        eval_stage = FunctionStage(
            name="evaluate",
            func=evaluate_model,
        )
        pipeline.add_stage(eval_stage)
        pipeline.add_dependency("train_model", "evaluate")

        # Run the pipeline
        results = pipeline.run()

        # Assertions
        assert "load_data" in results
        assert "split_data" in results
        assert "train_model" in results
        assert "evaluate" in results

        assert "accuracy" in results["evaluate"]
        assert results["evaluate"]["accuracy"] >= 0.0
        assert results["evaluate"]["accuracy"] <= 1.0

        assert "predictions" in results["evaluate"]
        assert len(results["evaluate"]["predictions"]) == len(results["split_data"]["X_test"])

    def test_data_transformation_pipeline(self):
        """Test a data transformation pipeline."""
        pipeline = Pipeline(name="transform_pipeline")

        # Stage 1: Generate data
        def generate_data(context):
            df = pd.DataFrame(
                {
                    "A": np.random.randn(100),
                    "B": np.random.randn(100),
                    "C": np.random.randn(100),
                }
            )
            return {"raw_data": df}

        gen_stage = FunctionStage(name="generate", func=generate_data)
        pipeline.add_stage(gen_stage)

        # Stage 2: Normalize data
        def normalize_data(context):
            df = context["generate"]["raw_data"]
            normalized = (df - df.mean()) / df.std()
            return {"normalized_data": normalized}

        norm_stage = FunctionStage(name="normalize", func=normalize_data)
        pipeline.add_stage(norm_stage)
        pipeline.add_dependency("generate", "normalize")

        # Stage 3: Add derived features
        def add_features(context):
            df = context["normalize"]["normalized_data"]
            df["A_B_product"] = df["A"] * df["B"]
            df["A_B_sum"] = df["A"] + df["B"]
            return {"featured_data": df}

        feature_stage = FunctionStage(name="add_features", func=add_features)
        pipeline.add_stage(feature_stage)
        pipeline.add_dependency("normalize", "add_features")

        # Run pipeline
        results = pipeline.run()

        # Assertions
        assert "generate" in results
        assert "normalize" in results
        assert "add_features" in results

        final_df = results["add_features"]["featured_data"]
        assert "A_B_product" in final_df.columns
        assert "A_B_sum" in final_df.columns
        assert len(final_df) == 100

        # Check normalization
        norm_df = results["normalize"]["normalized_data"]
        assert abs(norm_df["A"].mean()) < 1e-10
        assert abs(norm_df["A"].std() - 1.0) < 1e-10

    def test_parallel_stages_pipeline(self):
        """Test a pipeline with parallel stages."""
        pipeline = Pipeline(name="parallel_pipeline")

        # Initial stage
        def init_stage(context):
            return {"data": list(range(10))}

        init = FunctionStage(name="init", func=init_stage)
        pipeline.add_stage(init)

        # Parallel stage 1
        def process_a(context):
            data = context["init"]["data"]
            return {"result_a": [x * 2 for x in data]}

        stage_a = FunctionStage(name="process_a", func=process_a)
        pipeline.add_stage(stage_a)
        pipeline.add_dependency("init", "process_a")

        # Parallel stage 2
        def process_b(context):
            data = context["init"]["data"]
            return {"result_b": [x**2 for x in data]}

        stage_b = FunctionStage(name="process_b", func=process_b)
        pipeline.add_stage(stage_b)
        pipeline.add_dependency("init", "process_b")

        # Merge stage
        def merge(context):
            a = context["process_a"]["result_a"]
            b = context["process_b"]["result_b"]
            return {"merged": [a[i] + b[i] for i in range(len(a))]}

        merge_stage = FunctionStage(name="merge", func=merge)
        pipeline.add_stage(merge_stage)
        pipeline.add_dependency("process_a", "merge")
        pipeline.add_dependency("process_b", "merge")

        # Run pipeline
        results = pipeline.run()

        # Assertions
        assert len(results) == 4
        assert results["init"]["data"] == list(range(10))
        assert results["process_a"]["result_a"] == [x * 2 for x in range(10)]
        assert results["process_b"]["result_b"] == [x**2 for x in range(10)]
        assert results["merge"]["merged"] == [2 * i + i**2 for i in range(10)]

    def test_error_handling_pipeline(self):
        """Test pipeline error handling."""
        pipeline = Pipeline(name="error_pipeline")

        # Stage that succeeds
        def good_stage(context):
            return {"success": True}

        # Stage that fails
        def bad_stage(context):
            raise RuntimeError("Intentional error")

        good = FunctionStage(name="good", func=good_stage)
        bad = FunctionStage(name="bad", func=bad_stage)

        pipeline.add_stage(good)
        pipeline.add_stage(bad)
        pipeline.add_dependency("good", "bad")

        # Should raise error when running
        with pytest.raises(RuntimeError, match="Intentional error"):
            pipeline.run()

    @pytest.mark.slow
    def test_large_pipeline(self):
        """Test a large pipeline with many stages."""
        pipeline = Pipeline(name="large_pipeline")

        # Create 50 stages
        for i in range(50):

            def stage_func(context, stage_num=i):
                prev_sum = context.get(f"stage_{stage_num - 1}", {}).get("sum", 0)
                return {"sum": prev_sum + stage_num}

            stage = FunctionStage(
                name=f"stage_{i}",
                func=stage_func,
            )
            pipeline.add_stage(stage)

            if i > 0:
                pipeline.add_dependency(f"stage_{i - 1}", f"stage_{i}")

        # Run pipeline
        results = pipeline.run()

        # Check final result
        assert len(results) == 50
        assert results["stage_49"]["sum"] == sum(range(50))

    def test_conditional_pipeline(self):
        """Test pipeline with conditional execution logic."""
        pipeline = Pipeline(name="conditional_pipeline")

        # Initial stage
        def init_stage(context):
            import random

            return {"value": random.choice([0, 1])}

        init = FunctionStage(name="init", func=init_stage)
        pipeline.add_stage(init)

        # Conditional processing
        def process(context):
            value = context["init"]["value"]
            if value == 0:
                return {"result": "path_a", "processed": value * 10}
            else:
                return {"result": "path_b", "processed": value * 100}

        process_stage = FunctionStage(name="process", func=process)
        pipeline.add_stage(process_stage)
        pipeline.add_dependency("init", "process")

        # Run multiple times to test both paths
        for _ in range(10):
            results = pipeline.run()
            value = results["init"]["value"]
            result = results["process"]["result"]
            processed = results["process"]["processed"]

            if value == 0:
                assert result == "path_a"
                assert processed == 0
            else:
                assert result == "path_b"
                assert processed == 100
