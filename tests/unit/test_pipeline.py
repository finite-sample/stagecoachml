"""Unit tests for pipeline module."""

import pytest

from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage


class TestPipeline:
    """Test Pipeline class functionality."""

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipeline = Pipeline(name="test")
        assert pipeline.name == "test"
        assert len(pipeline.stages) == 0
        assert pipeline.validate()

    def test_pipeline_name_validation(self):
        """Test pipeline name validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Pipeline(name="")

        with pytest.raises(ValueError, match="cannot be empty"):
            Pipeline(name="   ")

    def test_add_stage(self):
        """Test adding stages to pipeline."""
        pipeline = Pipeline(name="test")
        stage = FunctionStage(name="stage1", func=lambda x: x)

        pipeline.add_stage(stage)
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "stage1"

    def test_add_duplicate_stage(self):
        """Test that duplicate stage names are rejected."""
        pipeline = Pipeline(name="test")
        stage1 = FunctionStage(name="stage1", func=lambda x: x)
        stage2 = FunctionStage(name="stage1", func=lambda x: x * 2)

        pipeline.add_stage(stage1)
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_stage(stage2)

    def test_add_dependency(self):
        """Test adding dependencies between stages."""
        pipeline = Pipeline(name="test")
        stage1 = FunctionStage(name="stage1", func=lambda x: x)
        stage2 = FunctionStage(name="stage2", func=lambda x: x * 2)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_dependency("stage1", "stage2")

        assert pipeline.graph.has_edge("stage1", "stage2")

    def test_cyclic_dependency_detection(self):
        """Test that cyclic dependencies are detected."""
        pipeline = Pipeline(name="test")
        stage1 = FunctionStage(name="stage1", func=lambda x: x)
        stage2 = FunctionStage(name="stage2", func=lambda x: x * 2)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_dependency("stage1", "stage2")

        with pytest.raises(ValueError, match="cycle"):
            pipeline.add_dependency("stage2", "stage1")

    def test_get_execution_order(self):
        """Test topological sorting of stages."""
        pipeline = Pipeline(name="test")

        stage1 = FunctionStage(name="stage1", func=lambda x: x)
        stage2 = FunctionStage(name="stage2", func=lambda x: x)
        stage3 = FunctionStage(name="stage3", func=lambda x: x)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)

        pipeline.add_dependency("stage1", "stage2")
        pipeline.add_dependency("stage2", "stage3")

        order = pipeline.get_execution_order()
        assert order.index("stage1") < order.index("stage2")
        assert order.index("stage2") < order.index("stage3")

    def test_run_empty_pipeline(self):
        """Test running an empty pipeline."""
        pipeline = Pipeline(name="test")
        result = pipeline.run()
        assert result == {}

    def test_run_simple_pipeline(self, simple_pipeline):
        """Test running a simple pipeline."""
        result = simple_pipeline.run()
        assert "stage1" in result
        assert "stage2" in result
        assert result["stage1"]["data"] == [1, 2, 3]
        assert result["stage2"]["processed"] == [2, 4, 6]

    def test_run_with_context(self):
        """Test running pipeline with initial context."""
        pipeline = Pipeline(name="test")

        def stage_func(context):
            return {"value": context.get("input_value", 0) * 2}

        stage = FunctionStage(name="stage1", func=stage_func)
        pipeline.add_stage(stage)

        result = pipeline.run({"input_value": 5})
        assert result["stage1"]["value"] == 10

    def test_visualize_empty_pipeline(self):
        """Test visualizing an empty pipeline."""
        pipeline = Pipeline(name="test")
        viz = pipeline.visualize()
        assert "test" in viz
        assert "empty" in viz

    def test_visualize_pipeline(self, simple_pipeline):
        """Test visualizing a pipeline with stages."""
        viz = simple_pipeline.visualize()
        assert "test_pipeline" in viz
        assert "stage1" in viz
        assert "stage2" in viz
        assert "Dependencies" in viz or "Leads to" in viz

    def test_get_nonexistent_stage(self):
        """Test getting a stage that doesn't exist."""
        pipeline = Pipeline(name="test")
        with pytest.raises(ValueError, match="not found"):
            pipeline._get_stage("nonexistent")

    def test_validate_dag(self):
        """Test DAG validation."""
        pipeline = Pipeline(name="test")
        stage1 = FunctionStage(name="stage1", func=lambda x: x)
        stage2 = FunctionStage(name="stage2", func=lambda x: x)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        # Create a cycle directly in the graph (for testing)
        pipeline.graph.add_edge("stage1", "stage2")
        pipeline.graph.add_edge("stage2", "stage1")

        with pytest.raises(ValueError, match="directed acyclic graph"):
            pipeline.validate()

    @pytest.mark.parametrize(
        "n_stages,n_dependencies",
        [(3, 2), (5, 4), (10, 15)],
    )
    def test_complex_pipeline(self, n_stages, n_dependencies):
        """Test pipelines with various complexities."""
        pipeline = Pipeline(name="complex")

        # Add stages
        for i in range(n_stages):
            stage = FunctionStage(
                name=f"stage_{i}",
                func=lambda x, i=i: {f"output_{i}": i},
            )
            pipeline.add_stage(stage)

        # Add some dependencies (ensuring no cycles)
        deps_added = 0
        for i in range(n_stages - 1):
            if deps_added >= n_dependencies:
                break
            pipeline.add_dependency(f"stage_{i}", f"stage_{i+1}")
            deps_added += 1

        assert pipeline.validate()
        result = pipeline.run()
        assert len(result) == n_stages
