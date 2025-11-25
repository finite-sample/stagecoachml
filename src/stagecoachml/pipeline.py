"""Core pipeline implementation for orchestrating ML workflows."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, field_validator

from stagecoachml.stage import Stage

logger = logging.getLogger(__name__)


class Pipeline(BaseModel):
    """Orchestrate machine learning workflows as directed acyclic graphs."""

    name: str
    stages: list[Stage] = Field(default_factory=list)
    graph: nx.DiGraph = Field(default_factory=nx.DiGraph, exclude=True)
    config: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pipeline name cannot be empty")
        return v.strip()

    def add_stage(self, stage: Stage) -> Pipeline:
        """Add a stage to the pipeline."""
        if stage.name in [s.name for s in self.stages]:
            raise ValueError(f"Stage '{stage.name}' already exists in pipeline")
        self.stages.append(stage)
        self.graph.add_node(stage.name, stage=stage)
        return self

    def add_dependency(self, upstream: str, downstream: str) -> Pipeline:
        """Create a dependency between two stages."""
        upstream_stage = self._get_stage(upstream)
        downstream_stage = self._get_stage(downstream)

        self.graph.add_edge(upstream_stage.name, downstream_stage.name)

        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(upstream_stage.name, downstream_stage.name)
            raise ValueError(f"Adding dependency creates a cycle: {upstream} -> {downstream}")

        return self

    def _get_stage(self, name: str) -> Stage:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"Stage '{name}' not found in pipeline")

    def get_execution_order(self) -> list[str]:
        """Get stages in topological execution order."""
        if not self.stages:
            return []
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Pipeline contains cycles: {e}")

    def validate(self) -> bool:
        """Validate the pipeline structure."""
        if not self.stages:
            logger.warning("Pipeline has no stages")
            return True

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Pipeline must be a directed acyclic graph")

        disconnected = set(self.graph.nodes()) - set().union(
            *[set(comp) for comp in nx.weakly_connected_components(self.graph)]
        )
        if disconnected:
            logger.warning(f"Stages are not connected to pipeline: {disconnected}")

        return True

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the pipeline in topological order."""
        self.validate()
        context = context or {}
        results: dict[str, Any] = {}

        execution_order = self.get_execution_order()
        logger.info(f"Executing pipeline '{self.name}' with {len(execution_order)} stages")

        for stage_name in execution_order:
            stage = self._get_stage(stage_name)
            logger.info(f"Running stage: {stage_name}")

            stage_context = {**context, **results}
            result = stage.execute(stage_context)
            results[stage_name] = result

        logger.info(f"Pipeline '{self.name}' completed successfully")
        return results

    def visualize(self) -> str:
        """Generate a text representation of the pipeline."""
        if not self.stages:
            return f"Pipeline '{self.name}' (empty)"

        lines = [f"Pipeline: {self.name}"]
        lines.append("=" * 40)

        for stage in self.stages:
            predecessors = list(self.graph.predecessors(stage.name))
            successors = list(self.graph.successors(stage.name))

            lines.append(f"Stage: {stage.name}")
            if predecessors:
                lines.append(f"  Dependencies: {', '.join(predecessors)}")
            if successors:
                lines.append(f"  Leads to: {', '.join(successors)}")

        return "\n".join(lines)
