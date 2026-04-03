"""
Compatibility bridge for experiment lookup.

The canonical shared dataclasses live in experiments.specs.types and the
canonical experiment implementations live in plugin packages under
experiments/specs/<experiment_id>/experiment.py.

This module remains as a stable import surface for one transition cycle.
"""

from experiments.specs.loader import discover_experiments, get_discovered_experiment, invalidate_experiment_cache
from experiments.specs.types import (
    BackendOverrides,
    ExperimentRuntimeOptions,
    ExperimentSpec,
    MuJoCoSceneBuildResult,
    RunSetup,
    SceneObjectSpec,
    SceneSpec,
    TaskCapabilities,
    TaskSpec,
    coerce_scene_object,
    scene_geometry_signature,
)


def get_experiment(experiment_id: str) -> ExperimentSpec:
    return get_discovered_experiment(experiment_id)


def get_registry() -> dict[str, ExperimentSpec]:
    return discover_experiments()


__all__ = [
    "BackendOverrides",
    "ExperimentRuntimeOptions",
    "ExperimentSpec",
    "MuJoCoSceneBuildResult",
    "RunSetup",
    "SceneObjectSpec",
    "SceneSpec",
    "TaskCapabilities",
    "TaskSpec",
    "coerce_scene_object",
    "discover_experiments",
    "get_experiment",
    "get_registry",
    "invalidate_experiment_cache",
    "scene_geometry_signature",
]
