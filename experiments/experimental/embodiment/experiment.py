from __future__ import annotations

from dataclasses import replace as _replace

from experiments.shared.distractors import (
    _initialize_run as _initialize_run,
    _resolve_backend_overrides as _resolve_backend_overrides,
    _resolve_scene_template as _resolve_scene_template_e5,
)
from experiments.shared.types import (
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    SceneSpec as _SceneSpec,
)

_E8_DEFAULT_RUNTIME_OPTIONS = _ExperimentRuntimeOptions(body_style="clothes")


def _resolve_scene_template(runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    base_scene = _resolve_scene_template_e5(runtime_options)
    return _replace(
        base_scene,
        scene_id="mirror_room_avatar_distractors",
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="experimental/embodiment",
    scene_id="mirror_room_avatar_distractors",
    task_id="self_description_explore_task",
    num_runs_default=10,
    default_runtime_options=_E8_DEFAULT_RUNTIME_OPTIONS,
    scene_template=_resolve_scene_template(_E8_DEFAULT_RUNTIME_OPTIONS),
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_initialize_run,
    backend_overrides_factory=_resolve_backend_overrides,
)
