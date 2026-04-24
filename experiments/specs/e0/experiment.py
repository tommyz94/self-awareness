from __future__ import annotations

import random as _random
from dataclasses import replace as _replace

from experiments.specs.builtin_helpers import (
    _AGENT_COLOR_NAMES as _AGENT_COLOR_NAMES,
    build_mirror_room_scene as _build_mirror_room_scene,
    sample_non_agent_wall_rgba as _sample_non_agent_wall_rgba,
)
from experiments.specs.types import (
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    RunSetup as _RunSetup,
    SceneSpec as _SceneSpec,
)

_SCENE_TEMPLATE = _replace(
    _build_mirror_room_scene(mirror_enabled=True),
    scene_id="mirror_room_direct_self_view",
    spawn_pos=(-1.2, 0.0),
    spawn_yaw=0.0,
    mirror_window_z=(-7.5, 7.5),
    turn_step_deg=30.0,
    objects=[],
)


def _resolve_scene_template(_runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    return _SCENE_TEMPLATE


def _build_run_setup(
    run_id: int,
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
) -> _RunSetup:
    del run_id
    agent_color = _random.choice(_AGENT_COLOR_NAMES)
    wall_color_rgba = (
        _sample_non_agent_wall_rgba(excluded_color_names=[agent_color])
        if runtime_options.random_wall_color
        else scene_template.wall_color_rgba
    )
    scene_spec = _replace(
        scene_template,
        wall_color_rgba=wall_color_rgba,
        objects=[],
    )
    return _RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=None,
        metadata={
            "setup": "direct-mirror-self-perception",
            "self_perception_mode": str(
                runtime_options.self_perception_mode or "color"
            ).strip().lower(),
        },
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="e0",
    scene_id=_SCENE_TEMPLATE.scene_id,
    task_id="self_perception_task",
    num_runs_default=10,
    scene_template=_SCENE_TEMPLATE,
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_build_run_setup,
)
