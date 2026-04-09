from __future__ import annotations

from dataclasses import replace as _replace

from experiments.specs.builtin_helpers import (
    MIRROR_ROOM_WALL_CORNER_SLOTS as _MIRROR_ROOM_WALL_CORNER_SLOTS,
    build_cube_run_setup as _build_cube_run_setup,
    build_mirror_room_scene as _build_mirror_room_scene,
)
from experiments.specs.types import (
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    SceneSpec as _SceneSpec,
)

_SCENE_TEMPLATE = _replace(
    _build_mirror_room_scene(mirror_enabled=True),
    turn_step_deg=30.0,
)


def _resolve_scene_template(_runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    return _SCENE_TEMPLATE


def _build_run_setup(
    run_id: int,
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
):
    del run_id
    return _build_cube_run_setup(
        scene_template,
        cube_slots=_MIRROR_ROOM_WALL_CORNER_SLOTS,
        randomize_wall=True,
        randomize_wall_color=runtime_options.random_wall_color,
        mirror_window_center_jitter=2.5,
        mirror_window_cover_cubes=False,
        mirror_cubes_ignore_aperture=True,
        randomize_spawn_yaw=True,
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="e1",
    scene_id=_SCENE_TEMPLATE.scene_id,
    task_id="cube_task",
    num_runs_default=10,
    scene_template=_SCENE_TEMPLATE,
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_build_run_setup,
)
