from __future__ import annotations

from dataclasses import replace as _replace
import random as _random

from experiments.specs.builtin_helpers import (
    _AGENT_COLOR_NAMES as _AGENT_COLOR_NAMES,
    build_mirror_room_scene as _build_mirror_room_scene,
    sample_non_agent_wall_rgba as _sample_non_agent_wall_rgba,
)
from experiments.specs.e5.experiment import (
    _MAX_DISTRACTORS_PER_RUN as _MAX_DISTRACTORS_PER_RUN,
    _MIN_DISTRACTORS_PER_RUN as _MIN_DISTRACTORS_PER_RUN,
    _apply_mujoco_reset as _apply_mujoco_reset,
    _apply_mujoco_step as _apply_mujoco_step,
    _build_mujoco_scene as _build_mujoco_scene,
    _distractor_placeholders as _distractor_placeholders,
    _randomized_mirror_window_z as _randomized_mirror_window_z,
    _reflect_pos_x as _reflect_pos_x,
    _resolve_mirror_plane_x as _resolve_mirror_plane_x,
    _sample_distractor_colors as _sample_distractor_colors,
    _sample_distractor_start_positions as _sample_distractor_start_positions,
    _sample_cardinal_yaw as _sample_cardinal_yaw,
    _validate_distractor_slots as _validate_distractor_slots,
)
from experiments.specs.types import (
    BackendOverrides as _BackendOverrides,
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    MuJoCoSceneBuildResult as _MuJoCoSceneBuildResult,
    RunSetup as _RunSetup,
    SceneObjectSpec as _SceneObjectSpec,
    SceneSpec as _SceneSpec,
)


def _resolve_scene_template(runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del runtime_options
    _validate_distractor_slots()
    base_scene = _build_mirror_room_scene(mirror_enabled=True)
    return _replace(
        base_scene,
        scene_id="mirror_room_robot_distractors_no_cubes",
        spawn_pos=(-10.0, 0.0),
        mirror_window_z=(-4.5, 4.5),
        turn_step_deg=30.0,
        objects=_distractor_placeholders(_MAX_DISTRACTORS_PER_RUN),
    )


def _initialize_run(
    run_id: int,
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
) -> _RunSetup:
    del run_id

    mirror_plane_x = _resolve_mirror_plane_x(scene_template)
    use_reflected_room = _random.choice([False, True])
    spawn_pos = scene_template.spawn_pos
    if use_reflected_room:
        spawn_pos = _reflect_pos_x(spawn_pos, mirror_plane_x)
    spawn_yaw = _sample_cardinal_yaw()
    mirror_window_z = _randomized_mirror_window_z(
        scene_template,
        center_jitter=0.6,
    )
    agent_color = _random.choice(_AGENT_COLOR_NAMES)

    distractor_placeholders = [
        obj
        for obj in scene_template.objects
        if obj.kind == "distractor_robot"
    ]
    active_distractors = _random.randint(_MIN_DISTRACTORS_PER_RUN, _MAX_DISTRACTORS_PER_RUN)
    selected_placeholders = _random.sample(distractor_placeholders, active_distractors)
    distractor_positions = _sample_distractor_start_positions(
        scene_template,
        mirror_plane_x=mirror_plane_x,
        count=active_distractors,
        use_reflected_room=use_reflected_room,
        mirror_window_z=mirror_window_z,
    )
    distractor_colors = _sample_distractor_colors(
        count=len(selected_placeholders),
        agent_color=agent_color,
        color_mode=runtime_options.distractor_color_mode,
    )
    distractor_objects = [
        _SceneObjectSpec(
            object_id=obj.object_id,
            kind=obj.kind,
            position=distractor_positions[index],
            color=distractor_colors[index],
            metadata=dict(obj.metadata),
        )
        for index, obj in enumerate(selected_placeholders)
    ]
    excluded_wall_colors = [agent_color, *distractor_colors]
    scene_spec = _replace(
        scene_template,
        spawn_pos=spawn_pos,
        spawn_yaw=spawn_yaw,
        mirror_window_z=mirror_window_z,
        wall_color_rgba=(
            _sample_non_agent_wall_rgba(excluded_color_names=excluded_wall_colors)
            if runtime_options.random_wall_color
            else scene_template.wall_color_rgba
        ),
        objects=distractor_objects,
    )
    return _RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=None,
    )


def _resolve_backend_overrides(
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
) -> _BackendOverrides:
    del scene_template

    def _scene_builder(scene_spec: _SceneSpec, render_width: int, render_height: int) -> _MuJoCoSceneBuildResult:
        return _build_mujoco_scene(
            scene_spec,
            render_width,
            render_height,
            camera_height=runtime_options.camera_height,
            body_style=runtime_options.body_style,
        )

    return _BackendOverrides(
        mujoco_scene_builder=_scene_builder,
        mujoco_reset_applier=_apply_mujoco_reset,
        mujoco_step_applier=_apply_mujoco_step,
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="e6",
    scene_id="mirror_room_robot_distractors_no_cubes",
    task_id="explore_task",
    num_runs_default=10,
    default_runtime_options=_ExperimentRuntimeOptions(),
    scene_template=_resolve_scene_template(_ExperimentRuntimeOptions()),
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_initialize_run,
    backend_overrides_factory=_resolve_backend_overrides,
)
