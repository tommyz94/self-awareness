from __future__ import annotations

from dataclasses import replace as _replace
from pathlib import Path as _Path
import random as _random

from env.backends.mujoco import (
    _inject_asset_entries as _inject_asset_entries,
    _surface_texture_assets_xml as _surface_texture_assets_xml,
    _two_room_lights_xml as _two_room_lights_xml,
    _two_room_visual_xml as _two_room_visual_xml,
)
from env.scenes.mujoco_builder import (
    build_humanoid_scene as _build_humanoid_scene,
    build_humanoid_wrapper as _build_humanoid_wrapper,
    sample_avatar_mesh_choice as _sample_avatar_mesh_choice,
)
from experiments.shared.scenes import (
    _AGENT_COLOR_NAMES as _AGENT_COLOR_NAMES,
    build_mirror_room_scene as _build_mirror_room_scene,
    sample_non_agent_wall_rgba as _sample_non_agent_wall_rgba,
)
from experiments.shared.types import (
    BackendOverrides as _BackendOverrides,
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    MuJoCoSceneBuildResult as _MuJoCoSceneBuildResult,
    RunSetup as _RunSetup,
    SceneObjectSpec as _SceneObjectSpec,
    SceneSpec as _SceneSpec,
)
from utils.colors import rgba_for_color as _rgba_for_color

_ASSETS_DIR = _Path(__file__).resolve().parents[3] / "assets" / "mujoco"
_CUBE_XML_PATH = _ASSETS_DIR / "cube.xml"
_MUJOCO_SPAWN_Y = -1.0
_BOOTH_CENTER_Z = {
    "left": -2.8,
    "right": 2.8,
}
_BOOTH_RANGES = {
    "left": (-4.6, -1.0),
    "right": (1.0, 4.6),
}
_WINDOW_SPAN_Z = (-4.6, 4.6)
_DISTRACTOR_OBJECT_ID = "distractor_01"
_CONFIG_OBJECT_ID = "choice_config"
_PANE_MIRROR_RGBA = [0.86, 0.89, 0.93, 0.14]
_PANE_WINDOW_RGBA = [0.88, 0.95, 1.0, 0.05]
_CUBE_XML_RGBA = {"red": "1 0 0 1", "green": "0 0.8 0 1", "blue": "0 0.3 1 1"}
_MIRROR_CUBE_XML_RGBA = {"red": "1 0 0 1", "green": "0 0.8 0 1", "blue": "0 0.3 1 1"}
_INITIAL_CUBE_POSITIONS = {
    "red": (-13.0, -6.0),
    "green": (-13.0, 6.0),
    "blue": (-2.0, -6.0),
}


def _opposite_side(side: str) -> str:
    return "right" if side == "left" else "left"


def _reflect_x(x: float, mirror_plane_x: float) -> float:
    return round(2 * mirror_plane_x - float(x), 3)


def _booth_center(side: str) -> float:
    return float(_BOOTH_CENTER_Z[side])


def _distractor_position(scene_spec: _SceneSpec, *, real_side: str, spawn_x: float) -> tuple[float, float]:
    mirror_plane_x = (
        float(scene_spec.mirror_plane_x)
        if scene_spec.mirror_plane_x is not None
        else -scene_spec.wall_thickness / 2
    )
    return (_reflect_x(spawn_x, mirror_plane_x), _booth_center(real_side))


def _build_choice_objects(
    scene_spec: _SceneSpec,
    *,
    mirror_side: str,
    agent_color: str,
) -> list[_SceneObjectSpec]:
    real_side = _opposite_side(mirror_side)
    spawn_x = float(scene_spec.spawn_pos[0])
    distractor_pos = _distractor_position(scene_spec, real_side=real_side, spawn_x=spawn_x)
    return [
        _SceneObjectSpec(
            object_id=_CONFIG_OBJECT_ID,
            kind="mirror_choice_config",
            position=(0.0, 0.0),
            metadata={
                "mirror_side": mirror_side,
                "real_robot_side": real_side,
            },
        ),
        _SceneObjectSpec(
            object_id=_DISTRACTOR_OBJECT_ID,
            kind="distractor_robot",
            position=distractor_pos,
            color=agent_color,
            metadata={
                "yaw": 0.0,
                "side": real_side,
            },
        ),
    ]


def _resolve_scene_template(runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del runtime_options
    base_scene = _build_mirror_room_scene(mirror_enabled=True)
    scene_spec = _replace(
        base_scene,
        scene_id="mirror_side_choice_room",
        spawn_pos=(-10.0, _booth_center("left")),
        spawn_yaw=0.0,
        mirror_window_z=_WINDOW_SPAN_Z,
        turn_step_deg=30.0,
    )
    return _replace(
        scene_spec,
        objects=_build_choice_objects(
            scene_spec,
            mirror_side="left",
            agent_color="red",
        ),
    )


def _initialize_run(
    run_id: int,
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
) -> _RunSetup:
    del run_id
    mirror_side = _random.choice(("left", "right"))
    real_side = _opposite_side(mirror_side)
    agent_color = _random.choice(_AGENT_COLOR_NAMES)
    spawn_pos = (float(scene_template.spawn_pos[0]), _booth_center(mirror_side))
    scene_spec = _replace(
        scene_template,
        spawn_pos=spawn_pos,
        wall_color_rgba=(
            _sample_non_agent_wall_rgba(excluded_color_names=[agent_color])
            if runtime_options.random_wall_color
            else scene_template.wall_color_rgba
        ),
        objects=_build_choice_objects(
            _replace(scene_template, spawn_pos=spawn_pos),
            mirror_side=mirror_side,
            agent_color=agent_color,
        ),
    )
    run_setup = _RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=None,
        cube_positions={},
    )
    run_setup.metadata.update({
        "mirror_side": mirror_side,
        "real_robot_side": real_side,
        "spawn_side": mirror_side,
    })
    return run_setup


def _load_cube_template() -> str:
    if not _CUBE_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo cube asset not found at {_CUBE_XML_PATH}")
    return _CUBE_XML_PATH.read_text().strip()


def _build_cube_xml(*, include_mirror_cubes: bool) -> str:
    template = _load_cube_template()
    cube_bodies = []
    for color_name in ("red", "green", "blue"):
        x, z = _INITIAL_CUBE_POSITIONS[color_name]
        cube_bodies.append(
            template.format(
                body_name=f"cube_{color_name}",
                joint_name=f"cube_{color_name}_j",
                geom_name=f"cube_{color_name}_g",
                x=x,
                z=z,
                rgba=_CUBE_XML_RGBA[color_name],
            )
        )
    if include_mirror_cubes:
        for color_name in ("red", "green", "blue"):
            x, z = _INITIAL_CUBE_POSITIONS[color_name]
            cube_bodies.append(
                template.format(
                    body_name=f"mirror_cube_{color_name}",
                    joint_name=f"mirror_cube_{color_name}_j",
                    geom_name=f"mirror_cube_{color_name}_g",
                    x=x,
                    z=z,
                    rgba=_MIRROR_CUBE_XML_RGBA[color_name],
                )
            )
    return "\n    <!-- Hidden compatibility cubes -->\n    " + "\n    ".join(cube_bodies)


def _pane_rgba_xml(rgba: list[float]) -> str:
    return " ".join(f"{float(value):.3f}" for value in rgba)


def _build_mujoco_scene(
    scene_spec: _SceneSpec,
    render_width: int,
    render_height: int,
    camera_height: float = 1.0,
    body_style: str = "default",
) -> _MuJoCoSceneBuildResult:
    avatar_choice = (
        _sample_avatar_mesh_choice()
        if str(body_style).strip().lower() == "clothes"
        else None
    )
    humanoid_scene = _build_humanoid_scene(
        mirror_present=True,
        spawn_pos=(scene_spec.spawn_pos[0], _MUJOCO_SPAWN_Y, scene_spec.spawn_pos[1]),
        camera_height=camera_height,
        body_style=body_style,
        avatar_choice=avatar_choice,
    )
    asset_xml = _inject_asset_entries(
        humanoid_scene.asset_xml,
        _surface_texture_assets_xml(),
    )

    distractor_body_xml = ""
    distractor_geom_names: list[str] = []
    for obj in scene_spec.objects:
        if obj.kind != "distractor_robot":
            continue
        root_name = f"{obj.object_id}_root"
        body_xml, geom_names = _build_humanoid_wrapper(
            prefix=f"{obj.object_id}_",
            wrapper_name=root_name,
            root_joint_name=root_name,
            spawn_pos=(obj.position[0], _MUJOCO_SPAWN_Y, obj.position[1]),
            include_cameras=False,
            body_style=body_style,
            avatar_choice=avatar_choice,
        )
        distractor_body_xml = body_xml
        distractor_geom_names = geom_names
        break

    rl = scene_spec.length
    rw = scene_spec.width
    rh = scene_spec.height
    wt = scene_spec.wall_thickness
    cx = -(rl / 2 + wt / 2)
    room_min_z = -rw / 2
    room_max_z = rw / 2
    left_outer_len = _BOOTH_RANGES["left"][0] - room_min_z
    left_outer_center = (room_min_z + _BOOTH_RANGES["left"][0]) / 2
    divider_len = _BOOTH_RANGES["right"][0] - _BOOTH_RANGES["left"][1]
    divider_center = (_BOOTH_RANGES["left"][1] + _BOOTH_RANGES["right"][0]) / 2
    right_outer_len = room_max_z - _BOOTH_RANGES["right"][1]
    right_outer_center = (_BOOTH_RANGES["right"][1] + room_max_z) / 2
    left_pane_center = (_BOOTH_RANGES["left"][0] + _BOOTH_RANGES["left"][1]) / 2
    left_pane_len = _BOOTH_RANGES["left"][1] - _BOOTH_RANGES["left"][0]
    right_pane_center = (_BOOTH_RANGES["right"][0] + _BOOTH_RANGES["right"][1]) / 2
    right_pane_len = _BOOTH_RANGES["right"][1] - _BOOTH_RANGES["right"][0]

    xml = f"""<mujoco model="mirror_side_choice">
  <compiler angle="degree"/>
  <option gravity="0 -9.81 0" timestep="0.01"/>
  {_two_room_visual_xml(render_width, render_height)}
  {asset_xml}
  {humanoid_scene.default_xml}

  <worldbody>
    {_two_room_lights_xml()}
    <geom name="choice_floor_left" type="box"
          pos="{cx:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>
    <geom name="choice_floor_right" type="box"
          pos="{rl/2+wt/2:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_back_left" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_front_left" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_outer_left" type="box"
          pos="{-rl-wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_back_right" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_front_right" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_outer_right" type="box"
          pos="{rl+wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_ceiling_left" type="box"
          pos="{cx:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_ceiling_right" type="box"
          pos="{rl/2+wt/2:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_segment_left" type="box"
          pos="0 {rh/2-0.95:.3f} {left_outer_center:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {left_outer_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_segment_mid" type="box"
          pos="0 {rh/2-0.95:.3f} {divider_center:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {divider_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_wall_segment_right" type="box"
          pos="0 {rh/2-0.95:.3f} {right_outer_center:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {right_outer_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="choice_pane_left" type="box"
          pos="0 {rh/2-0.95:.3f} {left_pane_center:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {left_pane_len/2:.3f}"
          rgba="{_pane_rgba_xml(_PANE_MIRROR_RGBA)}" contype="0" conaffinity="0"/>
    <geom name="choice_pane_right" type="box"
          pos="0 {rh/2-0.95:.3f} {right_pane_center:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {right_pane_len/2:.3f}"
          rgba="{_pane_rgba_xml(_PANE_WINDOW_RGBA)}" contype="0" conaffinity="0"/>
    <geom name="choice_marker_left" type="box"
          pos="-5.6 -1.01 {left_pane_center:.3f}"
          size="2.8 .015 .42"
          rgba=".38 .40 .44 .55" contype="0" conaffinity="0"/>
    <geom name="choice_marker_right" type="box"
          pos="-5.6 -1.01 {right_pane_center:.3f}"
          size="2.8 .015 .42"
          rgba=".38 .40 .44 .55" contype="0" conaffinity="0"/>
    {_build_cube_xml(include_mirror_cubes=scene_spec.mirror_enabled)}

    {humanoid_scene.robot_body_xml}
    {humanoid_scene.mirror_body_xml}
    {distractor_body_xml}
  </worldbody>
</mujoco>"""
    return _MuJoCoSceneBuildResult(
        xml=xml,
        robot_geom_names=humanoid_scene.robot_geom_names,
        mirror_geom_names=humanoid_scene.mirror_geom_names,
    )


def _apply_pane_rgba(env, geom_name: str, rgba: list[float]) -> None:
    geom_id = env._geom_id(geom_name)
    env._model.geom_rgba[geom_id] = [float(value) for value in rgba]


def _apply_mujoco_reset(env, run_setup: _RunSetup) -> None:
    mirror_side = str(run_setup.metadata.get("mirror_side", "left")).strip().lower()
    if mirror_side not in {"left", "right"}:
        mirror_side = "left"
    if mirror_side == "left":
        _apply_pane_rgba(env, "choice_pane_left", _PANE_MIRROR_RGBA)
        _apply_pane_rgba(env, "choice_pane_right", _PANE_WINDOW_RGBA)
    else:
        _apply_pane_rgba(env, "choice_pane_left", _PANE_WINDOW_RGBA)
        _apply_pane_rgba(env, "choice_pane_right", _PANE_MIRROR_RGBA)

    textured_body_style = str(getattr(env, "_body_style", "default")).strip().lower() == "clothes"
    distractor = next(
        (obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"),
        None,
    )
    root_addr = env._jnt_addr_if_exists(f"{_DISTRACTOR_OBJECT_ID}_root")
    if root_addr is None:
        return
    if distractor is None:
        env._park_body(root_addr)
        return
    yaw = float(distractor.metadata.get("yaw", 0.0))
    x, z = distractor.position
    env._set_body_pose(root_addr, float(x), env._spawn_y, float(z), yaw)
    env._apply_arm_rest_pose_for_prefix(f"{_DISTRACTOR_OBJECT_ID}_")
    if not textured_body_style:
        env._set_color(
            env._geom_ids_with_prefix(f"{_DISTRACTOR_OBJECT_ID}_"),
            _rgba_for_color(distractor.color or run_setup.agent_color_name, alpha=1.0),
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
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="experimental/mirror_side_choice",
    scene_id="mirror_side_choice_room",
    task_id="mirror_side_choice_task",
    num_runs_default=10,
    default_runtime_options=_ExperimentRuntimeOptions(),
    scene_template=_resolve_scene_template(_ExperimentRuntimeOptions()),
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_initialize_run,
    backend_overrides_factory=_resolve_backend_overrides,
)
