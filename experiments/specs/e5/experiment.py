from __future__ import annotations

from dataclasses import replace as _replace
import math as _math
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
)
from experiments.specs.builtin_helpers import (
    MIRROR_ROOM_CUBE_CORNER_SLOTS as _MIRROR_ROOM_CUBE_CORNER_SLOTS,
    _AGENT_COLOR_NAMES as _AGENT_COLOR_NAMES,
    build_cube_objects as _build_cube_objects,
    build_mirror_room_scene as _build_mirror_room_scene,
    mirror_window_covering_positions as _mirror_window_covering_positions,
    sample_non_agent_wall_rgba as _sample_non_agent_wall_rgba,
)
from utils.colors import rgba_for_color as _rgba_for_color
from experiments.specs.types import (
    BackendOverrides as _BackendOverrides,
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    MuJoCoSceneBuildResult as _MuJoCoSceneBuildResult,
    RunSetup as _RunSetup,
    SceneObjectSpec as _SceneObjectSpec,
    SceneSpec as _SceneSpec,
)

_ASSETS_DIR = _Path(__file__).resolve().parents[3] / "assets" / "mujoco"
_CUBE_XML_PATH = _ASSETS_DIR / "cube.xml"
_MUJOCO_SPAWN_Y = -1.0
_CUBE_XML_RGBA = {"red": "1 0 0 1", "green": "0 0.8 0 1", "blue": "0 0.3 1 1"}
_MIRROR_CUBE_XML_RGBA = {"red": "1 0 0 1", "green": "0 0.8 0 1", "blue": "0 0.3 1 1"}
_INITIAL_CUBE_POSITIONS = {
    "red": (-13.0, -6.0),
    "green": (-13.0, 6.0),
    "blue": (-2.0, -6.0),
}
_DISTRACTOR_SLOTS = [
    (-3.8, -3.9, 0.0),
    (-3.8, -2.3, 0.0),
    (-3.8, -0.7, 0.0),
    (-3.8, 0.7, 0.0),
    (-3.8, 2.3, 0.0),
    (-3.8, 3.9, 0.0),
]
_DISTRACTOR_ACTION_MOVE_STEP = 2.0
_DISTRACTOR_TARGET_REACHED_EPS = 1.0
_DISTRACTOR_TARGET_MARGIN_X = 0.9
_DISTRACTOR_TARGET_MARGIN_Z = 0.7
_DISTRACTOR_PROGRESS_EPS = 0.05
_DISTRACTOR_STUCK_RESAMPLE_STEPS = 5
_DISTRACTOR_TURN_SWITCH_RESAMPLE = 4
_DISTRACTOR_START_MIN_SEPARATION = 1.1
_MIN_DISTRACTORS_PER_RUN = 1
_MAX_DISTRACTORS_PER_RUN = 6
_DISTRACTOR_COLOR_MODE_ALLOW_AGENT = "allow_agent_color"
_DISTRACTOR_COLOR_MODE_EXCLUDE_AGENT = "exclude_agent_color"
_DISTRACTOR_COLOR_MODE_ALLOW_DUPLICATES = "allow_duplicates"
_DISTRACTOR_COLOR_MODES = {
    _DISTRACTOR_COLOR_MODE_ALLOW_AGENT,
    _DISTRACTOR_COLOR_MODE_EXCLUDE_AGENT,
    _DISTRACTOR_COLOR_MODE_ALLOW_DUPLICATES,
}


def _resolve_mirror_plane_x(scene_spec: _SceneSpec) -> float:
    if scene_spec.mirror_plane_x is not None:
        return float(scene_spec.mirror_plane_x)
    return -scene_spec.wall_thickness / 2


def _reflect_pos_x(position: tuple[float, float], mirror_plane_x: float) -> tuple[float, float]:
    x, z = position
    return (round(2 * mirror_plane_x - float(x), 3), float(z))


def _sample_cardinal_yaw() -> float:
    return _random.choice((0.0, _math.pi / 2, _math.pi, 3 * _math.pi / 2))


def _sample_distractor_colors(
    *,
    count: int,
    agent_color: str,
    color_mode: str,
) -> list[str]:
    mode = str(color_mode).strip().lower()
    if mode not in _DISTRACTOR_COLOR_MODES:
        raise ValueError(
            "Unsupported distractor_color_mode="
            f"{color_mode!r}. Valid values: {sorted(_DISTRACTOR_COLOR_MODES)}"
        )
    if count <= 0:
        return []
    if mode == _DISTRACTOR_COLOR_MODE_ALLOW_DUPLICATES:
        return [_random.choice(_AGENT_COLOR_NAMES) for _ in range(count)]

    if mode == _DISTRACTOR_COLOR_MODE_EXCLUDE_AGENT:
        pool = [color for color in _AGENT_COLOR_NAMES if color != agent_color]
    else:
        pool = list(_AGENT_COLOR_NAMES)

    if not pool:
        return []
    if count <= len(pool):
        return _random.sample(pool, count)

    # Fallback for robustness: keep unique colors first, then allow repeats.
    colors = list(pool)
    while len(colors) < count:
        colors.append(_random.choice(pool))
    return colors


def _randomized_mirror_window_z(
    scene_spec: _SceneSpec,
    *,
    center_jitter: float | None,
) -> tuple[float, float] | None:
    if (
        center_jitter is None
        or not scene_spec.mirror_enabled
        or scene_spec.mirror_aperture_mode != "window"
        or scene_spec.mirror_window_z is None
    ):
        return scene_spec.mirror_window_z

    z_min, z_max = scene_spec.mirror_window_z
    width = float(z_max - z_min)
    if width <= 0:
        return scene_spec.mirror_window_z

    room_min = -scene_spec.width / 2 + 0.1
    room_max = scene_spec.width / 2 - 0.1
    half_width = width / 2
    center = (z_min + z_max) / 2
    center_min = room_min + half_width
    center_max = room_max - half_width
    if center_min > center_max:
        return scene_spec.mirror_window_z

    max_left = max(0.0, center - center_min)
    max_right = max(0.0, center_max - center)
    jitter = abs(float(center_jitter))
    shift_min = -min(jitter, max_left)
    shift_max = min(jitter, max_right)
    if shift_min == shift_max:
        new_center = center
    else:
        new_center = center + _random.uniform(shift_min, shift_max)

    return (
        round(new_center - half_width, 3),
        round(new_center + half_width, 3),
    )


def _validate_distractor_slots() -> None:
    if len(_DISTRACTOR_SLOTS) < _MAX_DISTRACTORS_PER_RUN:
        raise ValueError(
            f"e5 requires at least {_MAX_DISTRACTORS_PER_RUN} distractor slots, got {len(_DISTRACTOR_SLOTS)}."
        )


def _distractor_nav_bounds(
    scene_spec: _SceneSpec,
    *,
    mirror_plane_x: float,
    x: float,
) -> tuple[float, float, float, float]:
    z_min = -scene_spec.width / 2 + _DISTRACTOR_TARGET_MARGIN_Z
    z_max = scene_spec.width / 2 - _DISTRACTOR_TARGET_MARGIN_Z
    left_x_min = -scene_spec.length - scene_spec.wall_thickness + _DISTRACTOR_TARGET_MARGIN_X
    left_x_max = mirror_plane_x - _DISTRACTOR_TARGET_MARGIN_X
    right_x_min = mirror_plane_x + _DISTRACTOR_TARGET_MARGIN_X
    right_x_max = scene_spec.length + scene_spec.wall_thickness - _DISTRACTOR_TARGET_MARGIN_X
    if x <= mirror_plane_x:
        if left_x_max <= left_x_min:
            left_x_max = left_x_min + 0.1
        return (left_x_min, left_x_max, z_min, z_max)
    if right_x_max <= right_x_min:
        right_x_max = right_x_min + 0.1
    return (right_x_min, right_x_max, z_min, z_max)


def _sample_distractor_target(
    bounds: tuple[float, float, float, float],
    *,
    current: tuple[float, float],
) -> tuple[float, float]:
    x_min, x_max, z_min, z_max = bounds
    cx, cz = current
    for _ in range(24):
        tx = _random.uniform(x_min, x_max)
        tz = _random.uniform(z_min, z_max)
        if _math.hypot(tx - cx, tz - cz) >= 1.0:
            return (round(tx, 3), round(tz, 3))
    return (
        round(min(max(cx, x_min), x_max), 3),
        round(min(max(cz, z_min), z_max), 3),
    )


def _sample_distractor_start_positions(
    scene_spec: _SceneSpec,
    *,
    mirror_plane_x: float,
    count: int,
    use_reflected_room: bool,
    mirror_window_z: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []

    side_probe_x = mirror_plane_x + 2.0 if use_reflected_room else mirror_plane_x - 2.0
    x_min, x_max, z_min, z_max = _distractor_nav_bounds(
        scene_spec,
        mirror_plane_x=mirror_plane_x,
        x=side_probe_x,
    )

    near_span = min(4.2, max(2.0, scene_spec.length * 0.30))
    if use_reflected_room:
        near_x_min = mirror_plane_x + 1.0
        near_x_max = mirror_plane_x + near_span
    else:
        near_x_min = mirror_plane_x - near_span
        near_x_max = mirror_plane_x - 1.0
    sx_min = max(x_min, near_x_min)
    sx_max = min(x_max, near_x_max)
    if sx_min >= sx_max:
        sx_min, sx_max = x_min, x_max

    if mirror_window_z is not None:
        win_min, win_max = mirror_window_z
        win_margin = min(0.45, max(0.0, (win_max - win_min) * 0.08))
        sz_min = max(z_min, win_min + win_margin)
        sz_max = min(z_max, win_max - win_margin)
    else:
        sz_min, sz_max = z_min, z_max
    if sz_min >= sz_max:
        sz_min, sz_max = z_min, z_max

    sampled: list[tuple[float, float]] = []
    for _ in range(count):
        fallback = (round(_random.uniform(sx_min, sx_max), 3), round(_random.uniform(sz_min, sz_max), 3))
        chosen = fallback
        for _attempt in range(48):
            px = round(_random.uniform(sx_min, sx_max), 3)
            pz = round(_random.uniform(sz_min, sz_max), 3)
            if all(_math.hypot(px - ox, pz - oz) >= _DISTRACTOR_START_MIN_SEPARATION for ox, oz in sampled):
                chosen = (px, pz)
                break
        sampled.append(chosen)
    return sampled


def _normalize_angle(angle: float) -> float:
    return (angle + _math.pi) % (2 * _math.pi) - _math.pi


def _in_bounds(
    x: float,
    z: float,
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, z_min, z_max = bounds
    return x_min <= x <= x_max and z_min <= z <= z_max


def _simulate_distractor_action(
    *,
    x: float,
    z: float,
    yaw: float,
    action: str,
    move_step: float,
    turn_step: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, bool]:
    if action == "a":
        return (x, z, yaw + turn_step, False)
    if action == "d":
        return (x, z, yaw - turn_step, False)

    fwd_x = _math.cos(yaw) * move_step
    fwd_z = -_math.sin(yaw) * move_step
    if action == "w":
        nx, nz = x + fwd_x, z + fwd_z
    elif action == "s":
        nx, nz = x - fwd_x, z - fwd_z
    else:
        return (x, z, yaw, True)

    if _in_bounds(nx, nz, bounds):
        return (nx, nz, yaw, False)
    return (x, z, yaw, True)


def _select_distractor_action(
    *,
    x: float,
    z: float,
    yaw: float,
    target_x: float,
    target_z: float,
    bounds: tuple[float, float, float, float],
    move_step: float,
    turn_step: float,
) -> str:
    desired_yaw = _math.atan2(-(target_z - z), target_x - x)
    yaw_err = _normalize_angle(desired_yaw - yaw)
    align_eps = max(turn_step * 0.55, _math.radians(8.0))
    if abs(yaw_err) > align_eps:
        return "a" if yaw_err > 0 else "d"

    nx, nz, _nyaw, blocked = _simulate_distractor_action(
        x=x,
        z=z,
        yaw=yaw,
        action="w",
        move_step=move_step,
        turn_step=turn_step,
        bounds=bounds,
    )
    if not blocked:
        return "w"

    left_err = abs(_normalize_angle(desired_yaw - (yaw + turn_step)))
    right_err = abs(_normalize_angle(desired_yaw - (yaw - turn_step)))
    return "a" if left_err <= right_err else "d"


def _distractor_placeholders(distractors: int) -> list[_SceneObjectSpec]:
    placeholders: list[_SceneObjectSpec] = []
    for index, (x, z, yaw) in enumerate(_DISTRACTOR_SLOTS[:distractors], start=1):
        placeholders.append(
            _SceneObjectSpec(
                object_id=f"distractor_{index:02d}",
                kind="distractor_robot",
                position=(x, z),
                metadata={
                    "yaw": yaw,
                    "reflect_in_mirror": True,
                    "slot_index": index,
                },
            )
        )
    return placeholders


def _resolve_scene_template(runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del runtime_options
    _validate_distractor_slots()
    base_scene = _build_mirror_room_scene(mirror_enabled=True)
    return _replace(
        base_scene,
        scene_id="mirror_room_robot_distractors",
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
    cube_color_names = _random.sample(_AGENT_COLOR_NAMES, 3)
    agent_color = _random.choice(cube_color_names)
    slots = _random.sample(_MIRROR_ROOM_CUBE_CORNER_SLOTS, 3)
    if use_reflected_room:
        slots = [_reflect_pos_x(slot, mirror_plane_x) for slot in slots]
    cube_positions = {
        cube_color_names[0]: slots[0],
        cube_color_names[1]: slots[1],
        cube_color_names[2]: slots[2],
    }
    spawn_pos = scene_template.spawn_pos
    spawn_yaw = _sample_cardinal_yaw()
    if use_reflected_room:
        spawn_pos = _reflect_pos_x(spawn_pos, mirror_plane_x)
    mirror_window_z = _randomized_mirror_window_z(
        scene_template,
        center_jitter=0.6,
    )
    mirror_window_z = _mirror_window_covering_positions(
        scene_template,
        positions=tuple(cube_positions.values()),
        base_window_z=mirror_window_z,
    )
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
    excluded_wall_colors = [agent_color, *cube_color_names, *distractor_colors]
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
        objects=_build_cube_objects(cube_positions) + distractor_objects,
    )
    return _RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=None,
        cube_positions=cube_positions,
    )


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
    return "\n    <!-- Colored cubes -->\n    " + "\n    ".join(cube_bodies)


def _build_mujoco_scene(
    scene_spec: _SceneSpec,
    render_width: int,
    render_height: int,
    camera_height: float = 1.0,
) -> _MuJoCoSceneBuildResult:
    humanoid_scene = _build_humanoid_scene(
        mirror_present=True,
        spawn_pos=(scene_spec.spawn_pos[0], _MUJOCO_SPAWN_Y, scene_spec.spawn_pos[1]),
        camera_height=camera_height,
    )
    asset_xml = _inject_asset_entries(
        humanoid_scene.asset_xml,
        _surface_texture_assets_xml(),
    )
    extra_body_xml: list[str] = []
    distractor_geom_names: list[str] = []
    mirror_distractor_geom_names: list[str] = []
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
        )
        extra_body_xml.append(body_xml)
        distractor_geom_names.extend(geom_names)

        mirror_root_name = f"mirror_{obj.object_id}_root"
        mirror_body_xml, mirror_geom_names = _build_humanoid_wrapper(
            prefix=f"mirror_{obj.object_id}_",
            wrapper_name=mirror_root_name,
            root_joint_name=mirror_root_name,
            spawn_pos=(obj.position[0], _MUJOCO_SPAWN_Y, obj.position[1]),
            include_cameras=False,
        )
        extra_body_xml.append(mirror_body_xml)
        mirror_distractor_geom_names.extend(mirror_geom_names)

    rl = scene_spec.length
    rw = scene_spec.width
    rh = scene_spec.height
    wt = scene_spec.wall_thickness
    z_window = scene_spec.mirror_window_z or (1.0, 3.0)
    mg = z_window[0]
    mw = z_window[1] - z_window[0]
    cx = -(rl / 2 + wt / 2)
    left_len = mg - (-rw / 2)
    left_pz = (-rw / 2 + mg) / 2
    right_len = rw / 2 - (mg + mw)
    right_pz = (mg + mw + rw / 2) / 2
    mir_pz = mg + mw / 2

    xml = f"""<mujoco model="mirror_sim_e5">
  <compiler angle="degree"/>
  <option gravity="0 -9.81 0" timestep="0.01"/>
  {_two_room_visual_xml(render_width, render_height)}
  {asset_xml}
  {humanoid_scene.default_xml}

  <worldbody>
    {_two_room_lights_xml()}
    <geom name="floor_l" type="box"
          pos="{cx:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>
    <geom name="floor_r" type="box"
          pos="{rl/2+wt/2:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>
    <geom name="wall_back_l" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="wall_front_l" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="wall_outer_l" type="box"
          pos="{-rl-wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="wall_back_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="wall_front_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="wall_outer_r" type="box"
          pos="{rl+wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="ceiling_l" type="box"
          pos="{cx:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="ceiling_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="cwall_l" type="box"
          pos="0 {rh/2-0.95:.3f} {left_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {left_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <geom name="mirror_pane" type="box"
          pos="0 {rh/2-0.95:.3f} {mir_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {mw/2:.3f}"
          rgba=".85 .87 .90 .12" contype="0" conaffinity="0"/>
    <geom name="cwall_r" type="box"
          pos="0 {rh/2-0.95:.3f} {right_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {right_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    {_build_cube_xml(include_mirror_cubes=scene_spec.mirror_enabled)}

    {humanoid_scene.robot_body_xml}
    {humanoid_scene.mirror_body_xml}
    {' '.join(extra_body_xml)}
  </worldbody>
</mujoco>"""
    return _MuJoCoSceneBuildResult(
        xml=xml,
        robot_geom_names=humanoid_scene.robot_geom_names + distractor_geom_names,
        mirror_geom_names=humanoid_scene.mirror_geom_names + mirror_distractor_geom_names,
    )


def _apply_mujoco_reset(env, run_setup: _RunSetup) -> None:
    distractors = sorted(
        (obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"),
        key=lambda obj: obj.object_id,
    )
    active_ids = {obj.object_id for obj in distractors}
    mirror_plane_x = (
        float(run_setup.scene_spec.mirror_plane_x)
        if run_setup.scene_spec.mirror_plane_x is not None
        else -run_setup.scene_spec.wall_thickness / 2
    )
    nav_states: list[dict[str, object]] = []
    for obj in distractors:
        root_name = f"{obj.object_id}_root"
        mirror_root_name = f"mirror_{obj.object_id}_root"
        yaw = float(obj.metadata.get("yaw", 0.0))
        x, z = obj.position
        root_addr = env._jnt_addr_if_exists(root_name)
        if root_addr is not None:
            env._set_body_pose(root_addr, float(x), env._spawn_y, float(z), yaw)
        env._apply_arm_rest_pose_for_prefix(f"{obj.object_id}_")
        env._set_color(
            env._geom_ids_with_prefix(f"{obj.object_id}_"),
            _rgba_for_color(obj.color or "", alpha=1.0),
        )

        mirror_root_addr = env._jnt_addr_if_exists(mirror_root_name)
        if mirror_root_addr is not None:
            mirror_x = 2 * env._mirror_plane_x - float(x)
            if env._is_in_mirror_aperture(float(z)):
                env._set_body_pose(mirror_root_addr, mirror_x, env._spawn_y, float(z), -yaw)
            else:
                env._park_body(mirror_root_addr)
        env._apply_arm_rest_pose_for_prefix(f"mirror_{obj.object_id}_")
        env._set_color(
            env._geom_ids_with_prefix(f"mirror_{obj.object_id}_"),
            _rgba_for_color(obj.color or "", alpha=1.0),
        )
        bounds = _distractor_nav_bounds(
            run_setup.scene_spec,
            mirror_plane_x=mirror_plane_x,
            x=float(x),
        )
        target_x, target_z = _sample_distractor_target(
            bounds,
            current=(float(x), float(z)),
        )
        nav_states.append({
            "object_id": obj.object_id,
            "root_addr": root_addr,
            "mirror_root_addr": mirror_root_addr,
            "x": float(x),
            "z": float(z),
            "yaw": yaw,
            "turn_step": _math.radians(float(run_setup.scene_spec.turn_step_deg)),
            "target_x": target_x,
            "target_z": target_z,
            "bounds": bounds,
            "last_action": "none",
            "stuck_steps": 0,
            "turn_switches": 0,
        })
    for slot_index in range(1, _MAX_DISTRACTORS_PER_RUN + 1):
        object_id = f"distractor_{slot_index:02d}"
        if object_id in active_ids:
            continue
        root_addr = env._jnt_addr_if_exists(f"{object_id}_root")
        mirror_root_addr = env._jnt_addr_if_exists(f"mirror_{object_id}_root")
        if root_addr is not None:
            env._park_body(root_addr)
        if mirror_root_addr is not None:
            env._park_body(mirror_root_addr)
    env._e5_distractor_nav = nav_states


def _apply_mujoco_step(env, _action: str) -> None:
    states = getattr(env, "_e5_distractor_nav", None)
    if not states:
        return
    mirror_plane_x = getattr(env, "_mirror_plane_x", None)
    for state in states:
        prev_x = float(state["x"])
        prev_z = float(state["z"])
        x = prev_x
        z = prev_z
        yaw = float(state["yaw"])
        target_x = float(state["target_x"])
        target_z = float(state["target_z"])
        bounds = state["bounds"]
        turn_step = float(state.get("turn_step", _math.radians(30.0)))
        prev_action = str(state.get("last_action", "none"))

        dx = target_x - x
        dz = target_z - z
        dist = _math.hypot(dx, dz)
        if dist <= _DISTRACTOR_TARGET_REACHED_EPS:
            target_x, target_z = _sample_distractor_target(
                bounds,
                current=(x, z),
            )
            state["target_x"] = target_x
            state["target_z"] = target_z
            dx = target_x - x
            dz = target_z - z
            dist = _math.hypot(dx, dz)

        action = _select_distractor_action(
            x=x,
            z=z,
            yaw=yaw,
            target_x=target_x,
            target_z=target_z,
            bounds=bounds,
            move_step=_DISTRACTOR_ACTION_MOVE_STEP,
            turn_step=turn_step,
        )
        x, z, yaw, _blocked = _simulate_distractor_action(
            x=x,
            z=z,
            yaw=yaw,
            action=action,
            move_step=_DISTRACTOR_ACTION_MOVE_STEP,
            turn_step=turn_step,
            bounds=bounds,
        )

        state["x"] = round(x, 3)
        state["z"] = round(z, 3)
        state["yaw"] = yaw
        state["last_action"] = action

        new_dist = _math.hypot(target_x - x, target_z - z)
        moved = _math.hypot(float(state["x"]) - prev_x, float(state["z"]) - prev_z)
        no_progress = new_dist >= (dist - _DISTRACTOR_PROGRESS_EPS)
        turn_switched = (
            action in {"a", "d"}
            and prev_action in {"a", "d"}
            and action != prev_action
        )
        stuck_steps = int(state.get("stuck_steps", 0))
        turn_switches = int(state.get("turn_switches", 0))
        if _blocked or moved < 1e-3 or no_progress:
            stuck_steps += 1
        else:
            stuck_steps = 0
        if turn_switched:
            turn_switches += 1
        else:
            turn_switches = 0
        state["stuck_steps"] = stuck_steps
        state["turn_switches"] = turn_switches

        should_resample = (
            new_dist <= _DISTRACTOR_TARGET_REACHED_EPS
            or stuck_steps >= _DISTRACTOR_STUCK_RESAMPLE_STEPS
            or turn_switches >= _DISTRACTOR_TURN_SWITCH_RESAMPLE
        )
        if should_resample:
            new_target_x, new_target_z = _sample_distractor_target(
                bounds,
                current=(x, z),
            )
            state["target_x"] = new_target_x
            state["target_z"] = new_target_z
            state["stuck_steps"] = 0
            state["turn_switches"] = 0

        root_addr = state.get("root_addr")
        if root_addr is not None:
            env._set_body_pose(root_addr, x, env._spawn_y, z, yaw)
        mirror_root_addr = state.get("mirror_root_addr")
        if mirror_root_addr is not None and mirror_plane_x is not None:
            mirror_x = 2 * float(mirror_plane_x) - x
            if env._is_in_mirror_aperture(z):
                env._set_body_pose(mirror_root_addr, mirror_x, env._spawn_y, z, -yaw)
            else:
                env._park_body(mirror_root_addr)


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
        )

    return _BackendOverrides(
        mujoco_scene_builder=_scene_builder,
        mujoco_reset_applier=_apply_mujoco_reset,
        mujoco_step_applier=_apply_mujoco_step,
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="e5",
    scene_id="mirror_room_robot_distractors",
    task_id="cube_task",
    num_runs_default=10,
    default_runtime_options=_ExperimentRuntimeOptions(),
    scene_template=_resolve_scene_template(_ExperimentRuntimeOptions()),
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_initialize_run,
    backend_overrides_factory=_resolve_backend_overrides,
)
