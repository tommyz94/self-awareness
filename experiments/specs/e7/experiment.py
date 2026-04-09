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
    _AGENT_COLOR_NAMES as _AGENT_COLOR_NAMES,
    build_mirror_room_scene as _build_mirror_room_scene,
    mirror_window_covering_positions as _mirror_window_covering_positions,
    sample_non_agent_wall_rgba as _sample_non_agent_wall_rgba,
)
from experiments.specs.e5.experiment import (
    _MAX_DISTRACTORS_PER_RUN as _MAX_DISTRACTORS_PER_RUN,
    _MIN_DISTRACTORS_PER_RUN as _MIN_DISTRACTORS_PER_RUN,
    _apply_mujoco_reset as _apply_mujoco_reset_distractors,
    _apply_mujoco_step as _apply_mujoco_step,
    _distractor_placeholders as _distractor_placeholders,
    _randomized_mirror_window_z as _randomized_mirror_window_z,
    _reflect_pos_x as _reflect_pos_x,
    _resolve_mirror_plane_x as _resolve_mirror_plane_x,
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
from utils.colors import rgba_for_color as _rgba_for_color

_ASSETS_DIR = _Path(__file__).resolve().parents[3] / "assets" / "mujoco"
_CUBE_XML_PATH = _ASSETS_DIR / "cube.xml"
_MUJOCO_SPAWN_Y = -1.0
_CUBE_Y = -0.75
_CUBE_HALF_EXTENT_Y = 0.25
_CUBE_STACK_STEP_Y = 2.0 * _CUBE_HALF_EXTENT_Y

_MIN_CUBES_PER_RUN = 5
_MAX_CUBES_PER_RUN = 15
_MIN_CUBE_STACK_HEIGHT = 1
_MAX_CUBE_STACK_HEIGHT = 5
_MAX_CUBE_SLOTS_PER_RUN = _MAX_CUBES_PER_RUN * _MAX_CUBE_STACK_HEIGHT
_CUBE_MARGIN_X = 0.9
_CUBE_MARGIN_Z = 0.8
_CUBE_MIN_SEPARATION = 1.05
_CUBE_MIN_DIST_FROM_AGENT = 1.6
_CUBE_MIN_DIST_FROM_DISTRACTOR = 1.2

# Keep the first three legacy slot names so generic MuJoCo assumptions remain compatible.
_LEGACY_SLOT_NAMES = {
    1: "red",
    2: "green",
    3: "blue",
}
_BASE_SLOT_PLACEHOLDER_POSITIONS = [
    (-12.5, -5.0),
    (-12.5, -1.0),
    (-12.5, 3.0),
    (-10.5, -5.0),
    (-10.5, -1.0),
    (-10.5, 3.0),
    (-8.5, -5.0),
    (-8.5, -1.0),
    (-8.5, 3.0),
    (2.5, -5.0),
    (2.5, -1.0),
    (2.5, 3.0),
    (4.5, -5.0),
    (4.5, -1.0),
    (4.5, 3.0),
]


def _build_slot_placeholder_positions(slot_count: int) -> list[tuple[float, float]]:
    if slot_count <= 0:
        return []
    positions: list[tuple[float, float]] = []
    layer = 0
    while len(positions) < slot_count:
        z_offset = layer * 0.04
        for x, z in _BASE_SLOT_PLACEHOLDER_POSITIONS:
            positions.append((x, round(z + z_offset, 3)))
            if len(positions) == slot_count:
                break
        layer += 1
    return positions


_SLOT_PLACEHOLDER_POSITIONS = _build_slot_placeholder_positions(_MAX_CUBE_SLOTS_PER_RUN)


def _slot_name(index: int) -> str:
    if index in _LEGACY_SLOT_NAMES:
        return _LEGACY_SLOT_NAMES[index]
    return f"slot_{index:02d}"


def _cube_body_name(index: int, *, mirror: bool = False) -> str:
    base = f"cube_{_slot_name(index)}"
    if mirror:
        return f"mirror_{base}"
    return base


def _cube_joint_name(index: int, *, mirror: bool = False) -> str:
    base = f"cube_{_slot_name(index)}_j"
    if mirror:
        return f"mirror_{base}"
    return base


def _cube_geom_name(index: int, *, mirror: bool = False) -> str:
    base = f"cube_{_slot_name(index)}_g"
    if mirror:
        return f"mirror_{base}"
    return base


def _load_cube_template() -> str:
    if not _CUBE_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo cube asset not found at {_CUBE_XML_PATH}")
    template = _CUBE_XML_PATH.read_text().strip()
    if not template:
        raise ValueError(f"MuJoCo cube asset at {_CUBE_XML_PATH} is empty")
    return template


def _build_cube_xml(*, include_mirror_cubes: bool) -> str:
    template = _load_cube_template()
    cube_bodies: list[str] = []
    for index, (x, z) in enumerate(_SLOT_PLACEHOLDER_POSITIONS, start=1):
        cube_bodies.append(
            template.format(
                body_name=_cube_body_name(index, mirror=False),
                joint_name=_cube_joint_name(index, mirror=False),
                geom_name=_cube_geom_name(index, mirror=False),
                x=x,
                z=z,
                rgba="0.55 0.55 0.55 1",
            )
        )
    if include_mirror_cubes:
        for index, (x, z) in enumerate(_SLOT_PLACEHOLDER_POSITIONS, start=1):
            cube_bodies.append(
                template.format(
                    body_name=_cube_body_name(index, mirror=True),
                    joint_name=_cube_joint_name(index, mirror=True),
                    geom_name=_cube_geom_name(index, mirror=True),
                    x=x,
                    z=z,
                    rgba="0.55 0.55 0.55 1",
                )
            )
    return "\n    <!-- e7 distractor cubes -->\n    " + "\n    ".join(cube_bodies)


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

    xml = f"""<mujoco model=\"mirror_sim_e7\">
  <compiler angle=\"degree\"/>
  <option gravity=\"0 -9.81 0\" timestep=\"0.01\"/>
  {_two_room_visual_xml(render_width, render_height)}
  {asset_xml}
  {humanoid_scene.default_xml}

  <worldbody>
    {_two_room_lights_xml()}
    <geom name=\"floor_l\" type=\"box\"
          pos=\"{cx:.3f} -1.05 0\"
          size=\"{rl/2:.3f} .05 {rw/2:.3f}\"
          material=\"floor_wood_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"floor_r\" type=\"box\"
          pos=\"{rl/2+wt/2:.3f} -1.05 0\"
          size=\"{rl/2:.3f} .05 {rw/2:.3f}\"
          material=\"floor_wood_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_back_l\" type=\"box\"
          pos=\"{cx:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}\"
          size=\"{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_front_l\" type=\"box\"
          pos=\"{cx:.3f} {rh/2-0.95:.3f} {rw/2:.3f}\"
          size=\"{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_outer_l\" type=\"box\"
          pos=\"{-rl-wt:.3f} {rh/2-0.95:.3f} 0\"
          size=\"{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_back_r\" type=\"box\"
          pos=\"{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}\"
          size=\"{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_front_r\" type=\"box\"
          pos=\"{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {rw/2:.3f}\"
          size=\"{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"wall_outer_r\" type=\"box\"
          pos=\"{rl+wt:.3f} {rh/2-0.95:.3f} 0\"
          size=\"{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"ceiling_l\" type=\"box\"
          pos=\"{cx:.3f} {rh-0.95:.3f} 0\"
          size=\"{rl/2:.3f} .05 {rw/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"ceiling_r\" type=\"box\"
          pos=\"{rl/2+wt/2:.3f} {rh-0.95:.3f} 0\"
          size=\"{rl/2:.3f} .05 {rw/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"cwall_l\" type=\"box\"
          pos=\"0 {rh/2-0.95:.3f} {left_pz:.3f}\"
          size=\"{wt/2:.3f} {rh/2:.3f} {left_len/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"mirror_pane\" type=\"box\"
          pos=\"0 {rh/2-0.95:.3f} {mir_pz:.3f}\"
          size=\"{wt/2:.3f} {rh/2:.3f} {mw/2:.3f}\"
          rgba=\".85 .87 .90 .12\" contype=\"0\" conaffinity=\"0\"/>
    <geom name=\"cwall_r\" type=\"box\"
          pos=\"0 {rh/2-0.95:.3f} {right_pz:.3f}\"
          size=\"{wt/2:.3f} {rh/2:.3f} {right_len/2:.3f}\"
          material=\"wall_plaster_mat\" contype=\"0\" conaffinity=\"0\"/>
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


def _scene_side_bounds(
    scene_spec: _SceneSpec,
    *,
    mirror_plane_x: float,
    side: str,
) -> tuple[float, float, float, float]:
    z_min = -scene_spec.width / 2 + _CUBE_MARGIN_Z
    z_max = scene_spec.width / 2 - _CUBE_MARGIN_Z
    left_x_min = -scene_spec.length - scene_spec.wall_thickness + _CUBE_MARGIN_X
    left_x_max = mirror_plane_x - _CUBE_MARGIN_X
    right_x_min = mirror_plane_x + _CUBE_MARGIN_X
    right_x_max = scene_spec.length + scene_spec.wall_thickness - _CUBE_MARGIN_X
    if side == "left":
        if left_x_max <= left_x_min:
            left_x_max = left_x_min + 0.1
        return (left_x_min, left_x_max, z_min, z_max)
    if right_x_max <= right_x_min:
        right_x_max = right_x_min + 0.1
    return (right_x_min, right_x_max, z_min, z_max)


def _cube_candidate_valid(
    candidate: tuple[float, float],
    *,
    sampled: list[tuple[float, float]],
    spawn_pos: tuple[float, float],
    distractor_positions: list[tuple[float, float]],
) -> bool:
    px, pz = candidate
    sx, sz = spawn_pos
    if _math.hypot(px - sx, pz - sz) < _CUBE_MIN_DIST_FROM_AGENT:
        return False
    if any(_math.hypot(px - dx, pz - dz) < _CUBE_MIN_DIST_FROM_DISTRACTOR for dx, dz in distractor_positions):
        return False
    if any(_math.hypot(px - ox, pz - oz) < _CUBE_MIN_SEPARATION for ox, oz in sampled):
        return False
    return True


def _sample_cube_positions(
    scene_spec: _SceneSpec,
    *,
    mirror_plane_x: float,
    count: int,
    spawn_pos: tuple[float, float],
    distractor_positions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if count <= 0:
        return []

    sides = ["left", "right"] + [_random.choice(("left", "right")) for _ in range(max(0, count - 2))]
    sides = sides[:count]
    _random.shuffle(sides)

    sampled: list[tuple[float, float]] = []
    unresolved_sides: list[str] = []

    for side in sides:
        x_min, x_max, z_min, z_max = _scene_side_bounds(
            scene_spec,
            mirror_plane_x=mirror_plane_x,
            side=side,
        )
        found = False
        for _ in range(96):
            px = round(_random.uniform(x_min, x_max), 3)
            pz = round(_random.uniform(z_min, z_max), 3)
            if _cube_candidate_valid(
                (px, pz),
                sampled=sampled,
                spawn_pos=spawn_pos,
                distractor_positions=distractor_positions,
            ):
                sampled.append((px, pz))
                found = True
                break
        if not found:
            unresolved_sides.append(side)

    if unresolved_sides:
        fallback_candidates: list[tuple[float, str, float, float]] = []
        for side in ("left", "right"):
            x_min, x_max, z_min, z_max = _scene_side_bounds(
                scene_spec,
                mirror_plane_x=mirror_plane_x,
                side=side,
            )
            x = x_min
            while x <= x_max + 1e-6:
                z = z_min
                while z <= z_max + 1e-6:
                    fallback_candidates.append((_random.random(), side, round(x, 3), round(z, 3)))
                    z += 1.2
                x += 1.2
        fallback_candidates.sort(key=lambda item: item[0])

        for side in list(unresolved_sides):
            for _weight, candidate_side, px, pz in fallback_candidates:
                if candidate_side != side:
                    continue
                if _cube_candidate_valid(
                    (px, pz),
                    sampled=sampled,
                    spawn_pos=spawn_pos,
                    distractor_positions=distractor_positions,
                ):
                    sampled.append((px, pz))
                    unresolved_sides.remove(side)
                    break

    while len(sampled) < count:
        side = _random.choice(("left", "right"))
        x_min, x_max, z_min, z_max = _scene_side_bounds(
            scene_spec,
            mirror_plane_x=mirror_plane_x,
            side=side,
        )
        px = round(_random.uniform(x_min, x_max), 3)
        pz = round(_random.uniform(z_min, z_max), 3)
        if _cube_candidate_valid(
            (px, pz),
            sampled=sampled,
            spawn_pos=spawn_pos,
            distractor_positions=distractor_positions,
        ):
            sampled.append((px, pz))
            continue
        sampled.append((px, pz))

    return sampled[:count]


def _resolve_scene_template(runtime_options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del runtime_options
    _validate_distractor_slots()
    base_scene = _build_mirror_room_scene(mirror_enabled=True)
    return _replace(
        base_scene,
        scene_id="mirror_room_robot_distractors_with_random_cubes",
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
    distractor_colors = [_random.choice(_AGENT_COLOR_NAMES) for _ in selected_placeholders]
    if len(distractor_colors) > 1 and len(set(distractor_colors)) == 1:
        pivot = _random.randrange(len(distractor_colors))
        alternatives = [color for color in _AGENT_COLOR_NAMES if color != distractor_colors[pivot]]
        distractor_colors[pivot] = _random.choice(alternatives)
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

    cube_count = _random.randint(_MIN_CUBES_PER_RUN, _MAX_CUBES_PER_RUN)
    cube_positions = _sample_cube_positions(
        scene_template,
        mirror_plane_x=mirror_plane_x,
        count=cube_count,
        spawn_pos=spawn_pos,
        distractor_positions=distractor_positions,
    )
    mirror_window_z = _mirror_window_covering_positions(
        scene_template,
        positions=tuple(cube_positions),
        base_window_z=mirror_window_z,
    )
    cube_colors = [_random.choice(_AGENT_COLOR_NAMES) for _ in range(cube_count)]
    use_random_cube_heights = bool(runtime_options.e7_random_cube_heights)
    cube_objects: list[_SceneObjectSpec] = []
    slot_index = 1
    for index in range(cube_count):
        stack_height = (
            _random.randint(_MIN_CUBE_STACK_HEIGHT, _MAX_CUBE_STACK_HEIGHT)
            if use_random_cube_heights
            else 1
        )
        stack_base_id = f"cube_{index + 1:02d}"
        x, z = cube_positions[index]
        color = cube_colors[index]
        for stack_level in range(1, stack_height + 1):
            cube_objects.append(
                _SceneObjectSpec(
                    object_id=f"{stack_base_id}_h{stack_level:02d}",
                    kind="cube",
                    position=(x, z),
                    color=color,
                    metadata={
                        "reflect_in_mirror": True,
                        "slot_index": slot_index,
                        "stack_base_id": stack_base_id,
                        "stack_level": stack_level,
                        "stack_height": stack_height,
                    },
                )
            )
            slot_index += 1
    if len(cube_objects) > _MAX_CUBE_SLOTS_PER_RUN:
        cube_objects = cube_objects[:_MAX_CUBE_SLOTS_PER_RUN]

    excluded_wall_colors = [agent_color, *distractor_colors, *cube_colors]
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
        objects=distractor_objects + cube_objects,
    )
    return _RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=None,
    )


def _geom_id_if_exists(env, name: str):
    geom_id = env._mj.mj_name2id(env._model, env._mj.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        return None
    return int(geom_id)


def _apply_cubes_reset(env, run_setup: _RunSetup) -> None:
    cubes = [
        obj
        for obj in run_setup.scene_spec.objects
        if obj.kind == "cube"
    ]
    cubes.sort(key=lambda obj: int(obj.metadata.get("slot_index", 10_000)))
    cubes = cubes[:_MAX_CUBE_SLOTS_PER_RUN]
    env._object_pos_2d = {}

    for slot_index in range(1, _MAX_CUBE_SLOTS_PER_RUN + 1):
        cube = cubes[slot_index - 1] if slot_index <= len(cubes) else None
        joint_name = _cube_joint_name(slot_index, mirror=False)
        geom_name = _cube_geom_name(slot_index, mirror=False)
        addr = env._jnt_addr_if_exists(joint_name)
        geom_id = _geom_id_if_exists(env, geom_name)

        if cube is not None and addr is not None:
            x, z = cube.position
            stack_level = max(
                _MIN_CUBE_STACK_HEIGHT,
                min(_MAX_CUBE_STACK_HEIGHT, int(cube.metadata.get("stack_level", 1))),
            )
            cube_y = _CUBE_Y + (stack_level - 1) * _CUBE_STACK_STEP_Y
            env._set_body_pos(addr, float(x), float(cube_y), float(z))
            env._object_pos_2d[str(cube.object_id)] = (float(x), float(z))
            if geom_id is not None:
                env._set_color([geom_id], _rgba_for_color(cube.color or "", alpha=1.0))
        else:
            if addr is not None:
                env._set_body_pos(addr, 0.0, -100.0, 0.0)
            if geom_id is not None:
                env._set_color([geom_id], _rgba_for_color("gray", alpha=1.0))

    mirror_plane_x = getattr(env, "_mirror_plane_x", None)
    for slot_index in range(1, _MAX_CUBE_SLOTS_PER_RUN + 1):
        cube = cubes[slot_index - 1] if slot_index <= len(cubes) else None
        mirror_joint_name = _cube_joint_name(slot_index, mirror=True)
        mirror_geom_name = _cube_geom_name(slot_index, mirror=True)
        mirror_addr = env._jnt_addr_if_exists(mirror_joint_name)
        mirror_geom_id = _geom_id_if_exists(env, mirror_geom_name)

        should_reflect = (
            cube is not None
            and mirror_addr is not None
            and mirror_plane_x is not None
            and bool(cube.metadata.get("reflect_in_mirror", False))
            and env._is_in_mirror_aperture(float(cube.position[1]))
        )
        if should_reflect:
            x, z = cube.position
            stack_level = max(
                _MIN_CUBE_STACK_HEIGHT,
                min(_MAX_CUBE_STACK_HEIGHT, int(cube.metadata.get("stack_level", 1))),
            )
            cube_y = _CUBE_Y + (stack_level - 1) * _CUBE_STACK_STEP_Y
            mirror_x = 2 * float(mirror_plane_x) - float(x)
            env._set_body_pos(mirror_addr, mirror_x, float(cube_y), float(z))
            if mirror_geom_id is not None:
                env._set_color([mirror_geom_id], _rgba_for_color(cube.color or "", alpha=1.0))
        else:
            if mirror_addr is not None:
                env._park_body(mirror_addr)
            if mirror_geom_id is not None:
                env._set_color([mirror_geom_id], _rgba_for_color("gray", alpha=1.0))


def _apply_mujoco_reset(env, run_setup: _RunSetup) -> None:
    _apply_mujoco_reset_distractors(env, run_setup)
    _apply_cubes_reset(env, run_setup)


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


_E7_DEFAULT_RUNTIME_OPTIONS = _ExperimentRuntimeOptions(e7_random_cube_heights=True)


EXPERIMENT = _ExperimentSpec(
    experiment_id="e7",
    scene_id="mirror_room_robot_distractors_with_random_cubes",
    task_id="explore_task",
    num_runs_default=10,
    default_runtime_options=_E7_DEFAULT_RUNTIME_OPTIONS,
    scene_template=_resolve_scene_template(_E7_DEFAULT_RUNTIME_OPTIONS),
    scene_template_factory=_resolve_scene_template,
    run_setup_factory=_initialize_run,
    backend_overrides_factory=_resolve_backend_overrides,
)
