from __future__ import annotations

import colorsys
import math
import os
import random
from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

from env.base import CUBE_SLOTS
from experiments.shared.types import RunSetup, SceneObjectSpec, SceneSpec
from utils.colors import COLOR_POOL as _COLOR_POOL, COLOR_RGB as _COLOR_RGB

_AGENT_COLOR_NAMES = list(_COLOR_POOL)
_AGENT_CUBE_RGB = dict(_COLOR_RGB)
MIRROR_ROOM_CUBE_CORNER_SLOTS = [
    (-13.0, -6.0),
    (-13.0, 6.0),
    (-2.0, -6.0),
    (-2.0, 6.0),
]
MIRROR_ROOM_WALL_CORNER_SLOTS = [
    (-14.85, -7.15),
    (-14.85, 7.15),
    (-0.35, -7.15),
    (-0.35, 7.15),
]


def _resolve_mirror_plane_x(scene_spec: SceneSpec) -> float:
    if scene_spec.mirror_plane_x is not None:
        return float(scene_spec.mirror_plane_x)
    return -scene_spec.wall_thickness / 2


def _reflect_pos_x(position: Tuple[float, float], mirror_plane_x: float) -> Tuple[float, float]:
    x, z = position
    return (round(2 * mirror_plane_x - float(x), 3), float(z))


def _sample_cardinal_yaw() -> float:
    return random.choice((0.0, math.pi / 2, math.pi, 3 * math.pi / 2))


def _rgb_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def sample_non_agent_wall_rgba(
    *,
    excluded_color_names: Sequence[str] = _AGENT_COLOR_NAMES,
) -> tuple[float, float, float, float]:
    excluded_rgb = [
        _AGENT_CUBE_RGB[name]
        for name in excluded_color_names
        if name in _AGENT_CUBE_RGB
    ]
    for _ in range(64):
        hue = random.random()
        saturation = random.uniform(0.08, 0.35)
        value = random.uniform(0.65, 0.95)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        if not excluded_rgb or min(_rgb_distance(rgb, c) for c in excluded_rgb) >= 0.55:
            return (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3), 1.0)
    return (0.78, 0.78, 0.78, 1.0)


def _randomized_mirror_window_z(
    scene_spec: SceneSpec,
    *,
    center_jitter: Optional[float],
) -> Optional[Tuple[float, float]]:
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
        new_center = center + random.uniform(shift_min, shift_max)

    new_min = round(new_center - half_width, 3)
    new_max = round(new_center + half_width, 3)
    return (new_min, new_max)


def mirror_window_covering_positions(
    scene_spec: SceneSpec,
    *,
    positions: Sequence[Tuple[float, float]],
    base_window_z: Optional[Tuple[float, float]],
    padding: float = 0.35,
) -> Optional[Tuple[float, float]]:
    if (
        not scene_spec.mirror_enabled
        or scene_spec.mirror_aperture_mode != "window"
    ):
        return base_window_z

    if base_window_z is None:
        return None

    z_values = [float(z) for _x, z in positions]
    if not z_values:
        return base_window_z

    room_min = -scene_spec.width / 2 + 0.05
    room_max = scene_spec.width / 2 - 0.05
    required_min = max(room_min, min(z_values) - float(padding))
    required_max = min(room_max, max(z_values) + float(padding))
    if required_max < required_min:
        return base_window_z

    base_min, base_max = base_window_z
    merged_min = max(room_min, min(float(base_min), required_min))
    merged_max = min(room_max, max(float(base_max), required_max))
    if merged_max <= merged_min:
        return base_window_z
    return (round(merged_min, 3), round(merged_max, 3))


def build_mirror_room_scene(*, mirror_enabled: bool) -> SceneSpec:
    return SceneSpec(
        scene_id="mirror_room" if mirror_enabled else "mirror_room_no_mirror",
        mirror_enabled=mirror_enabled,
        layout_kind="two_room",
        length=15.0,
        width=15.0,
        height=7.0,
        wall_thickness=0.2,
        spawn_pos=(-10.0, 2.5),
        spawn_yaw=0.0,
        mirror_plane_x=-0.1,
        mirror_aperture_mode="window" if mirror_enabled else "none",
        mirror_window_z=(1.0, 3.0) if mirror_enabled else None,
    )


def build_corridor_scene(*, use_mirror: bool = True) -> SceneSpec:
    return SceneSpec(
        scene_id="narrow_corridor_mirror",
        mirror_enabled=use_mirror,
        layout_kind="corridor",
        length=10.0,
        width=6.0,
        height=7.0,
        wall_thickness=0.2,
        spawn_pos=(-4.5, 0.0),
        spawn_yaw=0.0,
        mirror_plane_x=5.0,
        mirror_aperture_mode="full" if use_mirror else "none",
        mirror_window_z=None,
    )


def build_cube_objects(
    cube_positions: Dict[str, Tuple[float, float]],
    *,
    mirror_cubes_ignore_aperture: bool = False,
) -> list[SceneObjectSpec]:
    return [
        SceneObjectSpec(
            object_id=f"cube_{color_name}",
            kind="cube",
            color=color_name,
            position=position,
            metadata={
                "reflect_in_mirror": True,
                "reflect_ignore_aperture": bool(mirror_cubes_ignore_aperture),
            },
        )
        for color_name, position in cube_positions.items()
    ]


def build_cube_run_setup(
    scene_template: SceneSpec,
    *,
    wrong_color: Optional[str] = None,
    cube_slots: Optional[Sequence[Tuple[float, float]]] = None,
    randomize_wall: bool = False,
    randomize_wall_color: bool = False,
    mirror_window_center_jitter: Optional[float] = None,
    mirror_window_cover_cubes: bool = True,
    mirror_cubes_ignore_aperture: bool = False,
    randomize_spawn_yaw: bool = False,
) -> RunSetup:
    available_slots = list(cube_slots) if cube_slots is not None else CUBE_SLOTS
    if len(available_slots) < 3:
        raise ValueError("Cube slot set must contain at least 3 positions.")

    cube_color_names = random.sample(_AGENT_COLOR_NAMES, 3)
    agent_color = random.choice(cube_color_names)
    if wrong_color == "__random__":
        wrong_color = random.choice([c for c in _AGENT_COLOR_NAMES if c != agent_color])

    slots = random.sample(available_slots, 3)
    mirror_plane_x = _resolve_mirror_plane_x(scene_template)
    use_reflected_room = randomize_wall and random.choice([False, True])
    if use_reflected_room:
        slots = [_reflect_pos_x(slot, mirror_plane_x) for slot in slots]

    cube_positions: Dict[str, Tuple[float, float]] = {
        cube_color_names[0]: slots[0],
        cube_color_names[1]: slots[1],
        cube_color_names[2]: slots[2],
    }

    msr_level = os.getenv("MSR_LEVEL", "2")
    
    # Override reflection logic for easy levels to prevent backwards spawning
    if msr_level in ["0", "1"]:
        use_reflected_room = False

    if msr_level == "0":
        spawn_pos = (-1.2, 2.0)
        spawn_yaw = 0.0
    elif msr_level == "1":
        spawn_pos = scene_template.spawn_pos
        spawn_yaw = 0.0
    else:
        spawn_pos = scene_template.spawn_pos
        spawn_yaw = _sample_cardinal_yaw() if randomize_spawn_yaw else scene_template.spawn_yaw

    if use_reflected_room:
        spawn_pos = _reflect_pos_x(spawn_pos, mirror_plane_x)

    mirror_window_z = _randomized_mirror_window_z(
        scene_template,
        center_jitter=mirror_window_center_jitter,
    )
    if mirror_window_cover_cubes:
        mirror_window_z = mirror_window_covering_positions(
            scene_template,
            positions=tuple(cube_positions.values()),
            base_window_z=mirror_window_z,
        )

    if msr_level == "0" and mirror_window_z is not None:
        # The mirror window shifts to cover cubes, so we must align our Z pos to its final center
        center_z = (mirror_window_z[0] + mirror_window_z[1]) / 2.0
        final_x = -1.2
        if use_reflected_room:
            final_x = 2 * mirror_plane_x - (-1.2)
        spawn_pos = (final_x, round(center_z, 3))

    excluded_wall_colors = [agent_color, *cube_color_names]
    wall_color_rgba = (
        sample_non_agent_wall_rgba(excluded_color_names=excluded_wall_colors)
        if randomize_wall_color else scene_template.wall_color_rgba
    )
    scene_spec = replace(
        scene_template,
        spawn_pos=spawn_pos,
        spawn_yaw=spawn_yaw,
        mirror_window_z=mirror_window_z,
        wall_color_rgba=wall_color_rgba,
        objects=build_cube_objects(
            cube_positions,
            mirror_cubes_ignore_aperture=mirror_cubes_ignore_aperture,
        ),
    )
    return RunSetup(
        agent_color_name=agent_color,
        scene_spec=scene_spec,
        wrong_color=wrong_color,
        cube_positions=cube_positions,
    )


def build_corridor_run_setup(scene_template: SceneSpec) -> RunSetup:
    return RunSetup(
        agent_color_name="red",
        scene_spec=replace(scene_template, objects=[]),
        wrong_color=None,
        cube_positions={},
    )
