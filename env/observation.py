from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from experiments.specs.types import SceneSpec


@dataclass(frozen=True)
class ObservationMetadata:
    view_position: tuple[float, float, float]
    view_direction: tuple[float, float, float]
    mirror_visible: bool
    ego_reflection_visible: Optional[bool] = None


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return tuple(component / norm for component in vector)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mirror_samples(scene_spec: SceneSpec) -> Iterable[tuple[float, float, float]]:
    if not scene_spec.mirror_enabled or scene_spec.mirror_plane_x is None:
        return []

    if scene_spec.layout_kind == "corridor":
        z_min = -scene_spec.width / 2
        z_max = scene_spec.width / 2
    elif scene_spec.mirror_window_z is not None:
        z_min, z_max = scene_spec.mirror_window_z
    else:
        return []

    y_center = scene_spec.height / 2 - 0.95
    y_min = y_center - scene_spec.height / 2
    y_max = y_center + scene_spec.height / 2
    z_center = (z_min + z_max) / 2
    x = scene_spec.mirror_plane_x

    return (
        (x, y_center, z_center),
        (x, y_min, z_min),
        (x, y_min, z_max),
        (x, y_max, z_min),
        (x, y_max, z_max),
        (x, y_center, z_min),
        (x, y_center, z_max),
    )


def mirror_visible_from_camera(
    *,
    scene_spec: SceneSpec,
    camera_position: tuple[float, float, float],
    camera_forward: tuple[float, float, float],
    vertical_fov_deg: float,
    aspect_ratio: float,
) -> bool:
    if not scene_spec.mirror_enabled or scene_spec.mirror_plane_x is None:
        return False

    # In the two-room scene, treat the active mirror opening as the visible
    # segment so off-window poses do not count as a mirror view.
    if (
        scene_spec.layout_kind != "corridor"
        and scene_spec.mirror_aperture_mode == "window"
        and scene_spec.mirror_window_z is not None
    ):
        z_min, z_max = scene_spec.mirror_window_z
        if not z_min <= camera_position[2] <= z_max:
            return False

    forward = _normalize(camera_forward)
    if forward == (0.0, 0.0, 0.0):
        return False

    world_up = (0.0, 1.0, 0.0)
    right = _normalize(_cross(forward, world_up))
    if right == (0.0, 0.0, 0.0):
        right = _normalize(_cross(forward, (1.0, 0.0, 0.0)))
    up = _normalize(_cross(right, forward))

    vertical_half = math.radians(vertical_fov_deg) / 2
    horizontal_half = math.atan(math.tan(vertical_half) * aspect_ratio)

    for point in _mirror_samples(scene_spec):
        offset = _sub(point, camera_position)
        depth = _dot(offset, forward)
        if depth <= 0:
            continue

        horizontal_angle = math.atan2(_dot(offset, right), depth)
        vertical_angle = math.atan2(_dot(offset, up), depth)
        if abs(horizontal_angle) <= horizontal_half and abs(vertical_angle) <= vertical_half:
            return True

    return False
