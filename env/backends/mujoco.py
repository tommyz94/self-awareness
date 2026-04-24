"""
MuJoCo backend for the mirror self-recognition simulator.

Requires: pip install mujoco pillow

The scene geometry is deterministic and shared across experiment presets.
Robot movement is kinematic: the humanoid root pose is updated directly
via qpos each step, while the articulated body stays in a stable standing
pose. The mirror is approximated by a visual clone reflected across the
active scene mirror plane.

Coordinate system: Y-up.
  gravity = (0, -9.81, 0)
  North (default facing) = +X axis.

Usage:
    env = MuJoCoEnv(scene_spec=SceneSpec("mirror_room", mirror_enabled=True), third_person=False)
    x, z, direction, bumped = env.step("w")
    path = env.capture_image("outputs/frame.jpg")
"""
from __future__ import annotations

import math
import os
import platform
import random
from pathlib import Path
from typing import Optional, Tuple

from env.base import BaseEnv
from env.observation import ObservationMetadata, mirror_visible_from_camera
from env.backends.rest_pose import MUJOCO_ARM_REST_POSE as _ARM_REST_POSE
from env.scenes.mujoco_builder import build_humanoid_scene
from experiments.specs.types import BackendOverrides, MuJoCoSceneBuildResult, RunSetup, SceneSpec
from utils.colors import rgba_for_color as _rgba_for_color

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "mujoco"
_CUBE_XML_PATH = _ASSETS_DIR / "cube.xml"
_TEXTURES_DIR = _ASSETS_DIR / "textures"
_FLOOR_TEXTURE_PATH = _TEXTURES_DIR / "floor_wood_1.png"
_WALL_TEXTURE_PATH = _TEXTURES_DIR / "wall_plaster.png"

# Default mirror-room constants.
_MIRROR_GAP = 1.0
_MIRROR_W   = 8.0

# Robot spawn position. Y is aligned so the imported humanoid stands on the floor.
_SPAWN_X, _SPAWN_Y, _SPAWN_Z = -10.0, -1.0, 2.5
_CUBE_Y = -0.75

# Step sizes
_MOVE_STEP = 2.0
_TURN_STEP = math.pi / 2
_E4_CORRIDOR_FORWARD_STEP = 1.0
_E4_CORRIDOR_ROWS_PER_SIDE = 10
_E4_MIRROR_PANE_ALPHA = 0.03

# Collision envelope for the kinematic root.
_ROBOT_SIZE_X = 0.5
_ROBOT_SIZE_Z = 0.3

_CUBE_XML_RGBA = {
    "red": "1 0 0 1",
    "green": "0 0.8 0 1",
    "blue": "0 0.3 1 1",
}
_MIRROR_CUBE_XML_RGBA = {
    "red": "1 0 0 1",
    "green": "0 0.8 0 1",
    "blue": "0 0.3 1 1",
}
_INITIAL_CUBE_POSITIONS = {
    "red": (-13.0, -6.0),
    "green": (-13.0, 6.0),
    "blue": (-2.0, -6.0),
}

_CUBE_GEOM_NAMES    = {"red": "cube_red_g", "green": "cube_green_g", "blue": "cube_blue_g"}
_CUBE_JOINT_NAMES   = {"red": "cube_red_j", "green": "cube_green_j", "blue": "cube_blue_j"}
_MIRROR_CUBE_GEOM_NAMES = {
    "red": "mirror_cube_red_g",
    "green": "mirror_cube_green_g",
    "blue": "mirror_cube_blue_g",
}
_MIRROR_CUBE_JOINT_NAMES = {
    "red": "mirror_cube_red_j",
    "green": "mirror_cube_green_j",
    "blue": "mirror_cube_blue_j",
}
_TWO_ROOM_OFFWIDTH = 1600
_TWO_ROOM_OFFHEIGHT = 1200
_TWO_ROOM_OFFSAMPLES = 8
_TWO_ROOM_SHADOWSIZE = 7000
_REFLECTION_PIXEL_THRESHOLD = 12


def _surface_texture_assets_xml() -> str:
    floor_path = _FLOOR_TEXTURE_PATH.resolve().as_posix()
    wall_path = _WALL_TEXTURE_PATH.resolve().as_posix()
    return f"""
    <texture name="floor_wood_tex" type="2d" file="{floor_path}"/>
    <texture name="wall_plaster_tex" type="2d" file="{wall_path}"/>
    <material name="floor_wood_mat"
              texture="floor_wood_tex"
              texrepeat="6 6"
              texuniform="true"
              rgba=".62 .50 .40 1"
              reflectance=".00"
              shininess=".00"
              specular=".00"
              emission=".12"/>
    <material name="wall_plaster_mat"
              rgba=".78 .78 .78 1"
              reflectance=".00"
              shininess=".00"
              specular=".00"
              emission=".22"/>"""


def _inject_asset_entries(asset_xml: str, extra_entries: str) -> str:
    if not asset_xml:
        return f"<asset>{extra_entries}\n  </asset>"
    closing_tag = "</asset>"
    if closing_tag not in asset_xml:
        raise ValueError("MuJoCo asset XML is missing </asset>.")
    return asset_xml.replace(closing_tag, f"{extra_entries}\n  {closing_tag}")


def _segmentation_has_geom_pixels(
    segmentation_frame,
    geom_ids: set[int],
    *,
    geom_obj_type: Optional[int] = None,
    min_pixels: int = _REFLECTION_PIXEL_THRESHOLD,
) -> Optional[bool]:
    if not geom_ids:
        return False
    if segmentation_frame is None:
        return None

    try:
        import numpy as np
    except Exception:
        return None

    frame = np.asarray(segmentation_frame)
    if frame.size == 0:
        return False

    geom_array = np.array(sorted(geom_ids), dtype=np.int64)
    min_pixels = max(1, int(min_pixels))

    def _hits(mask) -> bool:
        count = int(np.count_nonzero(mask))
        return count >= min_pixels

    # Common MuJoCo segmentation format: (H, W, 2) with object type + object id.
    if frame.ndim == 3 and frame.shape[2] >= 2:
        ch0 = frame[..., 0].astype(np.int64, copy=False)
        ch1 = frame[..., 1].astype(np.int64, copy=False)

        if geom_obj_type is not None:
            if _hits((ch0 == int(geom_obj_type)) & np.isin(ch1, geom_array)):
                return True
            if _hits((ch1 == int(geom_obj_type)) & np.isin(ch0, geom_array)):
                return True

        if _hits(np.isin(ch0, geom_array)) or _hits(np.isin(ch1, geom_array)):
            return True

        # Some pipelines encode object ids in RGB channels.
        if frame.shape[2] >= 3:
            r = frame[..., 0].astype(np.int64, copy=False)
            g = frame[..., 1].astype(np.int64, copy=False)
            b = frame[..., 2].astype(np.int64, copy=False)
            packed_rgb = r + 256 * g + 65536 * b
            packed_bgr = b + 256 * g + 65536 * r
            if _hits(np.isin(packed_rgb, geom_array)) or _hits(np.isin(packed_bgr, geom_array)):
                return True

        return False

    # Fallback: 2D id buffer.
    if frame.ndim == 2:
        return _hits(np.isin(frame.astype(np.int64, copy=False), geom_array))

    # Last-resort fallback over flat values.
    flat = frame.reshape(-1).astype(np.int64, copy=False)
    return _hits(np.isin(flat, geom_array))


def _two_room_visual_xml(render_width: int, render_height: int) -> str:
    offwidth = max(render_width, _TWO_ROOM_OFFWIDTH)
    offheight = max(render_height, _TWO_ROOM_OFFHEIGHT)
    return f"""<visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <quality shadowsize="1" offsamples="2"/>
    <headlight ambient=".70 .70 .70" diffuse=".35 .35 .35" specular=".00 .00 .00"/>
  </visual>"""


def _two_room_lights_xml() -> str:
    return """
    <light name="key_light"
          pos="-7.5 6.2 0.0"
          dir="0 -1 0"
          directional="true"
          diffuse=".42 .42 .42"
          specular=".00 .00 .00"
          castshadow="false"/>
    <light name="fill_light"
          pos="7.5 6.2 0.0"
          dir="0 -1 0"
          directional="true"
          diffuse=".42 .42 .42"
          specular=".00 .00 .00"
          castshadow="false"/>
    <light name="back_light"
          pos="0.0 6.2 -5.0"
          dir="0 -1 0"
          directional="true"
          diffuse=".34 .34 .34"
          specular=".00 .00 .00"
          castshadow="false"/>
    <light name="room_fill_left"
          pos="-7.6 5.8 0.0"
          dir="0 -1 0"
          directional="true"
          diffuse=".28 .28 .28"
          specular=".00 .00 .00"/>
    <light name="room_fill_right"
          pos="7.6 5.8 0.0"
          dir="0 -1 0"
          directional="true"
          diffuse=".28 .28 .28"
          specular=".00 .00 .00"/>"""


def _load_cube_template() -> str:
    if not _CUBE_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo cube asset not found at {_CUBE_XML_PATH}")
    template = _CUBE_XML_PATH.read_text().strip()
    if not template:
        raise ValueError(f"MuJoCo cube asset at {_CUBE_XML_PATH} is empty")
    return template


def _build_cube_xml(*, include_cubes: bool, include_mirror_cubes: bool) -> str:
    if not include_cubes and not include_mirror_cubes:
        return ""
    template = _load_cube_template()
    cube_bodies = []
    if include_cubes:
        for color_name in ("red", "green", "blue"):
            x, z = _INITIAL_CUBE_POSITIONS[color_name]
            cube_bodies.append(
                template.format(
                    body_name=f"cube_{color_name}",
                    joint_name=_CUBE_JOINT_NAMES[color_name],
                    geom_name=_CUBE_GEOM_NAMES[color_name],
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
                    joint_name=_MIRROR_CUBE_JOINT_NAMES[color_name],
                    geom_name=_MIRROR_CUBE_GEOM_NAMES[color_name],
                    x=x,
                    z=z,
                    rgba=_MIRROR_CUBE_XML_RGBA[color_name],
                )
            )
    return "\n    <!-- Colored cubes -->\n    " + "\n    ".join(cube_bodies)


def _build_two_room_xml(
    scene_spec: SceneSpec,
    render_width: int = 640,
    render_height: int = 480,
    camera_height: float = 1.0,
    body_style: str = "default",
) -> tuple[str, list[str], list[str]]:
    mirror_present = scene_spec.mirror_enabled
    humanoid_scene = build_humanoid_scene(
        mirror_present=mirror_present,
        spawn_pos=(scene_spec.spawn_pos[0], _SPAWN_Y, scene_spec.spawn_pos[1]),
        camera_height=camera_height,
        body_style=body_style,
    )
    asset_xml = _inject_asset_entries(
        humanoid_scene.asset_xml,
        _surface_texture_assets_xml(),
    )
    rl = scene_spec.length
    rw = scene_spec.width
    rh = scene_spec.height
    wt = scene_spec.wall_thickness
    z_window = scene_spec.mirror_window_z or (_MIRROR_GAP, _MIRROR_GAP + _MIRROR_W)
    room_min_z = -rw / 2
    room_max_z = rw / 2
    z0 = float(max(room_min_z, min(room_max_z, z_window[0])))
    z1 = float(max(room_min_z, min(room_max_z, z_window[1])))
    if z1 < z0:
        z0, z1 = z1, z0
    mg = z0
    mw = z1 - z0

    cx = -(rl / 2 + wt / 2)   # center x of left room ≈ -7.6

    # Central wall segments
    if mirror_present:
        left_len = max(0.0, mg - room_min_z)
        right_len = max(0.0, room_max_z - (mg + mw))
        mirror_len = max(0.0, mw)
        central_parts = []
        if left_len > 1e-6:
            left_pz = (room_min_z + mg) / 2
            central_parts.append(
                f"""
    <!-- Central wall left of mirror -->
    <geom name="cwall_l" type="box"
          pos="0 {rh/2-0.95:.3f} {left_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {left_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>"""
            )
        if mirror_len > 1e-6:
            mir_pz = mg + mirror_len / 2
            central_parts.append(
                f"""
    <!-- Mirror pane (semi-transparent) -->
    <geom name="mirror_pane" type="box"
          pos="0 {rh/2-0.95:.3f} {mir_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {mirror_len/2:.3f}"
          rgba=".85 .87 .90 .12" contype="0" conaffinity="0"/>"""
            )
        if right_len > 1e-6:
            right_pz = (mg + mirror_len + room_max_z) / 2
            central_parts.append(
                f"""
    <!-- Central wall right of mirror -->
    <geom name="cwall_r" type="box"
          pos="0 {rh/2-0.95:.3f} {right_pz:.3f}"
          size="{wt/2:.3f} {rh/2:.3f} {right_len/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>"""
            )
        central_xml = "".join(central_parts)
    else:
        central_xml = f"""
    <!-- Solid central wall (no mirror, E2) -->
    <geom name="cwall_full" type="box"
          pos="0 {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>"""

    xml = f"""<mujoco model="mirror_sim">
  <compiler angle="degree"/>
  <option gravity="0 -9.81 0" timestep="0.01"/>
  {_two_room_visual_xml(render_width, render_height)}
  {asset_xml}
  {humanoid_scene.default_xml}

  <worldbody>
    {_two_room_lights_xml()}
    <!-- Left room floor -->
    <geom name="floor_l" type="box"
          pos="{cx:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>
    <!-- Right room floor -->
    <geom name="floor_r" type="box"
          pos="{rl/2+wt/2:.3f} -1.05 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="floor_wood_mat" contype="0" conaffinity="0"/>

    <!-- Left room back wall (z = -rw/2) -->
    <geom name="wall_back_l" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Left room front wall (z = +rw/2) -->
    <geom name="wall_front_l" type="box"
          pos="{cx:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Left room outer wall (x = -rl) -->
    <geom name="wall_outer_l" type="box"
          pos="{-rl-wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Right room back wall (z = -rw/2) -->
    <geom name="wall_back_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {-rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Right room front wall (z = +rw/2) -->
    <geom name="wall_front_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh/2-0.95:.3f} {rw/2:.3f}"
          size="{rl/2:.3f} {rh/2:.3f} {wt/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Right room outer wall (x = +rl) -->
    <geom name="wall_outer_r" type="box"
          pos="{rl+wt:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Left room ceiling -->
    <geom name="ceiling_l" type="box"
          pos="{cx:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    <!-- Right room ceiling -->
    <geom name="ceiling_r" type="box"
          pos="{rl/2+wt/2:.3f} {rh-0.95:.3f} 0"
          size="{rl/2:.3f} .05 {rw/2:.3f}"
          material="wall_plaster_mat" contype="0" conaffinity="0"/>
    {central_xml}
    {_build_cube_xml(include_cubes=True, include_mirror_cubes=mirror_present)}

    {humanoid_scene.robot_body_xml}
    {humanoid_scene.mirror_body_xml}
  </worldbody>
</mujoco>"""
    return xml, humanoid_scene.robot_geom_names, humanoid_scene.mirror_geom_names


def _build_corridor_xml(
    scene_spec: SceneSpec,
    render_width: int = 640,
    render_height: int = 480,
    camera_height: float = 1.0,
    body_style: str = "default",
) -> tuple[str, list[str], list[str]]:
    humanoid_scene = build_humanoid_scene(
        mirror_present=True,
        spawn_pos=(scene_spec.spawn_pos[0], _SPAWN_Y, scene_spec.spawn_pos[1]),
        camera_height=camera_height,
        body_style=body_style,
    )
    length = scene_spec.length
    width = scene_spec.width
    rh = scene_spec.height
    wt = scene_spec.wall_thickness
    half_len = length / 2
    half_w = width / 2
    rows_per_side = _E4_CORRIDOR_ROWS_PER_SIDE
    tile_len = length / rows_per_side
    tile_width = width / 3
    mirror_x = scene_spec.mirror_plane_x if scene_spec.mirror_plane_x is not None else half_len
    right_center_x = mirror_x + half_len
    right_end_x = mirror_x + length
    left_tile_xml: list[str] = []
    for row in range(rows_per_side):
        tile_x = -half_len + tile_len / 2 + row * tile_len
        for lane in range(3):
            tile_z = -half_w + tile_width / 2 + lane * tile_width
            tile_rgba = ".92 .92 .92 1" if (row + lane) % 2 == 0 else ".22 .22 .22 1"
            left_tile_xml.append(
                f"""    <geom name="floor_tile_left_r{row}_c{lane}" type="box"
          pos="{tile_x:.3f} -1.05 {tile_z:.3f}"
          size="{tile_len/2:.3f} .05 {tile_width/2:.3f}"
          rgba="{tile_rgba}" contype="0" conaffinity="0"/>"""
            )
    left_tiles_xml = "\n    ".join(left_tile_xml)
    right_corridor_xml = ""
    if scene_spec.mirror_plane_x is not None:
        right_tile_xml: list[str] = []
        for row in range(rows_per_side):
            tile_x = -half_len + tile_len / 2 + row * tile_len
            mirror_tile_x = 2 * mirror_x - tile_x
            for lane in range(3):
                tile_z = -half_w + tile_width / 2 + lane * tile_width
                tile_rgba = ".92 .92 .92 1" if (row + lane) % 2 == 0 else ".22 .22 .22 1"
                right_tile_xml.append(
                    f"""    <geom name="floor_tile_right_r{row}_c{lane}" type="box"
          pos="{mirror_tile_x:.3f} -1.05 {tile_z:.3f}"
          size="{tile_len/2:.3f} .05 {tile_width/2:.3f}"
          rgba="{tile_rgba}" contype="0" conaffinity="0"/>"""
                )
        right_tiles_xml = "\n    ".join(right_tile_xml)
        right_corridor_xml = f"""
    {right_tiles_xml}
    <geom name="wall_left_right" type="box"
          pos="{right_center_x:.3f} {rh/2-0.95:.3f} {-half_w:.3f}"
          size="{half_len:.3f} {rh/2:.3f} {wt/2:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>
    <geom name="wall_right_right" type="box"
          pos="{right_center_x:.3f} {rh/2-0.95:.3f} {half_w:.3f}"
          size="{half_len:.3f} {rh/2:.3f} {wt/2:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>
    <geom name="wall_end" type="box"
          pos="{right_end_x:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {half_w:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>"""
    mirror_pane_xml = f"""
    <geom name="mirror_pane" type="box"
          pos="{mirror_x:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {half_w:.3f}"
          rgba=".70 .70 .70 {_E4_MIRROR_PANE_ALPHA:.2f}" contype="0" conaffinity="0"/>"""

    xml = f"""<mujoco model="corridor_sim">
  <compiler angle="degree"/>
  <option gravity="0 -9.81 0" timestep="0.01"/>
  <visual>
    <global offwidth="{render_width}" offheight="{render_height}"/>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular=".1 .1 .1"/>
  </visual>
  {humanoid_scene.asset_xml}
  {humanoid_scene.default_xml}

  <worldbody>
    {left_tiles_xml}
    <geom name="wall_left" type="box"
          pos="0 {rh/2-0.95:.3f} {-half_w:.3f}"
          size="{half_len:.3f} {rh/2:.3f} {wt/2:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>
    <geom name="wall_right" type="box"
          pos="0 {rh/2-0.95:.3f} {half_w:.3f}"
          size="{half_len:.3f} {rh/2:.3f} {wt/2:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>
    <geom name="wall_start" type="box"
          pos="{-half_len:.3f} {rh/2-0.95:.3f} 0"
          size="{wt/2:.3f} {rh/2:.3f} {half_w:.3f}"
          rgba=".7 .7 .7 1" contype="0" conaffinity="0"/>
    {mirror_pane_xml}
    {right_corridor_xml}

    {humanoid_scene.robot_body_xml}
    {humanoid_scene.mirror_body_xml}
  </worldbody>
</mujoco>"""
    return xml, humanoid_scene.robot_geom_names, humanoid_scene.mirror_geom_names


def _build_xml(
    scene_spec: SceneSpec,
    render_width: int = 640,
    render_height: int = 480,
    camera_height: float = 1.0,
    body_style: str = "default",
) -> tuple[str, list[str], list[str]]:
    if scene_spec.layout_kind == "corridor":
        return _build_corridor_xml(
            scene_spec,
            render_width=render_width,
            render_height=render_height,
            camera_height=camera_height,
            body_style=body_style,
        )
    return _build_two_room_xml(
        scene_spec,
        render_width=render_width,
        render_height=render_height,
        camera_height=camera_height,
        body_style=body_style,
    )


def _build_scene_result(
    scene_spec: SceneSpec,
    render_width: int = 640,
    render_height: int = 480,
    camera_height: float = 1.0,
    body_style: str = "default",
) -> MuJoCoSceneBuildResult:
    xml, robot_geom_names, mirror_geom_names = _build_xml(
        scene_spec,
        render_width=render_width,
        render_height=render_height,
        camera_height=camera_height,
        body_style=body_style,
    )
    return MuJoCoSceneBuildResult(
        xml=xml,
        robot_geom_names=robot_geom_names,
        mirror_geom_names=mirror_geom_names,
    )


def _yaw_to_quat(yaw: float) -> list:
    """Rotation by yaw radians around Y-axis → [qw, qx, qy, qz]."""
    return [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]


def _yaw_to_direction(yaw: float) -> str:
    """Map yaw to this repo's cardinal convention: N=+X, W=-Z, S=-X, E=+Z."""
    norm_yaw = yaw % (2 * math.pi)
    sector = round(norm_yaw / (math.pi / 2)) % 4
    return ["N", "W", "S", "E"][sector]


def _uses_lateral_actions(scene_id: str | None) -> bool:
    return scene_id == "narrow_corridor_mirror"


class MuJoCoEnv(BaseEnv):
    """MuJoCo/offscreen rendering simulation environment."""

    def __init__(
        self,
        scene_spec: SceneSpec = None,
        backend_overrides: Optional[BackendOverrides] = None,
        third_person: bool = False,
        camera_height: float = 1.0,
        body_style: str = "default",
        render_width: int = _TWO_ROOM_OFFWIDTH,
        render_height: int = _TWO_ROOM_OFFHEIGHT,
        headless: bool = True,
    ):
        if headless and platform.system() == "Linux":
            # MuJoCo must choose a headless GL backend before the mujoco module is imported.
            # Prefer EGL on Linux GPU nodes, while still allowing callers to override it
            # explicitly with MUJOCO_GL=osmesa for CPU-only environments.
            os.environ.setdefault("MUJOCO_GL", "egl")

        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo Python bindings not found. Install with: pip install mujoco"
            ) from exc

        if scene_spec is None:
            scene_spec = SceneSpec(scene_id="mirror_room", mirror_enabled=True)
        self._scene_spec = scene_spec
        self._scene_id = scene_spec.scene_id
        self._strafe_actions = _uses_lateral_actions(scene_spec.scene_id)
        self._corridor_diagonal_actions = self._strafe_actions
        self._layout_kind = scene_spec.layout_kind
        self._scene_length = scene_spec.length
        self._scene_width = scene_spec.width
        self._scene_height = scene_spec.height
        self._wall_thickness = scene_spec.wall_thickness
        self._spawn_x = scene_spec.spawn_pos[0]
        self._spawn_z = scene_spec.spawn_pos[1]
        self._spawn_yaw = scene_spec.spawn_yaw
        self._spawn_y = _SPAWN_Y
        self._mirror_plane_x = scene_spec.mirror_plane_x
        self._mirror_aperture_mode = scene_spec.mirror_aperture_mode
        self._mirror_window_z = scene_spec.mirror_window_z

        mirror_present = scene_spec.mirror_enabled
        self._mj = mujoco
        self._mirror_present = mirror_present
        self._use_mirror = mirror_present
        self._third_person = third_person
        self._camera_height = camera_height
        self._body_style = str(body_style or "default")
        self._turn_step = math.radians(float(scene_spec.turn_step_deg))
        self._corridor_forward_step = _E4_CORRIDOR_FORWARD_STEP if self._corridor_diagonal_actions else _MOVE_STEP
        self._corridor_lane_step = (
            (self._scene_width / 3.0) if self._corridor_diagonal_actions else _MOVE_STEP
        )

        scene_builder = (
            backend_overrides.mujoco_scene_builder
            if backend_overrides is not None else None
        )
        self._reset_applier = (
            backend_overrides.mujoco_reset_applier
            if backend_overrides is not None else None
        )
        self._step_applier = (
            backend_overrides.mujoco_step_applier
            if backend_overrides is not None else None
        )
        if scene_builder is not None:
            scene_build = scene_builder(scene_spec, render_width, render_height)
        else:
            scene_build = _build_scene_result(
                scene_spec,
                render_width=render_width,
                render_height=render_height,
                camera_height=self._camera_height,
                body_style=self._body_style,
            )
        self._model = mujoco.MjModel.from_xml_string(scene_build.xml)
        self._data  = mujoco.MjData(self._model)
        self._wall_material_id = self._material_id_if_exists("wall_plaster_mat")
        self._default_wall_rgba = (
            [float(v) for v in self._model.mat_rgba[self._wall_material_id]]
            if self._wall_material_id is not None
            else [0.78, 0.78, 0.78, 1.0]
        )

        # Renderer
        try:
            self._renderer = mujoco.Renderer(
                self._model,
                height=render_height,
                width=render_width,
            )
        except Exception as exc:
            if headless and platform.system() == "Linux":
                backend = os.environ.get("MUJOCO_GL", "unset")
                raise RuntimeError(
                    "Failed to initialize MuJoCo headless rendering. "
                    f"Current MUJOCO_GL={backend!r}. "
                    "Try MUJOCO_GL=egl on GPU nodes or MUJOCO_GL=osmesa on CPU-only nodes."
                ) from exc
            raise

        # Cache joint qpos addresses
        self._robot_jnt_addr  = self._jnt_addr("robot_root")
        self._mirror_present  = mirror_present
        self._secondary_present = bool(scene_build.mirror_geom_names)
        self._secondary_active = self._layout_kind == "corridor" and not self._use_mirror
        if self._secondary_present:
            self._mirror_jnt_addr = self._jnt_addr("mirror_root")
        self._cube_jnt_addrs = {}
        self._mirror_cube_jnt_addrs = {}
        if self._layout_kind != "corridor":
            self._cube_jnt_addrs = {
                color: self._jnt_addr(jname)
                for color, jname in _CUBE_JOINT_NAMES.items()
            }
            if mirror_present:
                self._mirror_cube_jnt_addrs = {
                    color: self._jnt_addr(jname)
                    for color, jname in _MIRROR_CUBE_JOINT_NAMES.items()
                }

        # Cache geom ids for color changes
        self._robot_geom_ids  = [self._geom_id(name) for name in scene_build.robot_geom_names]
        mirror_geom_names = (
            list(scene_build.mirror_geom_names)
            if self._secondary_present else []
        )
        self._mirror_geom_ids = [self._geom_id(name) for name in mirror_geom_names]
        # Track only the mirrored ego body (exclude distractors/cubes) so we can
        # detect whether the actual self-reflection appears in the rendered frame.
        self._mirror_ego_geom_ids = [
            self._geom_id(name)
            for name in mirror_geom_names
            if not name.startswith("mirror_distractor_")
            and not name.startswith("mirror_cube_")
        ]
        self._cube_geom_ids = {}
        self._mirror_cube_geom_ids = {}
        if self._layout_kind != "corridor":
            self._cube_geom_ids = {
                color: self._geom_id(gname)
                for color, gname in _CUBE_GEOM_NAMES.items()
            }
            if mirror_present:
                self._mirror_cube_geom_ids = {
                    color: self._geom_id(gname)
                    for color, gname in _MIRROR_CUBE_GEOM_NAMES.items()
                }
        self._robot_arm_jnt_addrs = self._arm_joint_addrs_for_prefix("robot_")
        self._mirror_arm_jnt_addrs = (
            self._arm_joint_addrs_for_prefix("mirror_")
            if self._secondary_present else {}
        )

        # Kinematic state
        self._x:   float = round(self._spawn_x, 2)
        self._z:   float = round(self._spawn_z, 2)
        self._yaw: float = self._spawn_yaw
        self._secondary_x: float = round(
            (2 * self._mirror_plane_x - self._spawn_x) if self._mirror_plane_x is not None else self._spawn_x,
            2,
        )
        self._secondary_z: float = round(self._spawn_z, 2)
        self._secondary_yaw: float = self._spawn_yaw + math.pi
        self._object_pos_2d: dict = {}

        # Apply initial kinematics
        self._apply_arm_rest_pose()
        self._sync_robot()
        self._sync_mirror()
        self._mj.mj_forward(self._model, self._data)

        # Window / display state
        self._window_enabled = not headless   # True → show OpenCV window
        self._window_open = False             # True once namedWindow is created
        self._cv_window = "Mirror AI - MuJoCo"

        # Human-mode state (populated by bind_keys; no-op when None)
        self._human_callback = None
        self._human_camera = "third_person" if third_person else "first_person"

    # ------------------------------------------------------------------
    # MuJoCo helpers
    # ------------------------------------------------------------------

    def _jnt_addr(self, name: str) -> int:
        jnt_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_JOINT, name)
        return int(self._model.jnt_qposadr[jnt_id])

    def _geom_id(self, name: str) -> int:
        return self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_GEOM, name)

    def _jnt_addr_if_exists(self, name: str) -> Optional[int]:
        jnt_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            return None
        return int(self._model.jnt_qposadr[jnt_id])

    def _material_id_if_exists(self, name: str) -> Optional[int]:
        material_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_MATERIAL, name)
        if material_id < 0:
            return None
        return int(material_id)

    def _geom_ids_with_prefix(self, prefix: str) -> list[int]:
        geom_ids: list[int] = []
        for geom_id in range(self._model.ngeom):
            name = self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_GEOM, geom_id)
            if name and name.startswith(prefix):
                geom_ids.append(geom_id)
        return geom_ids

    def _set_body_pose(self, jnt_addr: int, x: float, y: float, z: float, yaw: float) -> None:
        qw, qx, qy, qz = _yaw_to_quat(yaw)
        d = self._data.qpos
        d[jnt_addr]   = x
        d[jnt_addr+1] = y
        d[jnt_addr+2] = z
        d[jnt_addr+3] = qw
        d[jnt_addr+4] = qx
        d[jnt_addr+5] = qy
        d[jnt_addr+6] = qz

    def _set_body_pos(self, jnt_addr: int, x: float, y: float, z: float) -> None:
        d = self._data.qpos
        d[jnt_addr]   = x
        d[jnt_addr+1] = y
        d[jnt_addr+2] = z

    def _park_body(self, jnt_addr: int) -> None:
        self._set_body_pos(jnt_addr, 1000.0, 0.0, 0.0)

    def _set_joint_qpos(self, jnt_addr: int, value: float) -> None:
        self._data.qpos[jnt_addr] = value

    def _arm_joint_addrs_for_prefix(self, prefix: str) -> dict[str, int]:
        joint_addrs: dict[str, int] = {}
        for joint_name in _ARM_REST_POSE:
            addr = self._jnt_addr_if_exists(f"{prefix}{joint_name}")
            if addr is not None:
                joint_addrs[joint_name] = addr
        return joint_addrs

    def _apply_arm_rest_pose_to_joint_addrs(self, joint_addrs: dict[str, int]) -> None:
        for joint_name, value in _ARM_REST_POSE.items():
            addr = joint_addrs.get(joint_name)
            if addr is not None:
                self._set_joint_qpos(addr, value)

    def _apply_arm_rest_pose_for_prefix(self, prefix: str) -> None:
        self._apply_arm_rest_pose_to_joint_addrs(self._arm_joint_addrs_for_prefix(prefix))

    def _set_color(self, geom_ids: list, rgba: list) -> None:
        for gid in geom_ids:
            self._model.geom_rgba[gid] = rgba

    def _apply_wall_color(self, wall_rgba: Optional[tuple[float, float, float, float]]) -> None:
        if self._wall_material_id is None:
            return
        rgba = list(wall_rgba) if wall_rgba is not None else list(self._default_wall_rgba)
        if len(rgba) != 4:
            return
        self._model.mat_rgba[self._wall_material_id] = [float(value) for value in rgba]

    def _sync_robot(self) -> None:
        """Write current (_x, _z, _yaw) to MuJoCo qpos for the main robot."""
        self._set_body_pose(self._robot_jnt_addr, self._x, self._spawn_y, self._z, self._yaw)

    def _sync_mirror(self) -> None:
        """Update the scene's secondary body, either as a reflection or an agent."""
        if not self._secondary_present:
            return
        if self._mirror_present and self._mirror_plane_x is not None:
            mirror_x = 2 * self._mirror_plane_x - self._x
            mirror_yaw = -self._yaw  # flip yaw for reflection
            self._set_body_pose(self._mirror_jnt_addr, mirror_x, self._spawn_y, self._z, mirror_yaw)
            return
        if self._secondary_active:
            self._set_body_pose(
                self._mirror_jnt_addr,
                self._secondary_x,
                self._spawn_y,
                self._secondary_z,
                self._secondary_yaw,
            )
        else:
            self._park_body(self._mirror_jnt_addr)

    def _is_in_mirror_aperture(self, z: float) -> bool:
        if not self._mirror_present:
            return False
        if self._mirror_aperture_mode == "window" and self._mirror_window_z is not None:
            z_min, z_max = self._mirror_window_z
            return z_min <= z <= z_max
        return True

    def _apply_arm_rest_pose(self) -> None:
        """Apply a relaxed arm pose to the humanoid and its mirror clone."""
        self._apply_arm_rest_pose_to_joint_addrs(self._robot_arm_jnt_addrs)
        if self._secondary_present:
            self._apply_arm_rest_pose_to_joint_addrs(self._mirror_arm_jnt_addrs)

    def _set_box_geom(
        self,
        geom_name: str,
        *,
        x: float,
        y: float,
        z: float,
        size_x: float,
        size_y: float,
        size_z: float,
    ) -> None:
        geom_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            return
        self._model.geom_pos[geom_id, 0] = float(x)
        self._model.geom_pos[geom_id, 1] = float(y)
        self._model.geom_pos[geom_id, 2] = float(z)
        self._model.geom_size[geom_id, 0] = max(float(size_x), 1e-4)
        self._model.geom_size[geom_id, 1] = max(float(size_y), 1e-4)
        self._model.geom_size[geom_id, 2] = max(float(size_z), 1e-4)

    def _set_or_hide_two_room_segment(
        self,
        geom_name: str,
        *,
        wall_x: float,
        y: float,
        z_center: float,
        z_length: float,
    ) -> None:
        if z_length <= 1e-4:
            self._set_box_geom(
                geom_name,
                x=wall_x,
                y=y,
                z=1000.0,
                size_x=self._wall_thickness / 2,
                size_y=self._scene_height / 2,
                size_z=1e-4,
            )
            return
        self._set_box_geom(
            geom_name,
            x=wall_x,
            y=y,
            z=z_center,
            size_x=self._wall_thickness / 2,
            size_y=self._scene_height / 2,
            size_z=z_length / 2,
        )

    def _apply_two_room_mirror_window_geometry(self) -> None:
        if self._layout_kind == "corridor":
            return
        if not self._mirror_present or self._mirror_window_z is None:
            return

        room_min_z = -self._scene_width / 2
        room_max_z = self._scene_width / 2
        z0 = float(max(room_min_z, min(room_max_z, self._mirror_window_z[0])))
        z1 = float(max(room_min_z, min(room_max_z, self._mirror_window_z[1])))
        if z1 < z0:
            z0, z1 = z1, z0

        left_len = max(0.0, z0 - room_min_z)
        right_len = max(0.0, room_max_z - z1)
        mirror_len = max(0.0, z1 - z0)
        wall_x = self._mirror_plane_x if self._mirror_plane_x is not None else -self._wall_thickness / 2
        wall_y = self._scene_height / 2 - 0.95

        self._set_or_hide_two_room_segment(
            "cwall_l",
            wall_x=wall_x,
            y=wall_y,
            z_center=(room_min_z + z0) / 2,
            z_length=left_len,
        )
        self._set_or_hide_two_room_segment(
            "cwall_r",
            wall_x=wall_x,
            y=wall_y,
            z_center=(z1 + room_max_z) / 2,
            z_length=right_len,
        )
        self._set_or_hide_two_room_segment(
            "mirror_pane",
            wall_x=wall_x,
            y=wall_y,
            z_center=(z0 + z1) / 2,
            z_length=mirror_len,
        )

    def _blocking_walls(self) -> list[tuple[float, float, float, float]]:
        """Return the blocking wall boxes used by the current scene."""
        walls: list[tuple[float, float, float, float]] = []
        wt = self._wall_thickness

        if self._layout_kind == "corridor":
            if self._mirror_present and self._mirror_plane_x is not None:
                walls.append((self._mirror_plane_x, 0.0, wt, self._scene_width))
            return walls

        if self._mirror_present and self._mirror_window_z is not None:
            z_min, z_max = self._mirror_window_z
            min_z = -self._scene_width / 2
            max_z = self._scene_width / 2
            left_len = z_min - min_z
            left_pz = (min_z + z_min) / 2
            right_len = max_z - z_max
            right_pz = (z_max + max_z) / 2
            mirror_pz = (z_min + z_max) / 2
            walls.extend([
                (self._mirror_plane_x if self._mirror_plane_x is not None else -wt / 2, mirror_pz, wt, z_max - z_min),
                (0.0, left_pz, wt, left_len),
                (0.0, right_pz, wt, right_len),
            ])
        else:
            walls.append((0.0, 0.0, wt, self._scene_width))

        return walls

    def _try_move(self, dx: float, dz: float) -> bool:
        """
        Move in small increments so a 2-unit step cannot tunnel through the thin
        center wall or mirror pane.
        """
        distance = math.hypot(dx, dz)
        steps = max(1, int(distance / 0.1))
        origin_x, origin_z = self._x, self._z

        for i in range(1, steps + 1):
            nx = origin_x + dx * (i / steps)
            nz = origin_z + dz * (i / steps)
            if not self._is_position_free(nx, nz):
                return False

        self._x = round(origin_x + dx, 2)
        self._z = round(origin_z + dz, 2)
        return True

    def _is_position_free(self, x: float, z: float) -> bool:
        """Check collisions for the kinematic root against scene bounds and walls."""
        if self._layout_kind == "corridor":
            min_x = -self._scene_length / 2
            if not self._mirror_present and self._mirror_plane_x is not None:
                max_x = self._mirror_plane_x + self._scene_length
            else:
                max_x = self._scene_length / 2
            min_z = -self._scene_width / 2
            max_z = self._scene_width / 2
        else:
            min_x = -self._scene_length - self._wall_thickness
            max_x = self._scene_length + self._wall_thickness
            min_z = -self._scene_width / 2
            max_z = self._scene_width / 2

        if not (min_x < x < max_x and min_z < z < max_z):
            return False

        robot_min_x = x - _ROBOT_SIZE_X / 2
        robot_max_x = x + _ROBOT_SIZE_X / 2
        robot_min_z = z - _ROBOT_SIZE_Z / 2
        robot_max_z = z + _ROBOT_SIZE_Z / 2

        for wall_x, wall_z, wall_size_x, wall_size_z in self._blocking_walls():
            wall_min_x = wall_x - wall_size_x / 2
            wall_max_x = wall_x + wall_size_x / 2
            wall_min_z = wall_z - wall_size_z / 2
            wall_max_z = wall_z + wall_size_z / 2
            overlap = (
                robot_min_x <= wall_max_x and robot_max_x >= wall_min_x and
                robot_min_z <= wall_max_z and robot_max_z >= wall_min_z
            )
            if overlap:
                return False

        return True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def reset(self, run_setup: RunSetup) -> None:
        self._scene_spec = run_setup.scene_spec
        self._strafe_actions = _uses_lateral_actions(run_setup.scene_spec.scene_id)
        self._corridor_diagonal_actions = self._strafe_actions
        self._layout_kind = run_setup.scene_spec.layout_kind
        self._scene_length = run_setup.scene_spec.length
        self._scene_width = run_setup.scene_spec.width
        self._scene_height = run_setup.scene_spec.height
        self._wall_thickness = run_setup.scene_spec.wall_thickness
        self._mirror_plane_x = run_setup.scene_spec.mirror_plane_x
        self._mirror_aperture_mode = run_setup.scene_spec.mirror_aperture_mode
        self._mirror_window_z = run_setup.scene_spec.mirror_window_z
        self._turn_step = math.radians(float(run_setup.scene_spec.turn_step_deg))
        self._corridor_forward_step = _E4_CORRIDOR_FORWARD_STEP if self._corridor_diagonal_actions else _MOVE_STEP
        self._corridor_lane_step = (
            (self._scene_width / 3.0) if self._corridor_diagonal_actions else _MOVE_STEP
        )
        self._use_mirror = run_setup.scene_spec.mirror_enabled
        self._mirror_present = self._use_mirror
        self._secondary_active = self._layout_kind == "corridor" and not self._use_mirror
        if hasattr(self, "_apply_two_room_mirror_window_geometry"):
            self._apply_two_room_mirror_window_geometry()
        self._x = round(run_setup.scene_spec.spawn_pos[0], 2)
        self._z = round(run_setup.scene_spec.spawn_pos[1], 2)
        self._yaw = run_setup.scene_spec.spawn_yaw
        self._secondary_x = round(
            (
                2 * run_setup.scene_spec.mirror_plane_x - run_setup.scene_spec.spawn_pos[0]
                if run_setup.scene_spec.mirror_plane_x is not None
                else run_setup.scene_spec.spawn_pos[0]
            ),
            2,
        )
        self._secondary_z = round(run_setup.scene_spec.spawn_pos[1], 2)
        self._secondary_yaw = run_setup.scene_spec.spawn_yaw + math.pi

        apply_wall_color = getattr(self, "_apply_wall_color", None)
        if callable(apply_wall_color):
            apply_wall_color(run_setup.scene_spec.wall_color_rgba)

        if self._body_style != "clothes":
            rgba = _rgba_for_color(run_setup.agent_color_name, alpha=1.0)
            self._set_color(self._robot_geom_ids, rgba)
            if self._secondary_present:
                alpha = 1.0 if self._mirror_present else 1.0
                self._set_color(self._mirror_geom_ids, [rgba[0], rgba[1], rgba[2], alpha])

        # Cubes
        self._object_pos_2d = {}
        cube_objects = [
            {
                "color": str(obj.color),
                "position": (float(obj.position[0]), float(obj.position[1])),
                "reflect_in_mirror": bool(obj.metadata.get("reflect_in_mirror") is True),
                "reflect_ignore_aperture": bool(obj.metadata.get("reflect_ignore_aperture") is True),
            }
            for obj in run_setup.scene_spec.objects
            if obj.kind == "cube" and obj.color is not None
        ]
        slot_names = list(_CUBE_JOINT_NAMES.keys())
        cube_geom_ids = getattr(self, "_cube_geom_ids", {})
        mirror_cube_geom_ids = getattr(self, "_mirror_cube_geom_ids", {})
        for slot_index, slot_name in enumerate(slot_names):
            addr = self._cube_jnt_addrs.get(slot_name)
            if addr is None:
                continue
            geom_id = cube_geom_ids.get(slot_name)
            if slot_index < len(cube_objects):
                cube = cube_objects[slot_index]
                cx, cz = cube["position"]
                self._set_body_pos(addr, float(cx), _CUBE_Y, float(cz))
                self._object_pos_2d[cube["color"]] = (float(cx), float(cz))
                if geom_id is not None:
                    self._set_color([geom_id], _rgba_for_color(cube["color"], alpha=1.0))
            else:
                self._set_body_pos(addr, 0.0, -100.0, 0.0)
                if geom_id is not None:
                    self._set_color([geom_id], _rgba_for_color("gray", alpha=1.0))
        for slot_index, slot_name in enumerate(slot_names):
            addr = self._mirror_cube_jnt_addrs.get(slot_name)
            if addr is None:
                continue
            geom_id = mirror_cube_geom_ids.get(slot_name)
            if (
                slot_index < len(cube_objects)
                and self._mirror_plane_x is not None
            ):
                cube = cube_objects[slot_index]
                cx, cz = cube["position"]
                reflects = bool(cube.get("reflect_in_mirror", False))
                in_aperture = bool(
                    cube.get("reflect_ignore_aperture", False)
                    or self._is_in_mirror_aperture(float(cz))
                )
                if reflects and in_aperture:
                    mirror_x = 2 * self._mirror_plane_x - float(cx)
                    self._set_body_pos(addr, mirror_x, _CUBE_Y, float(cz))
                    if geom_id is not None:
                        self._set_color([geom_id], _rgba_for_color(cube["color"], alpha=1.0))
                else:
                    self._park_body(addr)
                    if geom_id is not None:
                        self._set_color([geom_id], _rgba_for_color("gray", alpha=1.0))
            else:
                self._park_body(addr)
                if geom_id is not None:
                    self._set_color([geom_id], _rgba_for_color("gray", alpha=1.0))

        self._apply_arm_rest_pose()
        self._sync_robot()
        self._sync_mirror()
        if self._reset_applier is not None:
            self._reset_applier(self, run_setup)
        self._mj.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float]:
        return (round(self._x, 2), round(self._z, 2))

    @property
    def direction(self) -> str:
        return _yaw_to_direction(self._yaw)

    @property
    def object_positions_2d(self) -> dict:
        return dict(self._object_pos_2d)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _corridor_action_delta(self, action: str, *, yaw: float) -> tuple[float, float]:
        forward = self._corridor_forward_step
        lane = self._corridor_lane_step
        ux = math.cos(yaw)
        uz = -math.sin(yaw)
        left_x = uz
        left_z = -ux
        if action == "w":
            return (ux * forward, uz * forward)
        if action == "a":
            return (
                ux * forward + left_x * lane,
                uz * forward + left_z * lane,
            )
        if action == "d":
            return (
                ux * forward - left_x * lane,
                uz * forward - left_z * lane,
            )
        return (0.0, 0.0)

    def step(self, action: str) -> Tuple[float, float, str, bool]:
        bumped = False
        fwd_x =  math.cos(self._yaw) * _MOVE_STEP
        fwd_z = -math.sin(self._yaw) * _MOVE_STEP
        if action == "w":
            if self._corridor_diagonal_actions:
                dx, dz = self._corridor_action_delta("w", yaw=self._yaw)
                if not self._try_move(dx, dz):
                    bumped = True
            elif not self._try_move(fwd_x, fwd_z):
                bumped = True
        elif action == "s":
            if self._corridor_diagonal_actions:
                bumped = True
            elif not self._try_move(-fwd_x, -fwd_z):
                bumped = True
        elif action == "a":
            if self._strafe_actions:
                if self._corridor_diagonal_actions:
                    dx, dz = self._corridor_action_delta("a", yaw=self._yaw)
                    if not self._try_move(dx, dz):
                        bumped = True
                elif not self._try_move(fwd_z, -fwd_x):
                    bumped = True
            else:
                self._yaw += getattr(self, "_turn_step", _TURN_STEP)
        elif action == "d":
            if self._strafe_actions:
                if self._corridor_diagonal_actions:
                    dx, dz = self._corridor_action_delta("d", yaw=self._yaw)
                    if not self._try_move(dx, dz):
                        bumped = True
                elif not self._try_move(-fwd_z, fwd_x):
                    bumped = True
            else:
                self._yaw -= getattr(self, "_turn_step", _TURN_STEP)
        if getattr(self, "_secondary_active", False):
            self._advance_secondary_agent()
        step_applier = getattr(self, "_step_applier", None)
        if step_applier is not None:
            step_applier(self, action)

        self._sync_robot()
        self._sync_mirror()
        self._mj.mj_forward(self._model, self._data)
        return self._x, self._z, self.direction, bumped

    def _sample_secondary_action(self) -> str:
        return random.choice(["w", "a", "d"])

    def _try_move_secondary(self, dx: float, dz: float) -> bool:
        distance = math.hypot(dx, dz)
        steps = max(1, int(distance / 0.1))
        origin_x, origin_z = self._secondary_x, self._secondary_z

        for i in range(1, steps + 1):
            nx = origin_x + dx * (i / steps)
            nz = origin_z + dz * (i / steps)
            if not self._is_position_free(nx, nz):
                return False

        self._secondary_x = round(origin_x + dx, 2)
        self._secondary_z = round(origin_z + dz, 2)
        return True

    def _advance_secondary_agent(self) -> None:
        if not self._secondary_present:
            return
        action = self._sample_secondary_action()
        if self._corridor_diagonal_actions:
            dx, dz = self._corridor_action_delta(action, yaw=self._secondary_yaw)
            if action in {"w", "a", "d"}:
                self._try_move_secondary(dx, dz)
            return

        fwd_x = math.cos(self._secondary_yaw) * _MOVE_STEP
        fwd_z = -math.sin(self._secondary_yaw) * _MOVE_STEP
        if action == "w":
            self._try_move_secondary(fwd_x, fwd_z)
        elif action == "a":
            self._try_move_secondary(fwd_z, -fwd_x)
        elif action == "d":
            self._try_move_secondary(-fwd_z, fwd_x)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_rgb(self, camera: str):
        """Render scene from the given camera, return uint8 (H, W, 3) array."""
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    def _ego_reflection_visible_from_camera(
        self,
        *,
        camera: str,
        mirror_visible: bool,
    ) -> Optional[bool]:
        if not mirror_visible:
            return False
        if not self._mirror_present or not self._secondary_present:
            return False
        if not self._is_in_mirror_aperture(self._z):
            return False
        if not self._mirror_ego_geom_ids:
            return None

        enable_segmentation = getattr(self._renderer, "enable_segmentation_rendering", None)
        disable_segmentation = getattr(self._renderer, "disable_segmentation_rendering", None)
        if not callable(enable_segmentation) or not callable(disable_segmentation):
            return None

        segmentation_frame = None
        try:
            enable_segmentation()
            self._renderer.update_scene(self._data, camera=camera)
            segmentation_frame = self._renderer.render()
        except Exception:
            return None
        finally:
            try:
                disable_segmentation()
            except Exception:
                pass

        geom_obj_type = None
        try:
            geom_obj_type = int(self._mj.mjtObj.mjOBJ_GEOM)
        except Exception:
            geom_obj_type = None

        return _segmentation_has_geom_pixels(
            segmentation_frame,
            set(self._mirror_ego_geom_ids),
            geom_obj_type=geom_obj_type,
        )

    def capture_image(self, save_path: str) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Install pillow: pip install pillow") from exc

        camera = self._human_camera if self._human_callback else (
            "third_person" if self._third_person else "first_person"
        )
        pixels = self._render_rgb(camera)

        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(str(out), format="JPEG", quality=95)
        return str(out)

    def get_observation_metadata(self) -> ObservationMetadata:
        forward = (math.cos(self._yaw), 0.0, -math.sin(self._yaw))
        camera = self._human_camera if self._human_callback else (
            "third_person" if self._third_person else "first_person"
        )
        if camera == "third_person":
            cam_pos = (
                self._x - forward[0] * 3.0,
                self._spawn_y + 2.0,
                self._z - forward[2] * 3.0,
            )
            look_at = (
                self._x + forward[0] * 1.5,
                self._spawn_y,
                self._z + forward[2] * 1.5,
            )
            camera_forward = (
                look_at[0] - cam_pos[0],
                look_at[1] - cam_pos[1],
                look_at[2] - cam_pos[2],
            )
        else:
            cam_pos = (
                self._x + forward[0] * 0.15,
                self._spawn_y + 1.45 * self._camera_height,
                self._z + forward[2] * 0.15,
            )
            camera_forward = forward

        view_direction = camera_forward
        mirror_visible = mirror_visible_from_camera(
            scene_spec=self._scene_spec,
            camera_position=cam_pos,
            camera_forward=view_direction,
            vertical_fov_deg=90.0,
            aspect_ratio=4.0 / 3.0,
        )
        ego_reflection_visible = self._ego_reflection_visible_from_camera(
            camera=camera,
            mirror_visible=mirror_visible,
        )
        return ObservationMetadata(
            view_position=cam_pos,
            view_direction=view_direction,
            mirror_visible=mirror_visible,
            ego_reflection_visible=ego_reflection_visible,
        )

    # ------------------------------------------------------------------
    # Human-mode window / Per-frame update
    # ------------------------------------------------------------------

    def _ensure_window(self) -> None:
        """Lazily create the OpenCV window on first call."""
        if not self._window_open:
            import cv2
            cv2.namedWindow(self._cv_window, cv2.WINDOW_AUTOSIZE)
            self._window_open = True

    def bind_keys(self, callback) -> None:
        import importlib.util
        if importlib.util.find_spec("cv2") is None:
            raise ImportError(
                "opencv-python is required for MuJoCo human mode: pip install opencv-python"
            )
        self._human_callback = callback

    def update_frame(self):
        if not self._window_enabled and self._human_callback is None:
            return  # fully headless LLM mode: no-op

        import cv2

        # ---- human mode ----
        if self._human_callback is not None:
            self._ensure_window()
            frame = self._render_rgb(self._human_camera)
            cv2.imshow(self._cv_window, frame[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(self._cv_window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            keymap = {
                ord("w"): "w", ord("s"): "s",
                ord("a"): "a", ord("d"): "d",
                ord(" "): " ",
                ord("f"): "f",
            }
            if key in (ord("q"), 27):
                return False
            elif key == ord("c"):
                self._human_camera = (
                    "third_person" if self._human_camera == "first_person" else "first_person"
                )
            elif key in keymap:
                class _Evt:
                    pass
                evt = _Evt()
                evt.key = keymap[key]
                self._human_callback(evt)
            return

        # ---- LLM mode with live window (headless=false) ----
        self._ensure_window()
        camera = "third_person" if self._third_person else "first_person"
        frame = self._render_rgb(camera)
        cv2.imshow(self._cv_window, frame[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(self._cv_window, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key in (ord("q"), 27):
            return False

    def tick(self, fps: int = 30) -> None:
        if self._human_callback is not None:
            import time
            time.sleep(1 / fps)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._window_open:
            import cv2
            cv2.destroyWindow(self._cv_window)
            self._window_open = False
        self._renderer.close()
