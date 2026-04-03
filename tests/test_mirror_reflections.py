from __future__ import annotations

from types import SimpleNamespace
import pytest


def test_build_cube_objects_marks_cubes_as_reflectable():
    from experiments.specs.builtin_helpers import build_cube_objects

    cubes = build_cube_objects(
        {
            "red": (-10.0, -4.0),
            "green": (-8.0, -4.0),
            "blue": (-6.0, -4.0),
        }
    )

    assert {cube.color for cube in cubes} == {"red", "green", "blue"}
    assert all(cube.metadata.get("reflect_in_mirror") is True for cube in cubes)


def test_mujoco_two_room_xml_includes_mirror_cubes_only_for_mirror_scenes():
    from env.backends.mujoco import _build_two_room_xml
    from experiments.specs.types import SceneObjectSpec, SceneSpec

    mirror_scene = SceneSpec(
        scene_id="mirror_room",
        mirror_enabled=True,
        objects=[
            SceneObjectSpec("cube_red", "cube", (-10.0, -4.0), color="red"),
            SceneObjectSpec("cube_green", "cube", (-8.0, -4.0), color="green"),
            SceneObjectSpec("cube_blue", "cube", (-6.0, -4.0), color="blue"),
        ],
    )
    no_mirror_scene = SceneSpec(
        scene_id="mirror_room_no_mirror",
        mirror_enabled=False,
        objects=[
            SceneObjectSpec("cube_red", "cube", (-10.0, -4.0), color="red"),
            SceneObjectSpec("cube_green", "cube", (-8.0, -4.0), color="green"),
            SceneObjectSpec("cube_blue", "cube", (-6.0, -4.0), color="blue"),
        ],
    )

    mirror_xml, _, _ = _build_two_room_xml(mirror_scene)
    no_mirror_xml, _, _ = _build_two_room_xml(no_mirror_scene)

    assert 'name="mirror_cube_red_j"' in mirror_xml
    assert 'name="mirror_cube_green_j"' in mirror_xml
    assert 'name="mirror_cube_blue_j"' in mirror_xml
    assert 'name="mirror_cube_red_j"' not in no_mirror_xml
    assert 'name="mirror_cube_green_j"' not in no_mirror_xml
    assert 'name="mirror_cube_blue_j"' not in no_mirror_xml


def test_mujoco_reset_parks_mirrored_cubes_outside_aperture():
    from env.backends.mujoco import MuJoCoEnv
    from experiments.specs.types import RunSetup, SceneObjectSpec, SceneSpec

    parked_positions: list[tuple[int, float, float, float]] = []
    body_positions: list[tuple[int, float, float, float]] = []

    fake_env = SimpleNamespace(
        _x=0.0,
        _z=0.0,
        _yaw=0.0,
        _spawn_y=-1.0,
        _layout_kind="two_room",
        _mirror_present=True,
        _secondary_present=False,
        _mirror_plane_x=-0.1,
        _mirror_aperture_mode="window",
        _mirror_window_z=(-1.0, 1.0),
        _cube_jnt_addrs={"red": 1, "green": 2, "blue": 3},
        _mirror_cube_jnt_addrs={"red": 11, "green": 12, "blue": 13},
        _robot_geom_ids=[],
        _mirror_geom_ids=[],
        _cube_geom_ids={},
        _is_in_mirror_aperture=lambda z: -1.0 <= z <= 1.0,
        _object_pos_2d={},
        _set_color=lambda *args, **kwargs: None,
        _apply_arm_rest_pose=lambda: None,
        _sync_robot=lambda: None,
        _sync_mirror=lambda: None,
        _reset_applier=None,
        _mj=SimpleNamespace(mj_forward=lambda *args, **kwargs: None),
        _model=object(),
        _data=object(),
    )

    def _set_body_pos(addr: int, x: float, y: float, z: float) -> None:
        body_positions.append((addr, x, y, z))

    def _park_body(addr: int) -> None:
        parked_positions.append((addr, 1000.0, 0.0, 0.0))

    fake_env._set_body_pos = _set_body_pos
    fake_env._park_body = _park_body
    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="mirror_room",
            mirror_enabled=True,
            objects=[
                SceneObjectSpec(
                    object_id="cube_red",
                    kind="cube",
                    position=(-10.0, -0.5),
                    color="red",
                    metadata={"reflect_in_mirror": True},
                ),
                SceneObjectSpec(
                    object_id="cube_green",
                    kind="cube",
                    position=(-8.0, 2.5),
                    color="green",
                    metadata={"reflect_in_mirror": True},
                ),
            ],
        ),
    )

    MuJoCoEnv.reset(fake_env, run_setup)

    assert (11, 9.8, -0.75, -0.5) in body_positions
    assert (12, 1000.0, 0.0, 0.0) in parked_positions
    assert (13, 1000.0, 0.0, 0.0) in parked_positions


def test_segmentation_helper_detects_visible_mirror_ego_pixels():
    np = pytest.importorskip("numpy")
    from env.backends.mujoco import _segmentation_has_geom_pixels

    seg = np.zeros((4, 4, 2), dtype=np.int32)
    seg[..., 0] = 5  # object type
    seg[..., 1] = 11  # object id
    seg[1:3, 1:3, 1] = 42

    assert _segmentation_has_geom_pixels(seg, {42}, geom_obj_type=5, min_pixels=1) is True
    assert _segmentation_has_geom_pixels(seg, {42}, geom_obj_type=5, min_pixels=5) is False
    assert _segmentation_has_geom_pixels(seg, {99}, geom_obj_type=5, min_pixels=1) is False
