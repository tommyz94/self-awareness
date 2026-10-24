from __future__ import annotations

from pathlib import Path

import pytest

from utils.colors import COLOR_POOL


_EXPECTED_E5_DISTRACTOR_SLOTS = [
    (-3.8, -3.9),
    (-3.8, -2.3),
    (-3.8, -0.7),
    (-3.8, 0.7),
    (-3.8, 2.3),
    (-3.8, 3.9),
]


def test_paper_e4_experiment_preset_selects_experiment_only():
    preset = Path("conf/experiment/e4.yaml")

    assert preset.exists()
    text = preset.read_text()
    assert "experiment_id: e4" in text
    assert "distractors:" not in text


def test_paper_e4_plugin_is_auto_discovered():
    from experiments import get_experiment

    spec = get_experiment("e4")

    assert spec.task_id == "cube_task"
    assert spec.experiment_id == "e4"


def test_paper_e4_scene_template_contains_six_distractor_placeholders():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    distractors = [obj for obj in scene_template.objects if obj.kind == "distractor_robot"]
    assert len(distractors) == 6


def test_paper_e4_scene_template_uses_widened_centered_mirror_and_spawn_row():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())

    assert scene_template.scene_id == "mirror_room_robot_distractors"
    assert scene_template.spawn_pos == (-10.0, 0.0)
    assert scene_template.mirror_window_z == (-4.5, 4.5)
    assert scene_template.turn_step_deg == 30.0


def test_paper_e4_scene_template_uses_near_mirror_distractor_row_slots():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    distractors = [obj for obj in scene_template.objects if obj.kind == "distractor_robot"]

    assert [obj.position for obj in distractors] == _EXPECTED_E5_DISTRACTOR_SLOTS
    assert all(obj.metadata.get("yaw") == 0.0 for obj in distractors)


def test_paper_e4_run_setup_uses_pool_colors_for_agent_cubes_and_distractors():
    from experiments.shared.scenes import MIRROR_ROOM_CUBE_CORNER_SLOTS
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    runtime_options = ExperimentRuntimeOptions()
    run_setup = spec.initialize_run(1, runtime_options)

    cubes = [obj for obj in run_setup.scene_spec.objects if obj.kind == "cube"]
    distractors = [obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert len(cubes) == 3
    assert 1 <= len(distractors) <= 6
    assert len({obj.color for obj in cubes}) == 3
    assert all(obj.color in COLOR_POOL for obj in cubes)
    assert all(obj.metadata.get("reflect_in_mirror") is True for obj in distractors)
    mirror_plane_x = run_setup.scene_spec.mirror_plane_x
    assert mirror_plane_x is not None
    reflected_corner_slots = {
        (round(2 * mirror_plane_x - x, 3), z)
        for (x, z) in MIRROR_ROOM_CUBE_CORNER_SLOTS
    }
    assert {tuple(obj.position) for obj in cubes} <= (
        set(MIRROR_ROOM_CUBE_CORNER_SLOTS) | reflected_corner_slots
    )

    distractor_colors = [obj.color for obj in distractors]
    assert all(color in COLOR_POOL for color in distractor_colors)
    assert run_setup.agent_color_name in COLOR_POOL
    assert run_setup.agent_color_name in {obj.color for obj in cubes}


@pytest.mark.parametrize(
    ("mode", "expect_unique", "expect_exclude_agent"),
    [
        ("allow_agent_color", True, False),
        ("exclude_agent_color", True, True),
        ("allow_duplicates", False, False),
    ],
)
def test_paper_e4_distractor_color_modes(mode: str, expect_unique: bool, expect_exclude_agent: bool):
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions(distractor_color_mode=mode))
    distractor_colors = [obj.color for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert distractor_colors
    assert all(color in COLOR_POOL for color in distractor_colors)
    if expect_unique:
        assert len(set(distractor_colors)) == len(distractor_colors)
    if expect_exclude_agent:
        assert run_setup.agent_color_name not in set(distractor_colors)


def test_paper_e4_allow_duplicates_mode_can_generate_duplicate_distractor_colors(monkeypatch):
    import experiments.shared.distractors as e5_experiment

    monkeypatch.setattr(e5_experiment._random, "choice", lambda seq: seq[0])
    colors = e5_experiment._sample_distractor_colors(
        count=4,
        agent_color="red",
        color_mode="allow_duplicates",
    )

    assert colors == [colors[0], colors[0], colors[0], colors[0]]


def test_paper_e4_distractor_color_mode_validation():
    import experiments.shared.distractors as e5_experiment

    with pytest.raises(ValueError, match="Unsupported distractor_color_mode"):
        e5_experiment._sample_distractor_colors(
            count=2,
            agent_color="red",
            color_mode="invalid-mode",
        )


def test_paper_e4_supports_runtime_random_wall_color():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    primary_colors = (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.0),
        (0.0, 0.3, 1.0),
    )

    spec = get_experiment("e4")
    run_setup = spec.initialize_run(
        1,
        ExperimentRuntimeOptions(random_wall_color=True),
    )
    rgba = run_setup.scene_spec.wall_color_rgba

    assert rgba is not None
    assert rgba[3] == 1.0
    rgb = rgba[:3]
    assert min(
        ((rgb[0] - c[0]) ** 2 + (rgb[1] - c[1]) ** 2 + (rgb[2] - c[2]) ** 2) ** 0.5
        for c in primary_colors
    ) >= 0.55


def test_paper_e4_distractors_are_placed_nearer_to_the_mirror_than_spawn():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    mirror_x = run_setup.scene_spec.mirror_plane_x
    spawn_x = run_setup.scene_spec.spawn_pos[0]
    distractors = [obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert mirror_x is not None
    spawn_distance = abs(spawn_x - mirror_x)
    assert distractors
    assert all(abs(obj.position[0] - mirror_x) < spawn_distance for obj in distractors)


def test_paper_e4_distractors_stay_inside_the_enlarged_mirror_opening():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    z_window = run_setup.scene_spec.mirror_window_z
    distractors = [obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert z_window is not None
    assert distractors
    assert all(z_window[0] <= obj.position[1] <= z_window[1] for obj in distractors)


def test_paper_e4_mujoco_builder_contains_distractor_and_mirror_pairs():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    runtime_options = ExperimentRuntimeOptions()
    scene_template = spec.resolve_scene_template(runtime_options)
    overrides = spec.resolve_backend_overrides(runtime_options, scene_template)

    result = overrides.mujoco_scene_builder(scene_template, 640, 480)

    assert 'name="distractor_01_root"' in result.xml
    assert 'name="distractor_06_root"' in result.xml
    assert 'name="mirror_distractor_01_root"' in result.xml
    assert 'name="mirror_distractor_06_root"' in result.xml
    assert any(name.startswith("distractor_01_") for name in result.robot_geom_names)
    assert any(name.startswith("mirror_distractor_01_") for name in result.mirror_geom_names)
    assert 'size="0.100 3.500 4.500"' in result.xml
    assert callable(overrides.mujoco_step_applier)


def test_paper_e4_mirror_window_contains_all_shared_cube_slots():
    from env.base import CUBE_SLOTS
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("e4")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    z_window = scene_template.mirror_window_z

    assert z_window is not None
    assert all(z_window[0] <= z <= z_window[1] for _, z in CUBE_SLOTS)


def test_paper_e4_mujoco_reset_parks_mirrored_distractors_outside_aperture():
    from experiments.shared.distractors import _apply_mujoco_reset
    from experiments.shared.types import RunSetup, SceneObjectSpec, SceneSpec

    placed = []
    parked = []

    class _FakeEnv:
        _spawn_y = -1.0
        _mirror_plane_x = -0.1

        def _jnt_addr_if_exists(self, name: str):
            return name

        def _set_body_pose(self, jnt_addr, x, y, z, yaw):
            placed.append((jnt_addr, x, y, z, yaw))

        def _set_color(self, *args, **kwargs):
            return None

        def _geom_ids_with_prefix(self, prefix: str):
            return [prefix]

        def _apply_arm_rest_pose_for_prefix(self, prefix: str):
            return None

        def _is_in_mirror_aperture(self, z: float) -> bool:
            return -1.0 <= z <= 1.0

        def _park_body(self, jnt_addr):
            parked.append(jnt_addr)

    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="mirror_room_robot_distractors",
            mirror_enabled=True,
            mirror_window_z=(-1.0, 1.0),
            objects=[
                SceneObjectSpec(
                    object_id="distractor_01",
                    kind="distractor_robot",
                    position=(-3.8, -0.9),
                    color="blue",
                    metadata={"yaw": 0.0, "reflect_in_mirror": True},
                ),
                SceneObjectSpec(
                    object_id="distractor_02",
                    kind="distractor_robot",
                    position=(-3.8, 2.7),
                    color="blue",
                    metadata={"yaw": 0.0, "reflect_in_mirror": True},
                ),
            ],
        ),
    )

    _apply_mujoco_reset(_FakeEnv(), run_setup)

    assert any(entry[0] == "mirror_distractor_01_root" for entry in placed)
    assert "mirror_distractor_02_root" in parked


def test_paper_e4_mujoco_step_moves_distractor_toward_target_and_updates_mirror():
    from experiments.shared.distractors import _apply_mujoco_step

    poses: list[tuple[str, float, float, float, float]] = []
    parked: list[str] = []

    class _FakeEnv:
        _spawn_y = -1.0
        _mirror_plane_x = -0.1

        def _set_body_pose(self, jnt_addr, x, y, z, yaw):
            poses.append((jnt_addr, x, y, z, yaw))

        def _is_in_mirror_aperture(self, z: float) -> bool:
            return -10.0 <= z <= 10.0

        def _park_body(self, jnt_addr):
            parked.append(jnt_addr)

    env = _FakeEnv()
    env._e5_distractor_nav = [{
        "object_id": "distractor_01",
        "root_addr": "distractor_01_root",
        "mirror_root_addr": "mirror_distractor_01_root",
        "x": -3.8,
        "z": 0.0,
        "yaw": 0.0,
        "turn_step": 0.5235987755982988,
        "target_x": -7.0,
        "target_z": 0.0,
        "bounds": (-14.0, -0.9, -6.8, 6.8),
        "last_action": "none",
    }]

    _apply_mujoco_step(env, "w")

    state = env._e5_distractor_nav[0]
    assert state["last_action"] in {"a", "d"}
    assert float(state["x"]) == pytest.approx(-3.8, abs=1e-6)
    assert abs(float(state["z"])) < 1e-6
    assert any(pose[0] == "distractor_01_root" for pose in poses)
    assert any(pose[0] == "mirror_distractor_01_root" for pose in poses)
    assert parked == []


def test_paper_e4_run_setup_randomizes_distractor_count_between_1_and_6(monkeypatch):
    import experiments.shared.distractors as e5_experiment
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    values = iter([1, 6, 4])
    monkeypatch.setattr(e5_experiment._random, "randint", lambda _a, _b: next(values))

    spec = get_experiment("e4")
    counts = []
    for run_id in (1, 2, 3):
        run_setup = spec.initialize_run(run_id, ExperimentRuntimeOptions())
        counts.append(sum(obj.kind == "distractor_robot" for obj in run_setup.scene_spec.objects))

    assert counts == [1, 6, 4]
