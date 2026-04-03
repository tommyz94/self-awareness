from __future__ import annotations

from pathlib import Path

from utils.colors import COLOR_POOL


_EXPECTED_E7_DISTRACTOR_SLOTS = [
    (-3.8, -3.9),
    (-3.8, -2.3),
    (-3.8, -0.7),
    (-3.8, 0.7),
    (-3.8, 2.3),
    (-3.8, 3.9),
]


def test_e7_experiment_preset_sets_auto_defaults():
    preset = Path("conf/experiment/e7.yaml")

    assert preset.exists()
    text = preset.read_text()
    assert "experiment_id: e7" in text
    assert "num_runs: 100" in text
    assert "nav: auto" in text
    assert "max_steps: 100" in text
    assert "save_dataset: false" in text
    assert "e7_random_cube_heights: true" in text
    assert "distractors:" not in text


def test_e7_plugin_is_auto_discovered():
    from experiments.specs import get_experiment

    spec = get_experiment("e7")

    assert spec.task_id == "explore_task"
    assert spec.experiment_id == "e7"


def test_e7_scene_template_contains_six_distractor_placeholders_and_no_cubes():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e7")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    distractors = [obj for obj in scene_template.objects if obj.kind == "distractor_robot"]
    cubes = [obj for obj in scene_template.objects if obj.kind == "cube"]

    assert scene_template.scene_id == "mirror_room_robot_distractors_with_random_cubes"
    assert scene_template.spawn_pos == (-10.0, 0.0)
    assert scene_template.mirror_window_z == (-4.5, 4.5)
    assert scene_template.turn_step_deg == 30.0
    assert len(distractors) == 6
    assert [obj.position for obj in distractors] == _EXPECTED_E7_DISTRACTOR_SLOTS
    assert len(cubes) == 0


def test_e7_run_setup_uses_random_distractors_and_random_cubes_from_pool():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e7")
    runtime_options = ExperimentRuntimeOptions()
    run_setup = spec.initialize_run(1, runtime_options)

    scene = run_setup.scene_spec
    mirror_plane_x = (
        float(scene.mirror_plane_x)
        if scene.mirror_plane_x is not None
        else -scene.wall_thickness / 2
    )
    cubes = [obj for obj in scene.objects if obj.kind == "cube"]
    distractors = [obj for obj in scene.objects if obj.kind == "distractor_robot"]

    assert run_setup.agent_color_name in COLOR_POOL
    assert 1 <= len(distractors) <= 6
    assert 5 <= len(cubes) <= 15
    assert all(obj.color in COLOR_POOL for obj in distractors)
    assert all(obj.color in COLOR_POOL for obj in cubes)
    assert all(obj.metadata.get("reflect_in_mirror") is True for obj in cubes)

    left = sum(1 for obj in cubes if obj.position[0] < mirror_plane_x)
    right = sum(1 for obj in cubes if obj.position[0] > mirror_plane_x)
    assert left >= 1
    assert right >= 1

    x_min = -scene.length - scene.wall_thickness
    x_max = scene.length + scene.wall_thickness
    z_min = -scene.width / 2
    z_max = scene.width / 2
    for cube in cubes:
        x, z = cube.position
        assert x_min <= x <= x_max
        assert z_min <= z <= z_max


def test_e7_run_setup_supports_random_cube_stacks_and_disable_toggle():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e7")

    stacked_setup = spec.initialize_run(
        1,
        ExperimentRuntimeOptions(e7_random_cube_heights=True),
    )
    stacked_cubes = [obj for obj in stacked_setup.scene_spec.objects if obj.kind == "cube"]
    assert 5 <= len(stacked_cubes) <= 75
    assert all(obj.metadata.get("stack_base_id") for obj in stacked_cubes)
    assert all(1 <= int(obj.metadata.get("stack_level", 0)) <= 5 for obj in stacked_cubes)
    assert all(1 <= int(obj.metadata.get("stack_height", 0)) <= 5 for obj in stacked_cubes)
    assert all(
        int(obj.metadata.get("stack_level", 0)) <= int(obj.metadata.get("stack_height", 0))
        for obj in stacked_cubes
    )

    flat_setup = spec.initialize_run(
        2,
        ExperimentRuntimeOptions(e7_random_cube_heights=False),
    )
    flat_cubes = [obj for obj in flat_setup.scene_spec.objects if obj.kind == "cube"]
    assert 5 <= len(flat_cubes) <= 15
    assert all(int(obj.metadata.get("stack_level", 0)) == 1 for obj in flat_cubes)
    assert all(int(obj.metadata.get("stack_height", 0)) == 1 for obj in flat_cubes)


def test_e7_supports_runtime_random_wall_color():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    primary_colors = (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.0),
        (0.0, 0.3, 1.0),
    )

    spec = get_experiment("e7")
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


def test_e7_mujoco_builder_contains_cube_slots_and_distractor_pairs():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e7")
    runtime_options = ExperimentRuntimeOptions()
    scene_template = spec.resolve_scene_template(runtime_options)
    overrides = spec.resolve_backend_overrides(runtime_options, scene_template)

    result = overrides.mujoco_scene_builder(scene_template, 640, 480)

    assert 'name="distractor_01_root"' in result.xml
    assert 'name="mirror_distractor_01_root"' in result.xml
    assert 'name="cube_red_j"' in result.xml
    assert 'name="mirror_cube_red_j"' in result.xml
    assert 'name="cube_slot_15_j"' in result.xml
    assert 'name="mirror_cube_slot_15_j"' in result.xml
    assert 'name="cube_slot_75_j"' in result.xml
    assert 'name="mirror_cube_slot_75_j"' in result.xml
    assert any(name.startswith("distractor_01_") for name in result.robot_geom_names)
    assert any(name.startswith("mirror_distractor_01_") for name in result.mirror_geom_names)
    assert callable(overrides.mujoco_step_applier)
