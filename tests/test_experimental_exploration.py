from __future__ import annotations

from pathlib import Path

from utils.colors import COLOR_POOL


_EXPECTED_E6_DISTRACTOR_SLOTS = [
    (-3.8, -3.9),
    (-3.8, -2.3),
    (-3.8, -0.7),
    (-3.8, 0.7),
    (-3.8, 2.3),
    (-3.8, 3.9),
]


def test_e6_experiment_preset_sets_auto_defaults():
    preset = Path("conf/experiment/experimental/exploration.yaml")

    assert preset.exists()
    text = preset.read_text()
    assert "experiment_id: experimental/exploration" in text
    assert "num_runs: 100" in text
    assert "nav: auto" in text
    assert "max_steps: 100" in text
    assert "distractors:" not in text


def test_e6_plugin_is_auto_discovered():
    from experiments import get_experiment

    spec = get_experiment("experimental/exploration")

    assert spec.task_id == "explore_task"
    assert spec.experiment_id == "experimental/exploration"


def test_e6_scene_template_contains_six_distractor_placeholders():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    distractors = [obj for obj in scene_template.objects if obj.kind == "distractor_robot"]
    cubes = [obj for obj in scene_template.objects if obj.kind == "cube"]
    assert len(distractors) == 6
    assert len(cubes) == 0


def test_e6_scene_template_uses_widened_centered_mirror_and_spawn_row():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())

    assert scene_template.scene_id == "mirror_room_robot_distractors_no_cubes"
    assert scene_template.spawn_pos == (-10.0, 0.0)
    assert scene_template.mirror_window_z == (-4.5, 4.5)
    assert scene_template.turn_step_deg == 30.0


def test_e6_scene_template_uses_near_mirror_distractor_row_slots():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())
    distractors = [obj for obj in scene_template.objects if obj.kind == "distractor_robot"]

    assert [obj.position for obj in distractors] == _EXPECTED_E6_DISTRACTOR_SLOTS
    assert all(obj.metadata.get("yaw") == 0.0 for obj in distractors)


def test_e6_run_setup_uses_pool_colors_for_agent_and_distractors_only():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    runtime_options = ExperimentRuntimeOptions()
    run_setup = spec.initialize_run(1, runtime_options)

    cubes = [obj for obj in run_setup.scene_spec.objects if obj.kind == "cube"]
    distractors = [obj for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert len(cubes) == 0
    assert 1 <= len(distractors) <= 6
    assert all(obj.metadata.get("reflect_in_mirror") is True for obj in distractors)
    assert all(obj.color in COLOR_POOL for obj in distractors)
    assert run_setup.agent_color_name in COLOR_POOL


def test_e6_distractor_color_mode_excludes_agent_color_when_requested():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    run_setup = spec.initialize_run(
        1,
        ExperimentRuntimeOptions(distractor_color_mode="exclude_agent_color"),
    )
    distractor_colors = [obj.color for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert distractor_colors
    assert run_setup.agent_color_name not in set(distractor_colors)
    assert len(set(distractor_colors)) == len(distractor_colors)


def test_e6_distractor_color_mode_allow_agent_color_keeps_unique_distractor_colors():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
    run_setup = spec.initialize_run(
        1,
        ExperimentRuntimeOptions(distractor_color_mode="allow_agent_color"),
    )
    distractor_colors = [obj.color for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"]

    assert distractor_colors
    assert len(set(distractor_colors)) == len(distractor_colors)


def test_e6_supports_runtime_random_wall_color():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    primary_colors = (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.0),
        (0.0, 0.3, 1.0),
    )

    spec = get_experiment("experimental/exploration")
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


def test_e6_mujoco_builder_contains_distractor_and_mirror_pairs():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/exploration")
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
    assert callable(overrides.mujoco_step_applier)
