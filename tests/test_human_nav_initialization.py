from __future__ import annotations

from pathlib import Path

import pytest


class _FakeEnv:
    position = (0.0, 0.0)
    direction = "N"
    object_positions_2d = {}

    def __init__(self) -> None:
        self.reset_calls = []
        self.tick_calls = 0
        self.bound_callback = None

    def bind_keys(self, callback) -> None:
        self.bound_callback = callback

    def reset(self, run_setup) -> None:
        self.reset_calls.append(run_setup)

    def tick(self, fps: int = 30) -> None:
        self.tick_calls += 1

    def update_frame(self):
        return False

    def close(self) -> None:
        return None


class _FakeLogger:
    def __init__(self, output_file: str) -> None:
        self.output_file = output_file
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_human_nav_initializes_single_run_before_loop(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        mujoco_headless=False,
        heatmap=False,
        json_export=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "mirror_room"
    assert run_setup.scene_spec.mirror_enabled is True
    assert {obj.color for obj in run_setup.scene_spec.objects} == {"red", "green", "blue"}
    assert all(obj.metadata.get("reflect_in_mirror") is True for obj in run_setup.scene_spec.objects)


@pytest.mark.parametrize("experiment_id", ["experimental/direct_perception", "e1", "e2", "e3", "experimental/front_identity", "e4", "experimental/exploration", "e5"])
def test_builtin_experiment_outputs_are_scoped_under_experiment_root(monkeypatch, tmp_path, experiment_id):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)
    monkeypatch.setattr("runtime.controller.time.strftime", lambda _fmt: "20260423_120000")
    monkeypatch.setenv("MSR_LEVEL", "2")

    controller = Controller(
        nav_mode="human",
        experiment_id=experiment_id,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    expected_root = (
        tmp_path
        / "outputs"
        / "mujoco"
        / experiment_id.removeprefix("experimental/")
        / "ollama"
        / "human"
        / "level_2"
        / "20260423_120000"
    )
    if experiment_id == "experimental/direct_perception":
        expected_root = expected_root.parent.parent / "default__color" / expected_root.parent.name / expected_root.name
    assert Path(controller.output_dir) == expected_root


def test_single_run_outputs_keep_legacy_root_layout(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)
    monkeypatch.setattr("runtime.controller.time.strftime", lambda _fmt: "20260423_120000")
    monkeypatch.setenv("MSR_LEVEL", "2")

    controller = Controller(
        nav_mode="human",
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    expected_root = (
        tmp_path / "outputs" / "mujoco" / "ollama" / "human" / "level_2" / "20260423_120000"
    )
    assert Path(controller.output_dir) == expected_root


@pytest.mark.parametrize("experiment_id", ["experimental/exploration", "e5"])
def test_auto_nav_dispatches_to_auto_experiment_for_e6_e7(monkeypatch, tmp_path, experiment_id):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    called = {"auto": False}

    def _fake_auto(self):
        called["auto"] = True
        return []

    monkeypatch.setattr(Controller, "_run_auto_experiment", _fake_auto)

    controller = Controller(
        nav_mode="auto",
        experiment_id=experiment_id,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert called["auto"] is True


def test_human_nav_initializes_selected_experiment_before_loop(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id="e4",
        num_distractors=2,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "mirror_room_robot_distractors"
    assert run_setup.scene_spec.mirror_enabled is True
    assert sum(obj.kind == "cube" for obj in run_setup.scene_spec.objects) == 3
    distractor_count = sum(obj.kind == "distractor_robot" for obj in run_setup.scene_spec.objects)
    assert 1 <= distractor_count <= 6


def test_human_nav_initializes_e0_selected_experiment_before_loop(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id="experimental/direct_perception",
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "mirror_room_direct_self_view"
    assert run_setup.scene_spec.layout_kind == "two_room"
    assert run_setup.scene_spec.mirror_enabled is True
    assert run_setup.scene_spec.spawn_pos == (-1.2, 0.0)
    assert run_setup.scene_spec.mirror_window_z == (-7.5, 7.5)
    assert run_setup.scene_spec.objects == []


def test_human_nav_initializes_e6_selected_experiment_before_loop(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id="experimental/exploration",
        num_distractors=2,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "mirror_room_robot_distractors_no_cubes"
    assert run_setup.scene_spec.mirror_enabled is True
    assert sum(obj.kind == "cube" for obj in run_setup.scene_spec.objects) == 0
    distractor_count = sum(obj.kind == "distractor_robot" for obj in run_setup.scene_spec.objects)
    assert 1 <= distractor_count <= 6


def test_human_nav_initializes_e7_selected_experiment_before_loop(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id="e5",
        num_distractors=2,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "mirror_room_robot_distractors_with_random_cubes"
    assert run_setup.scene_spec.mirror_enabled is True
    cube_count = sum(obj.kind == "cube" for obj in run_setup.scene_spec.objects)
    distractor_count = sum(obj.kind == "distractor_robot" for obj in run_setup.scene_spec.objects)
    assert 5 <= cube_count <= 15
    assert 1 <= distractor_count <= 6
def test_human_nav_e4_ignores_use_mirror_override_and_uses_sampled_variant(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)
    monkeypatch.setattr("experiments.experimental.corridor.experiment._sample_variant", lambda: "long")

    controller = Controller(
        nav_mode="human",
        experiment_id="experimental/corridor",
        num_runs=1,
        use_mirror=True,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "narrow_corridor_mirror"
    assert run_setup.scene_spec.layout_kind == "corridor"
    assert run_setup.scene_spec.mirror_enabled is False
    assert run_setup.scene_spec.mirror_aperture_mode == "none"
    assert run_setup.metadata["e4_variant"] == "long"
    assert run_setup.metadata["expected_done_step"] == 20


@pytest.mark.parametrize(
    ("experiment_id", "expected_scene_id", "expected_layout_kind", "expected_mirror"),
    [
        ("e1", "mirror_room", "two_room", True),
        ("e2", "mirror_room_no_mirror", "two_room", False),
        ("e3", "mirror_room", "two_room", True),
    ],
)
def test_human_nav_uses_selected_experiment_scene_for_all_builtin_experiments(
    monkeypatch,
    tmp_path,
    experiment_id,
    expected_scene_id,
    expected_layout_kind,
    expected_mirror,
):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id=experiment_id,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == expected_scene_id
    assert run_setup.scene_spec.layout_kind == expected_layout_kind
    assert run_setup.scene_spec.mirror_enabled is expected_mirror

    assert sum(obj.kind == "cube" for obj in run_setup.scene_spec.objects) == 3


def test_human_nav_e4_run_setup_has_corridor_geometry_and_variant_metadata(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)
    monkeypatch.setattr("experiments.experimental.corridor.experiment._sample_variant", lambda: "mirror")

    controller = Controller(
        nav_mode="human",
        experiment_id="experimental/corridor",
        num_runs=1,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert len(fake_env.reset_calls) == 1
    run_setup = fake_env.reset_calls[0]
    assert run_setup.scene_spec.scene_id == "narrow_corridor_mirror"
    assert run_setup.scene_spec.layout_kind == "corridor"
    assert run_setup.scene_spec.length == 10.0
    assert run_setup.scene_spec.width == 6.0
    assert run_setup.scene_spec.spawn_pos == (-4.5, 0.0)
    assert run_setup.scene_spec.mirror_plane_x == 5.0
    assert run_setup.scene_spec.objects == []
    assert run_setup.metadata["e4_variant"] == "mirror"
    assert run_setup.metadata["expected_done_step"] == 10
