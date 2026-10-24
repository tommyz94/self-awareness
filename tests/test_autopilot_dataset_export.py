from __future__ import annotations

import json
from pathlib import Path

from experiments.evaluation.metrics import RunOutcome


class _FakeEnv:
    position = (-10.0, 0.0)
    direction = "N"
    object_positions_2d = {}

    def __init__(self) -> None:
        self.reset_calls = []

    def bind_keys(self, callback) -> None:
        return None

    def reset(self, run_setup) -> None:
        self.reset_calls.append(run_setup)

    def tick(self, fps: int = 30) -> None:
        return None

    def update_frame(self):
        return False

    def capture_image(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpg")
        return str(path)

    def close(self) -> None:
        return None


class _FakeLogger:
    def __init__(self, output_file: str) -> None:
        self.output_file = output_file

    def reset(self) -> None:
        return None


def test_e6_e7_save_dataset_writes_manifest_episode_json_and_rgb_frames(monkeypatch, tmp_path):
    from runtime.controller import Controller

    for experiment_id in ("experimental/exploration", "e5"):
        run_root = tmp_path / experiment_id
        run_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(run_root)
        fake_env = _FakeEnv()
        monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
        monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

        controller = Controller(
            nav_mode="auto",
            experiment_id=experiment_id,
            save_dataset=True,
            num_runs=3,
            max_steps=5,
            mujoco_headless=False,
            output_dir=str(run_root / "outputs"),
        )

        dataset_root = run_root / "Mirror_Video_Dataset"
        manifest_path = dataset_root / "dataset_manifest.json"

        assert controller._dataset_enabled is True
        assert dataset_root.exists()
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["dataset_name"] == "Mirror_Video_Dataset"
        assert manifest["experiment_id"] == experiment_id
        assert manifest["nav_mode"] == "auto"
        assert manifest["num_episodes_target"] == 3
        assert manifest["max_steps_per_episode"] == 5
        assert manifest["autopilot_sleep_sec"] == 0.5

        run_setup = controller.experiment_spec.initialize_run(1, controller.experiment_runtime_options)
        controller.current_run_setup = run_setup
        controller.agent_color_name = run_setup.agent_color_name
        controller.iteration = 2
        controller.trajectory = [(-10.0, 0.0), (-9.7, 0.0)]
        controller.step_records = [
            {
                "step": 1,
                "action": "w",
                "position": [-9.7, 0.0],
                "direction": "N",
                "bumped_wall": False,
                "mirror_visible": False,
                "ego_reflection_visible": False,
            },
            {
                "step": 2,
                "action": "a",
                "position": [-9.7, 0.0],
                "direction": "W",
                "bumped_wall": False,
                "mirror_visible": True,
                "ego_reflection_visible": True,
            },
        ]

        initial_frame = controller._capture_dataset_frame(run_id=1, frame_index=0)
        step_frame = controller._capture_dataset_frame(run_id=1, frame_index=1)

        assert initial_frame == "rgb/000000.jpg"
        assert step_frame == "rgb/000001.jpg"
        assert (dataset_root / "episodes" / "episode_0001" / initial_frame).exists()
        assert (dataset_root / "episodes" / "episode_0001" / step_frame).exists()

        step_details = [
            {
                "step": 1,
                "action": "w",
                "frame": step_frame,
                "position_before": [-10.0, 0.0],
                "position_after": [-9.7, 0.0],
                "target_before_step": [-8.0, 0.0],
                "distance_before": 2.0,
                "distance_after": 1.7,
                "bumped_wall": False,
                "mirror_visible": False,
                "ego_reflection_visible": False,
                "distractors": [],
            }
        ]
        outcome = RunOutcome(
            run_id=1,
            agent_color=controller.agent_color_name,
            wrong_color=None,
            selected_cube=None,
            completed=True,
            steps=controller.iteration,
            experiment_id=experiment_id,
            scene_id=run_setup.scene_spec.scene_id,
            task_id="explore_task",
        )

        controller._export_dataset_episode(
            run_id=1,
            outcome=outcome,
            step_details=step_details,
            initial_frame_relpath=initial_frame,
        )

        episode_json = dataset_root / "episodes" / "episode_0001" / "episode.json"
        assert episode_json.exists()

        payload = json.loads(episode_json.read_text())
        distractor_count = sum(
            1 for obj in run_setup.scene_spec.objects if obj.kind == "distractor_robot"
        )

        assert payload["episode_id"] == 1
        assert payload["experiment_id"] == experiment_id
        assert payload["completed"] is True
        assert payload["steps_executed"] == 2
        assert payload["max_steps"] == 5
        assert payload["actions"] == ["w", "a"]
        assert payload["trajectory"] == [[-10.0, 0.0], [-9.7, 0.0]]
        assert payload["ego_reflection_seen"] is True
        assert payload["agent"]["color"] == controller.agent_color_name
        assert payload["distractors"]["count"] == distractor_count
        assert len(payload["distractors"]["objects"]) == distractor_count
        assert payload["num_agents_total"] == 1 + distractor_count
        assert payload["initial_frame"] == "rgb/000000.jpg"
        assert payload["steps"][0]["frame"] == "rgb/000001.jpg"
        assert payload["steps"][0]["ego_reflection_visible"] is False
        assert payload["step_records"][0]["ego_reflection_visible"] is False
        assert payload["step_records"][1]["ego_reflection_visible"] is True
        assert payload["step_records"][1]["action"] == "a"


def test_save_dataset_is_ignored_outside_supported_auto_experiments(monkeypatch, tmp_path):
    from runtime.controller import Controller

    monkeypatch.chdir(tmp_path)
    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    controller = Controller(
        nav_mode="human",
        experiment_id="e5",
        save_dataset=True,
        mujoco_headless=False,
        output_dir=str(tmp_path / "outputs"),
    )

    assert controller._dataset_enabled is False
    assert not (tmp_path / "Mirror_Video_Dataset").exists()


def test_e6_e7_save_dataset_manifest_sets_zero_sleep_when_headless(monkeypatch, tmp_path):
    from runtime.controller import Controller

    monkeypatch.chdir(tmp_path)
    fake_env = _FakeEnv()
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    for experiment_id in ("experimental/exploration", "e5"):
        run_root = tmp_path / f"headless_{experiment_id}"
        run_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(run_root)
        fake_env = _FakeEnv()
        monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
        monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

        controller = Controller(
            nav_mode="auto",
            experiment_id=experiment_id,
            save_dataset=True,
            num_runs=2,
            max_steps=4,
            mujoco_headless=True,
            output_dir=str(run_root / "outputs"),
        )

        manifest_path = run_root / "Mirror_Video_Dataset" / "dataset_manifest.json"
        assert controller._dataset_enabled is True
        assert controller._autopilot_sleep_sec == 0.0
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["autopilot_sleep_sec"] == 0.0
