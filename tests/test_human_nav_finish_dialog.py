from __future__ import annotations

import json
from pathlib import Path

from env.observation import ObservationMetadata


class _FakeEnv:
    def __init__(self, scripted_keys_by_run: list[list[str]]) -> None:
        self._scripted_keys_by_run = [list(keys) for keys in scripted_keys_by_run]
        self._current_keys: list[str] = []
        self._run_index = -1
        self._position = (-10.0, 2.5)
        self._direction = "N"
        self.bound_callback = None
        self.reset_calls = []
        self.step_calls: list[str] = []
        self.tick_calls = 0
        self.closed = False

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def object_positions_2d(self) -> dict:
        return {
            "red": (-10.0, -4.0),
            "green": (-8.0, -4.0),
            "blue": (-6.0, -4.0),
        }

    def bind_keys(self, callback) -> None:
        self.bound_callback = callback

    def reset(self, run_setup) -> None:
        self._run_index += 1
        self._position = tuple(run_setup.scene_spec.spawn_pos)
        self._direction = "N"
        self._current_keys = list(self._scripted_keys_by_run[self._run_index])
        self.reset_calls.append(run_setup)

    def tick(self, fps: int = 30) -> None:
        self.tick_calls += 1

    def update_frame(self):
        if not self._current_keys:
            return False
        event = type("Evt", (), {"key": self._current_keys.pop(0)})()
        self.bound_callback(event)
        return True

    def step(self, action: str) -> tuple[float, float, str, bool]:
        self.step_calls.append(action)
        x, z = self._position
        if action == "w":
            self._position = (round(x + 1.0, 2), z)
        elif action == "s":
            self._position = (round(x - 1.0, 2), z)
        elif action == "a":
            self._direction = "W"
        elif action == "d":
            self._direction = "E"
        return self._position[0], self._position[1], self._direction, False

    def get_observation_metadata(self) -> ObservationMetadata:
        return ObservationMetadata(
            view_position=(-9.85, 0.9, 2.5),
            view_direction=(1.0, 0.0, 0.0),
            mirror_visible=True,
        )

    def jump(self) -> None:
        return None

    def set_caption(self, text: str) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeLogger:
    def __init__(self, output_file: str) -> None:
        self.output_file = output_file
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def _load_run_json(base_dir: Path, run_id: int) -> dict:
    path = base_dir / "mujoco" / "e1" / "ollama" / "human" / "runs" / f"run{run_id:02d}.json"
    return json.loads(path.read_text())


def test_human_cube_run_submits_finish_dialog_and_exports_json(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv([["w", "f"]])
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)
    monkeypatch.setattr(
        "runtime.controller.prompt_human_cube_selection",
        lambda **_: {
            "selected_cube": "green",
            "reasoning": "I saw a green reflection in the mirror.",
        },
    )

    controller = Controller(
        nav_mode="human",
        experiment_id="e1",
        num_runs=1,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    run_data = _load_run_json(tmp_path / "outputs", 1)
    assert run_data["completed"] is True
    assert run_data["selected_cube"] == "green"
    assert run_data["steps"] == 2
    assert run_data["mirror_before_done"] == 1
    assert [record["action"] for record in run_data["step_records"]] == ["w", "done"]
    assert run_data["step_records"][0]["reasoning"] == ""
    assert run_data["step_records"][1]["view_description"] == ""
    assert run_data["step_records"][1]["reasoning"] == "I saw a green reflection in the mirror."
    assert run_data["step_records"][1]["selected_cube"] == "green"
    assert run_data["step_records"][1]["identification"] == {
        "my_color": "green",
        "source": "other",
        "confidence": "high",
    }


def test_human_finish_dialog_cancel_keeps_navigation_running(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv([["f", "w", "f"]])
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    replies = iter(
        [
            None,
            {
                "selected_cube": "red",
                "reasoning": "The mirror shows a red version of me.",
            },
        ]
    )
    monkeypatch.setattr("runtime.controller.prompt_human_cube_selection", lambda **_: next(replies))

    controller = Controller(
        nav_mode="human",
        experiment_id="e1",
        num_runs=1,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    run_data = _load_run_json(tmp_path / "outputs", 1)
    assert [record["action"] for record in run_data["step_records"]] == ["w", "done"]
    assert run_data["steps"] == 2
    assert fake_env.step_calls == ["w"]


def test_human_cube_runs_advance_to_next_run_after_submit(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv([["f"], ["f"]])
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    replies = iter(
        [
            {"selected_cube": "red", "reasoning": "First run choice."},
            {"selected_cube": "blue", "reasoning": "Second run choice."},
        ]
    )
    monkeypatch.setattr("runtime.controller.prompt_human_cube_selection", lambda **_: next(replies))

    controller = Controller(
        nav_mode="human",
        experiment_id="e1",
        num_runs=2,
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    run01 = _load_run_json(tmp_path / "outputs", 1)
    run02 = _load_run_json(tmp_path / "outputs", 2)
    summary = json.loads(
        (
            tmp_path
            / "outputs"
            / "mujoco"
            / "e1"
            / "ollama"
            / "human"
            / "runs"
            / "experiment_summary.json"
        ).read_text()
    )

    assert len(fake_env.reset_calls) == 2
    assert run01["selected_cube"] == "red"
    assert run02["selected_cube"] == "blue"
    assert summary["num_runs"] == 2


def test_human_cube_mujoco_runs_use_opencv_finish_dialog(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = _FakeEnv([["f"]])
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

    dialog_backends: list[str] = []

    received_choices: list[tuple[str, ...]] = []

    def _fake_prompt_human_cube_selection(*, ui_backend: str, valid_cubes):
        dialog_backends.append(ui_backend)
        received_choices.append(tuple(valid_cubes))
        return {
            "selected_cube": "red",
            "reasoning": "MuJoCo run choice.",
        }

    monkeypatch.setattr("runtime.controller.prompt_human_cube_selection", _fake_prompt_human_cube_selection)

    controller = Controller(
        nav_mode="human",
        experiment_id="e1",
        num_runs=1,
        env_type="mujoco",
        mujoco_headless=False,
        heatmap=False,
        output_dir=str(tmp_path / "outputs"),
    )

    controller.run()

    assert dialog_backends == ["opencv"]
    assert len(received_choices) == 1
    assert len(received_choices[0]) == 3


def test_prompt_human_cube_selection_dispatches_by_ui_backend(monkeypatch):
    from runtime.human_finish_dialog import prompt_human_cube_selection

    calls: list[str] = []

    monkeypatch.setattr(
        "runtime.human_finish_dialog._prompt_human_cube_selection_opencv",
        lambda **_: calls.append("opencv") or {"selected_cube": "green", "reasoning": "opencv"},
    )
    monkeypatch.setattr(
        "runtime.human_finish_dialog._prompt_human_cube_selection_tkinter",
        lambda **_: calls.append("tkinter") or {"selected_cube": "blue", "reasoning": "tk"},
    )

    opencv_result = prompt_human_cube_selection(ui_backend="opencv")
    tkinter_result = prompt_human_cube_selection(ui_backend="tkinter")

    assert opencv_result == {"selected_cube": "green", "reasoning": "opencv"}
    assert tkinter_result == {"selected_cube": "blue", "reasoning": "tk"}
    assert calls == ["opencv", "tkinter"]


def test_opencv_finish_key_uses_only_numeric_cube_shortcuts():
    from runtime.human_finish_dialog import _handle_opencv_finish_key

    selected_cube, reasoning, status_message, submission, cancelled = _handle_opencv_finish_key(
        ord("2"),
        selected_cube="green",
        reasoning="",
    )

    assert selected_cube == "red"
    assert reasoning == ""
    assert status_message == "Red selected."
    assert submission is None
    assert cancelled is False


def test_opencv_finish_key_uses_dynamic_cube_shortcuts():
    from runtime.human_finish_dialog import _handle_opencv_finish_key

    selected_cube, reasoning, status_message, submission, cancelled = _handle_opencv_finish_key(
        ord("2"),
        selected_cube="cyan",
        reasoning="",
        valid_cubes=("cyan", "orange", "teal"),
    )

    assert selected_cube == "orange"
    assert reasoning == ""
    assert status_message == "Orange selected."
    assert submission is None
    assert cancelled is False


def test_opencv_finish_key_treats_letters_as_reasoning_text():
    from runtime.human_finish_dialog import _handle_opencv_finish_key

    selected_cube, reasoning, status_message, submission, cancelled = _handle_opencv_finish_key(
        ord("r"),
        selected_cube="green",
        reasoning="mi",
    )

    assert selected_cube == "green"
    assert reasoning == "mir"
    assert status_message == "Editing reasoning."
    assert submission is None
    assert cancelled is False

    selected_cube, reasoning, status_message, submission, cancelled = _handle_opencv_finish_key(
        ord("q"),
        selected_cube="green",
        reasoning="",
    )

    assert selected_cube == "green"
    assert reasoning == "q"
    assert status_message == "Editing reasoning."
    assert submission is None
    assert cancelled is False


def test_cube_fill_bgr_uses_shared_color_pool_mapping():
    from runtime.human_finish_dialog import _cube_fill_bgr

    assert _cube_fill_bgr("red") == (0, 0, 255)
    assert _cube_fill_bgr("blue") == (255, 76, 0)
    assert _cube_fill_bgr("orange") == (0, 140, 255)
    assert _cube_fill_bgr("unknown-color") == (190, 190, 190)
