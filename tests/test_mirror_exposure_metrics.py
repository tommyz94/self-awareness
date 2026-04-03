from __future__ import annotations

import json


def _build_two_room_run_setup():
    from experiments.specs.registry import RunSetup
    from experiments.specs.builtin_helpers import build_cube_objects, build_mirror_room_scene

    scene_spec = build_mirror_room_scene(mirror_enabled=True)
    cube_positions = {
        "red": (-10.0, -4.0),
        "green": (-8.0, -4.0),
        "blue": (-6.0, -4.0),
    }
    scene_spec.objects = build_cube_objects(cube_positions)
    return RunSetup(
        agent_color_name="red",
        scene_spec=scene_spec,
        cube_positions=cube_positions,
    )


def test_two_room_mirror_visibility_detects_spawn_facing_view():
    from env.observation import mirror_visible_from_camera
    from experiments.specs.builtin_helpers import build_mirror_room_scene

    visible = mirror_visible_from_camera(
        scene_spec=build_mirror_room_scene(mirror_enabled=True),
        camera_position=(-9.85, 0.9, 2.5),
        camera_forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    )

    assert visible is True


def test_two_room_mirror_visibility_rejects_turned_away_and_out_of_window_views():
    from env.observation import mirror_visible_from_camera
    from experiments.specs.builtin_helpers import build_mirror_room_scene

    scene_spec = build_mirror_room_scene(mirror_enabled=True)

    assert mirror_visible_from_camera(
        scene_spec=scene_spec,
        camera_position=(-9.85, 0.9, 2.5),
        camera_forward=(-1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    ) is False
    assert mirror_visible_from_camera(
        scene_spec=scene_spec,
        camera_position=(-9.85, 0.9, -2.0),
        camera_forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    ) is False


def test_corridor_mirror_visibility_detects_visible_and_not_yet_visible_views():
    from env.observation import mirror_visible_from_camera
    from experiments.specs.builtin_helpers import build_corridor_scene

    scene_spec = build_corridor_scene()

    assert mirror_visible_from_camera(
        scene_spec=scene_spec,
        camera_position=(-9.85, 1.45, 0.0),
        camera_forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    ) is True
    assert mirror_visible_from_camera(
        scene_spec=scene_spec,
        camera_position=(-9.85, 1.45, 0.0),
        camera_forward=(0.0, 0.0, 1.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    ) is False


def test_corridor_without_mirror_never_reports_mirror_visible():
    from env.observation import mirror_visible_from_camera
    from experiments.specs.builtin_helpers import build_corridor_scene

    scene_spec = build_corridor_scene(use_mirror=False)

    assert mirror_visible_from_camera(
        scene_spec=scene_spec,
        camera_position=(-3.85, 1.45, 0.0),
        camera_forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        aspect_ratio=4.0 / 3.0,
    ) is False


def test_extract_mirror_exposure_metrics_handles_missing_visibility_and_supported_experiments():
    from experiments.evaluation.metrics import extract_mirror_exposure_metrics

    step_records = [
        {"step": 1, "action": "w", "mirror_visible": False},
        {"step": 2, "action": "w", "mirror_visible": True},
        {"step": 3, "action": "a", "mirror_visible": True},
        {"step": 4, "action": "w", "mirror_visible": False},
        {"step": 5, "action": "done", "mirror_visible": True},
    ]

    assert extract_mirror_exposure_metrics(step_records, completed=True, experiment_id="e1") == {
        "mirror_before_done": 1,
        "steps_to_first_mirror_view": 2,
        "mirror_revisits": 1,
        "chose_after_mirror": 1,
    }
    assert extract_mirror_exposure_metrics(step_records, completed=True, experiment_id="e4") == {
        "mirror_before_done": 1,
        "steps_to_first_mirror_view": 2,
        "mirror_revisits": 1,
        "chose_after_mirror": 1,
    }
    assert extract_mirror_exposure_metrics(step_records, completed=True, experiment_id="e2") == {}


def test_extract_mirror_exposure_metrics_handles_never_seen_and_incomplete_runs():
    from experiments.evaluation.metrics import extract_mirror_exposure_metrics

    never_seen = [
        {"step": 1, "action": "w", "mirror_visible": False},
        {"step": 2, "action": "w", "mirror_visible": False},
    ]
    incomplete_after_seen = [
        {"step": 1, "action": "w", "mirror_visible": True},
        {"step": 2, "action": "w", "mirror_visible": False},
    ]

    assert extract_mirror_exposure_metrics(never_seen, completed=False, experiment_id="e3") == {
        "mirror_before_done": 0,
        "steps_to_first_mirror_view": None,
        "mirror_revisits": 0,
        "chose_after_mirror": 0,
    }
    assert extract_mirror_exposure_metrics(incomplete_after_seen, completed=False, experiment_id="e4") == {
        "mirror_before_done": 1,
        "steps_to_first_mirror_view": 1,
        "mirror_revisits": 0,
        "chose_after_mirror": 0,
    }


def test_controller_record_step_persists_observation_metadata():
    from runtime.controller import Controller

    controller = Controller.__new__(Controller)
    controller.step_records = []
    controller.iteration = 1
    controller.current_pos = (-8.0, 2.5)
    controller.current_dir = "N"
    controller.bumped_wall = False

    controller._record_step(
        move="w",
        view_description="mirror ahead",
        reasoning="move closer",
        summary="continue",
        identification={"my_color": "unknown", "source": "other", "confidence": "low"},
        inner_reasoning=None,
        view_position=(-9.85, 0.9, 2.5),
        view_direction=(1.0, 0.0, 0.0),
        mirror_visible=True,
    )

    assert controller.step_records == [
        {
            "step": 1,
            "action": "w",
            "position": [-8.0, 2.5],
            "direction": "N",
            "bumped_wall": False,
            "view_description": "mirror ahead",
            "reasoning": "move closer",
            "inner_reasoning": None,
            "summary": "continue",
            "identification": {"my_color": "unknown", "source": "other", "confidence": "low"},
            "selected_cube": None,
            "view_position": [-9.85, 0.9, 2.5],
            "view_direction": [1.0, 0.0, 0.0],
            "mirror_visible": True,
        }
    ]


def test_controller_new_outcome_populates_mirror_metrics_only_for_supported_experiments():
    from runtime.controller import Controller
    from experiments.tasks.cube import CUBE_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller.task_spec = CUBE_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 2
    controller.current_run_setup = _build_two_room_run_setup()

    controller.experiment_id = "e1"
    controller.step_records = [
        {"step": 1, "action": "w", "mirror_visible": False},
        {"step": 2, "action": "done", "mirror_visible": True, "selected_cube": "red", "identification": {}},
    ]
    supported = controller._new_outcome(1, "red", True)
    assert supported.mirror_exposure_metrics == {
        "mirror_before_done": 1,
        "steps_to_first_mirror_view": 2,
        "mirror_revisits": 0,
        "chose_after_mirror": 1,
    }

    controller.experiment_id = "e2"
    controller.step_records = [
        {"step": 1, "action": "done", "mirror_visible": False, "selected_cube": "red", "identification": {}},
    ]
    unsupported = controller._new_outcome(1, "red", True)
    assert unsupported.mirror_exposure_metrics == {}


def test_save_run_json_writes_mirror_metrics_after_trajectory_and_before_step_records(tmp_path):
    from experiments.evaluation.metrics import RunOutcome
    from utils.io.json_export import save_run_json

    outcome = RunOutcome(
        run_id=1,
        agent_color="red",
        wrong_color=None,
        selected_cube="red",
        completed=True,
        steps=2,
        trajectory=[(-8.0, 2.5)],
        experiment_id="e1",
        scene_id="mirror_room",
        task_id="cube_task",
        mirror_exposure_metrics={
            "mirror_before_done": 1,
            "steps_to_first_mirror_view": 1,
            "mirror_revisits": 0,
            "chose_after_mirror": 1,
        },
    )
    step_records = [
        {
            "step": 1,
            "action": "done",
            "position": [-8.0, 2.5],
            "direction": "N",
            "bumped_wall": False,
            "view_description": "mirror ahead",
            "reasoning": "done",
            "inner_reasoning": None,
            "summary": "done",
            "identification": {"my_color": "red", "source": "other", "confidence": "high"},
            "selected_cube": "red",
            "view_position": [-9.85, 0.9, 2.5],
            "view_direction": [1.0, 0.0, 0.0],
            "mirror_visible": True,
        }
    ]

    path = save_run_json(
        1,
        outcome,
        step_records,
        str(tmp_path),
        task_output={"selected_cube": "red"},
        run_setup=_build_two_room_run_setup(),
    )

    text = json.loads(path and open(path, "r", encoding="utf-8").read())
    assert text["mirror_before_done"] == 1
    assert text["steps_to_first_mirror_view"] == 1
    assert text["mirror_revisits"] == 0
    assert text["chose_after_mirror"] == 1
    assert text["step_records"][0]["mirror_visible"] is True

    raw_text = open(path, "r", encoding="utf-8").read()
    assert raw_text.index('"trajectory"') < raw_text.index('"mirror_before_done"')
    assert raw_text.index('"mirror_before_done"') < raw_text.index('"steps_to_first_mirror_view"')
    assert raw_text.index('"steps_to_first_mirror_view"') < raw_text.index('"mirror_revisits"')
    assert raw_text.index('"mirror_revisits"') < raw_text.index('"chose_after_mirror"')
    assert raw_text.index('"chose_after_mirror"') < raw_text.index('"step_records"')


def test_save_run_json_omits_mirror_metrics_for_unsupported_experiment(tmp_path):
    from experiments.evaluation.metrics import RunOutcome
    from utils.io.json_export import save_run_json

    outcome = RunOutcome(
        run_id=1,
        agent_color="red",
        wrong_color=None,
        selected_cube="red",
        completed=True,
        steps=1,
        trajectory=[],
        experiment_id="e2",
        scene_id="mirror_room_no_mirror",
        task_id="cube_task",
    )

    path = save_run_json(
        1,
        outcome,
        [],
        str(tmp_path),
        task_output={"selected_cube": "red"},
        run_setup=_build_two_room_run_setup(),
    )

    data = json.loads(open(path, "r", encoding="utf-8").read())
    assert "mirror_before_done" not in data
    assert "steps_to_first_mirror_view" not in data
    assert "mirror_revisits" not in data
    assert "chose_after_mirror" not in data


def test_save_run_json_preserves_corridor_task_output_shape(tmp_path):
    from experiments.evaluation.metrics import RunOutcome
    from utils.io.json_export import save_run_json
    from experiments.specs.builtin_helpers import build_corridor_run_setup, build_corridor_scene

    run_setup = build_corridor_run_setup(build_corridor_scene(use_mirror=False))
    outcome = RunOutcome(
        run_id=1,
        agent_color="red",
        wrong_color=None,
        selected_cube=None,
        completed=True,
        steps=4,
        trajectory=[(-4.0, 0.0), (-2.0, 0.0), (0.0, 0.0)],
        experiment_id="e4",
        scene_id="narrow_corridor_mirror",
        task_id="corridor_task",
        task_result={
            "success": True,
            "target_zone": "corridor-end",
            "stop_position": [14.0, 0.0],
            "goal_status": {
                "status": "at-goal",
                "reason": "I reached the last row.",
                "confidence": "high",
            },
        },
    )

    path = save_run_json(
        1,
        outcome,
        [],
        str(tmp_path),
        task_output=outcome.task_result,
        run_setup=run_setup,
    )

    data = json.loads(open(path, "r", encoding="utf-8").read())
    assert data["task_output"] == {
        "success": True,
        "target_zone": "corridor-end",
        "stop_position": [14.0, 0.0],
        "goal_status": {
            "status": "at-goal",
            "reason": "I reached the last row.",
            "confidence": "high",
        },
    }
