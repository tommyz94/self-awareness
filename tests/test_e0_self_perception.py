from __future__ import annotations

import math
from pathlib import Path

from utils.colors import COLOR_POOL, COLOR_RGB


def _build_e0_run_setup(agent_color: str = "red"):
    from experiments.specs.types import RunSetup, SceneSpec

    return RunSetup(
        agent_color_name=agent_color,
        scene_spec=SceneSpec(
            scene_id="mirror_room_direct_self_view",
            mirror_enabled=True,
            spawn_pos=(-1.2, 0.0),
            spawn_yaw=0.0,
            mirror_window_z=(-7.5, 7.5),
            objects=[],
        ),
    )


def test_e0_experiment_preset_sets_single_step_defaults():
    preset = Path("conf/experiment/e0.yaml")

    assert preset.exists()
    text = preset.read_text()
    assert "experiment_id: e0" in text
    assert "num_runs: 10" in text
    assert "max_steps: 1" in text


def test_e0_plugin_is_auto_discovered():
    from experiments.specs import get_experiment

    spec = get_experiment("e0")

    assert spec.experiment_id == "e0"
    assert spec.task_id == "self_perception_task"


def test_e0_scene_template_is_direct_full_width_mirror_view():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e0")
    scene_template = spec.resolve_scene_template(ExperimentRuntimeOptions())

    assert scene_template.scene_id == "mirror_room_direct_self_view"
    assert scene_template.mirror_enabled is True
    assert scene_template.layout_kind == "two_room"
    assert scene_template.spawn_pos == (-1.2, 0.0)
    assert scene_template.spawn_yaw == 0.0
    assert scene_template.mirror_window_z == (-7.5, 7.5)
    assert scene_template.objects == []


def test_e0_full_width_mirror_scene_omits_zero_length_central_walls():
    from env.backends.mujoco import _build_scene_result

    scene_build = _build_scene_result(_build_e0_run_setup().scene_spec)
    xml = scene_build.xml

    assert 'name="mirror_pane"' in xml
    assert 'name="cwall_l"' not in xml
    assert 'name="cwall_r"' not in xml


def test_e0_run_setup_has_no_scene_objects_and_supports_random_wall_color():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e0")

    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    assert run_setup.agent_color_name in COLOR_POOL
    assert run_setup.wrong_color is None
    assert run_setup.scene_spec.objects == []
    assert run_setup.metadata["setup"] == "direct-mirror-self-perception"
    assert run_setup.metadata["self_perception_mode"] == "color"
    assert run_setup.scene_spec.wall_color_rgba is None

    randomized = spec.initialize_run(1, ExperimentRuntimeOptions(random_wall_color=True))
    rgba = randomized.scene_spec.wall_color_rgba
    assert rgba is not None
    assert rgba[3] == 1.0
    agent_rgb = COLOR_RGB[randomized.agent_color_name]
    assert math.sqrt(sum((rgba[i] - agent_rgb[i]) ** 2 for i in range(3))) >= 0.55


def test_e0_run_setup_accepts_description_mode():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e0")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions(self_perception_mode="description"))

    assert run_setup.metadata["self_perception_mode"] == "description"


def test_get_next_move_supports_self_perception_task(monkeypatch):
    from experiments.tasks.self_perception import SELF_PERCEPTION_TASK_SPEC
    from llm.navigation.service import get_next_move

    monkeypatch.setattr(
        "llm.navigation.service.chat",
        lambda **_kwargs: (
            """{
                "view-description": "I can see my reflection directly in the mirror.",
                "reasoning": "The mirrored robot body is clearly red, so my body color is red.",
                "action": "done",
                "summary": "My body color is red.",
                "identification": {
                    "my_color": "red",
                    "source": "mirror-direct-view",
                    "confidence": "high"
                }
            }""",
            None,
        ),
    )

    result = get_next_move(
        model="dummy",
        task_spec=SELF_PERCEPTION_TASK_SPEC,
        run_setup=_build_e0_run_setup("red"),
        history=[],
        current_pos=(-1.2, 0.0),
        client_type="ollama",
    )

    assert result[2] == "done"
    assert result[4] is None
    assert result[6] == {
        "my_color": "red",
        "source": "mirror-direct-view",
        "confidence": "high",
    }
    assert result[7] == {}


def test_self_perception_task_output_marks_match_and_mismatch():
    from experiments.tasks.self_perception import SELF_PERCEPTION_TASK_SPEC

    run_setup = _build_e0_run_setup("red")
    extract = SELF_PERCEPTION_TASK_SPEC.capabilities.extract_task_output

    success = extract(
        [
            {
                "action": "done",
                "identification": {
                    "my_color": "red",
                    "source": "mirror-direct-view",
                    "confidence": "high",
                },
            }
        ],
        True,
        run_setup,
    )
    assert success == {
        "mode": "color",
        "success": True,
        "reported_color": "red",
        "identification": {
            "my_color": "red",
            "source": "mirror-direct-view",
            "confidence": "high",
        },
        "failure_reason": None,
    }

    failure = extract(
        [
            {
                "action": "done",
                "identification": {
                    "my_color": "blue",
                    "source": "guessing",
                    "confidence": "low",
                },
            }
        ],
        True,
        run_setup,
    )
    assert failure["success"] is False
    assert failure["reported_color"] == "blue"
    assert failure["failure_reason"] == "Reported self color did not match the agent body color."


def test_self_perception_description_mode_extracts_structured_description():
    from experiments.tasks.self_perception import SELF_PERCEPTION_TASK_SPEC, build_instruction

    run_setup = _build_e0_run_setup("red")
    run_setup.metadata["self_perception_mode"] = "description"
    extract = SELF_PERCEPTION_TASK_SPEC.capabilities.extract_task_output

    assert "detailed description" in build_instruction(run_setup).lower()

    result = extract(
        [
            {
                "action": "done",
                "task_payload": {
                    "self_description": {
                        "overall": "I appear as a clothed adult human figure with a clear mirrored full-body view.",
                        "clothing": "I am wearing a dark jacket, long pants, and visible shoes.",
                        "body_and_skin": "My skin is visible on the face and hands, and I have visible hair.",
                        "pose_and_orientation": "I am standing upright and facing the mirror directly.",
                        "distinguishing_features": "The outfit is layered and the silhouette is clearly human-like.",
                        "source": "mirror-direct-view",
                        "confidence": "high",
                    }
                },
            }
        ],
        True,
        run_setup,
    )

    assert result["success"] is True
    assert result["mode"] == "description"
    assert result["self_description"]["source"] == "mirror-direct-view"
    assert result["failure_reason"] is None


def test_controller_new_outcome_for_e0_uses_task_result_and_mirror_metrics():
    from experiments.tasks.self_perception import SELF_PERCEPTION_TASK_SPEC
    from runtime.controller import Controller

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = SELF_PERCEPTION_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 1
    controller.experiment_id = "e0"
    controller.current_run_setup = _build_e0_run_setup("red")
    controller.step_records = [
        {
            "step": 1,
            "action": "done",
            "mirror_visible": True,
            "identification": {
                "my_color": "red",
                "source": "mirror-direct-view",
                "confidence": "high",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, True)

    assert outcome.task_result == {
        "mode": "color",
        "success": True,
        "reported_color": "red",
        "identification": {
            "my_color": "red",
            "source": "mirror-direct-view",
            "confidence": "high",
        },
        "failure_reason": None,
    }
    assert outcome.mirror_exposure_metrics == {
        "mirror_before_done": 1,
        "steps_to_first_mirror_view": 1,
        "mirror_revisits": 0,
        "chose_after_mirror": 1,
    }
