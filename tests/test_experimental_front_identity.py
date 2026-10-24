from __future__ import annotations

from experiments import ExperimentRuntimeOptions, get_experiment
from experiments.tasks.front_identity import _extract_task_output


def test_e4_front_identity_prompt_mentions_self_vs_figure_ahead():
    spec = get_experiment("experimental/front_identity")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    task = spec.resolve_task()

    system_prompt = task.build_system_prompt(run_setup, third_person=False)
    user_prompt = task.build_user_prompt(
        run_setup,
        history_text="No previous steps.\n",
        position_text="Current position: x=-4.5, z=0.0; facing N\n",
        bumped_wall=False,
        third_person=False,
    )

    assert "front_identity.is_self" in system_prompt.lower()
    assert "decide whether the moving figure ahead is you or not" in user_prompt.lower()
    assert "you may move before deciding" not in user_prompt.lower()
    assert "move diagonally forward-left" in system_prompt.lower()
    assert "move diagonally forward-right" in system_prompt.lower()
    assert "front_identity" in system_prompt
    assert "yes, no, or uncertain" in system_prompt.lower()
    assert "done" in system_prompt.lower()
    assert "uncertain" in system_prompt.lower()
    assert "motion-consistency" not in system_prompt.lower()
    assert "appearance-only" not in system_prompt.lower()


def test_e4_corridor_scene_build_keeps_mirror_pane_and_secondary_body():
    from env.backends.mujoco import _build_corridor_xml

    spec = get_experiment("experimental/front_identity")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    xml, robot_geom_names, mirror_geom_names = _build_corridor_xml(run_setup.scene_spec)

    assert 'name="mirror_pane"' in xml
    assert 'name="mirror_root"' in xml
    assert 'name="wall_end"' in xml
    assert 'name="floor_tile_right_r0_c0"' in xml
    assert robot_geom_names
    assert mirror_geom_names


def test_e4_front_identity_allows_uncertain_before_done_and_tracks_lateral_use():
    spec = get_experiment("experimental/front_identity")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    run_setup.metadata["front_identity_variant"] = "self"

    step_records = [
        {
            "action": "w",
            "task_payload": {
                "front_identity": {
                    "is_self": "uncertain",
                    "confidence": "low",
                    "evidence_source": "distance",
                    "reason": "Too far away.",
                }
            },
        },
        {
            "action": "a",
            "task_payload": {
                "front_identity": {
                    "is_self": "uncertain",
                    "confidence": "low",
                    "evidence_source": "viewpoint",
                    "reason": "Still checking.",
                }
            },
        },
        {
            "action": "done",
            "task_payload": {
                "front_identity": {
                    "is_self": "yes",
                    "confidence": "high",
                    "evidence_source": "matching motion",
                    "reason": "The figure moved like a reflection.",
                }
            },
        },
    ]

    task_output = _extract_task_output(step_records, completed=True, run_setup=run_setup)

    assert task_output["success"] is True
    assert task_output["actual_answer"] == "yes"
    assert task_output["used_lateral_action_before_done"] is True
    assert task_output["lateral_action_count_before_done"] == 1
    assert task_output["forward_action_count_before_done"] == 1
