import json
import math
import os
from pathlib import Path
from typing import List

from experiments.evaluation.metrics import RunOutcome
from experiments import RunSetup, SceneObjectSpec


def _serialize_scene_object(obj: SceneObjectSpec | dict) -> dict:
    if isinstance(obj, SceneObjectSpec):
        data = {
            "object_id": obj.object_id,
            "kind": obj.kind,
            "position": [float(obj.position[0]), float(obj.position[1])],
        }
        if obj.color is not None:
            data["color"] = obj.color
        if obj.size is not None:
            data["size"] = list(obj.size)
        if obj.asset_id is not None:
            data["asset_id"] = obj.asset_id
        if obj.metadata:
            data["metadata"] = obj.metadata
        return data
    return dict(obj)


def _serialize_run_setup(run_setup: RunSetup | None) -> dict | None:
    if run_setup is None:
        return None
    scene_spec = run_setup.scene_spec
    _seed_str = os.getenv("EXPERIMENT_SEED")
    data = {
        "msr_level": str(os.getenv("MSR_LEVEL", "2")),
        "seed": int(_seed_str) if _seed_str is not None else None,
        "agent_color": run_setup.agent_color_name,
        "scene_id": scene_spec.scene_id,
        "wrong_color": run_setup.wrong_color,
        "spawn_pos": [float(scene_spec.spawn_pos[0]), float(scene_spec.spawn_pos[1])],
        "spawn_yaw": float(scene_spec.spawn_yaw),
        "objects": [_serialize_scene_object(obj) for obj in scene_spec.objects],
    }
    if run_setup.metadata:
        data["metadata"] = dict(run_setup.metadata)
    return data


def save_run_json(
    run_id: int,
    outcome: RunOutcome,
    step_records: list,
    output_dir: str,
    task_output: dict = None,
    run_setup: RunSetup | None = None,
    run_context: dict | None = None,
) -> str:
    path = Path(output_dir) / f"run{run_id:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    context = run_context if run_context is not None else getattr(outcome, "run_context", {})
    run_setup_data = _serialize_run_setup(run_setup)
    if run_setup_data is None and outcome.scene_id is not None:
        run_setup_data = {
            "agent_color": outcome.agent_color,
            "scene_id": outcome.scene_id,
            "wrong_color": outcome.wrong_color,
        }
    data = {
        # Phase 4 generic fields (None when not on experiment path)
        "experiment_id": outcome.experiment_id,
        "scene_id": outcome.scene_id,
        "task_id": outcome.task_id,
        "termination_reason": outcome.termination_reason,
        "run_context": context,
        **(
            {"paper_experiment_id": context["paper_experiment_id"]}
            if context.get("paper_experiment_id") is not None else {}
        ),
        "run_setup": run_setup_data,
        "task_output": task_output if task_output is not None else {},
        # Legacy flat fields (kept for judge / W&B compat)
        "run_id": run_id,
        "agent_color": outcome.agent_color,
        "wrong_color": outcome.wrong_color,
        "selected_cube": outcome.selected_cube,
        "completed": outcome.completed,
        "steps": outcome.steps,
        "trajectory": outcome.trajectory,
        **outcome.mirror_exposure_metrics,
        "step_records": step_records,
    }
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def save_experiment_json(
    outcomes: List[RunOutcome], experiment_id: str, output_dir: str,
    run_context: dict | None = None,
) -> str:
    path = Path(output_dir) / "experiment_summary.json"
    task_id = outcomes[0].task_id if outcomes and outcomes[0].task_id else None
    scene_id = outcomes[0].scene_id if outcomes and outcomes[0].scene_id else None
    from experiments.tasks import resolve_task

    capabilities = resolve_task(
        task_id=task_id or "cube_task",
        experiment_id=experiment_id,
    ).capabilities
    summary = None
    if capabilities.supports_summary_metrics and capabilities.summarize_experiment is not None:
        summary = capabilities.summarize_experiment(outcomes, experiment_id)
    is_summary_task = summary is not None
    _seed_str = os.getenv("EXPERIMENT_SEED")
    data: dict = {
        "experiment_id": experiment_id,
        "experiment_type": experiment_id,  # kept for legacy tooling
        "task_id": task_id,
        "scene_id": scene_id,
        "msr_level": str(os.getenv("MSR_LEVEL", "2")),
        "seed": int(_seed_str) if _seed_str is not None else None,
        "num_runs": len(outcomes),
    }
    context = run_context if run_context is not None else (
        getattr(outcomes[0], "run_context", {}) if outcomes else {}
    )
    if context.get("paper_experiment_id") is not None:
        data["paper_experiment_id"] = context["paper_experiment_id"]
    if context.get("experiment_id_scheme") is not None or context.get("paper_experiment_id") is not None:
        data["run_context"] = context
    if is_summary_task:
        if task_id == "front_identity_task":
            data["accuracy"] = round(summary["accuracy"], 4)
            data["done_rate"] = round(summary["done_rate"], 4)
            data["used_lateral_action_before_done_rate"] = round(summary["used_lateral_action_before_done_rate"], 4)
            data["accuracy_with_lateral"] = (
                round(summary["accuracy_with_lateral"], 4)
                if summary["accuracy_with_lateral"] is not None
                else None
            )
            data["accuracy_without_lateral"] = (
                round(summary["accuracy_without_lateral"], 4)
                if summary["accuracy_without_lateral"] is not None
                else None
            )
        else:
            sia = summary["sia"]
            gi = summary["gi"]
            data["sia"] = round(sia, 4)
            data["gi"] = round(gi, 4) if not math.isnan(gi) else None
    data["runs"] = [
        {
            "run_id": o.run_id,
            "completed": o.completed,
            "steps": o.steps,
            "termination_reason": o.termination_reason,
            **(
                (
                    {
                        "front_identity_variant": (o.task_result or {}).get("front_identity_variant"),
                        "expected_answer": (o.task_result or {}).get("expected_answer"),
                        "actual_answer": (o.task_result or {}).get("actual_answer"),
                        "success": bool((o.task_result or {}).get("success")),
                        "used_lateral_action_before_done": bool((o.task_result or {}).get("used_lateral_action_before_done")),
                        "lateral_action_count_before_done": (o.task_result or {}).get("lateral_action_count_before_done"),
                        "forward_action_count_before_done": (o.task_result or {}).get("forward_action_count_before_done"),
                    }
                    if task_id == "front_identity_task"
                    else {
                        "agent_color": o.agent_color,
                        "wrong_color": o.wrong_color,
                        "selected_cube": o.selected_cube,
                        "correct": o.selected_cube == o.agent_color,
                    }
                )
                if is_summary_task
                else {}
            ),
        }
        for o in outcomes
    ]
    path.write_text(json.dumps(data, indent=2))
    return str(path)
