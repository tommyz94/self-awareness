import json
import math
import os
from pathlib import Path
from typing import List

from experiments.evaluation.metrics import RunOutcome
from experiments.specs import RunSetup, SceneObjectSpec


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
    data = {
        "msr_level": str(os.getenv("MSR_LEVEL", "2")),
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
) -> str:
    path = Path(output_dir) / f"run{run_id:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_experiment_json(outcomes: List[RunOutcome], experiment_id: str, output_dir: str) -> str:
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
    data: dict = {
        "experiment_id": experiment_id,
        "experiment_type": experiment_id,  # kept for legacy tooling
        "task_id": task_id,
        "scene_id": scene_id,
        "msr_level": str(os.getenv("MSR_LEVEL", "2")),
        "num_runs": len(outcomes),
    }
    if is_summary_task:
        sia = summary["sia"]
        gi = summary["gi"]
        data["sia"] = round(sia, 4)
        data["gi"] = round(gi, 4) if not math.isnan(gi) else None
    data["runs"] = [
        {
            "run_id": o.run_id,
            "completed": o.completed,
            "steps": o.steps,
            **(
                {
                    "agent_color": o.agent_color,
                    "wrong_color": o.wrong_color,
                    "selected_cube": o.selected_cube,
                    "correct": o.selected_cube == o.agent_color,
                }
                if is_summary_task else {}
            ),
        }
        for o in outcomes
    ]
    path.write_text(json.dumps(data, indent=2))
    return str(path)
