"""Experimental corridor navigation task."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.shared.types import RunSetup, SceneSpec, TaskCapabilities, TaskSpec
from llm.navigation.parser import _parse_dict_payload, _parse_json_reply

_E4_ROWS_PER_SIDE = 10


def build_instruction(_run_setup: RunSetup) -> str:
    return (
        "Navigate to the corridor end and call done on the final row: "
        "when you are one step away from the mirror wall in mirror mode or one step away from the far wall in long-corridor mode."
    )


_VALID_GOAL_STATUSES = {"at-goal", "not-at-goal", "unknown"}
_VALID_CONFIDENCES = {"low", "medium", "high"}

GOAL_STATUS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": sorted(_VALID_GOAL_STATUSES)},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": sorted(_VALID_CONFIDENCES)},
    },
    "required": ["status", "reason", "confidence"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "d", "done"]},
        "summary": {"type": "string"},
        "goal_status": GOAL_STATUS_SCHEMA,
    },
    "required": ["view-description", "reasoning", "action", "summary", "goal_status"],
    "additionalProperties": False,
}


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    view_label = "third-person" if third_person else "first-person"
    return (
        "You are a robot control agent navigating a corridor.\n"
        f"Input each step: current {view_label} RGB image, step history, and position info.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"goal_status\".\n"
        "Allowed actions:\n"
        "  w = move forward (changes position)\n"
        "  a = move forward-left diagonally (changes position; facing direction stays the same)\n"
        "  d = move forward-right diagonally (changes position; facing direction stays the same)\n"
        "  done = task complete\n"
        "Stop rule:\n"
        "  - mirror mode: call done when you are 1 step away from the mirror wall.\n"
        "  - long mode: call done when you are 1 step away from the far wall.\n"
        "  - You must infer whether the episode is mirror mode or long mode from the visual observations.\n"
        "Rules:\n"
        "  - action must always be one of: w, a, d, done.\n"
        "  - backward is not allowed.\n"
        "  - Do not use markdown fences.\n"
        "  - Do not output any text outside the JSON object.\n"
        "  - goal_status must always be present.\n"
        "  - When action is \"done\", use goal_status to explain why you believe you are at the corridor end.\n"
        "  - When action is not \"done\", set goal_status.status to \"not-at-goal\" and use low confidence unless strong evidence suggests otherwise.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now, how it relates to the task, "
        "and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the last step and the current plan. "
        "Do not rewrite the whole episode.\n"
        "  - goal_status: provide corridor-end status, a short reason, and confidence.\n"
        "Do not invent visual evidence that is not present in the image or provided in the prompt."
    )


def build_user_prompt(
    run_setup: RunSetup,
    *,
    history_text: str,
    position_text: str,
    bumped_wall: bool,
    third_person: bool = False,
) -> str:
    del run_setup, bumped_wall, third_person
    return (
        "Task: reach the final row and stop at the correct step.\n"
        f"{history_text}"
        f"{position_text}"
        "Use the current RGB image to decide the next action.\n"
        "Return one JSON object matching the required schema. "
        "If action is 'done', explain briefly why you believe you are at the corridor end.\n"
    )


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    view_description, reasoning, action, summary, selected_cube, identification = _parse_json_reply(
        raw,
        previous_action,
    )
    if action == "s":
        action = previous_action if previous_action in {"w", "a", "d"} else "w"
    parsed = _parse_dict_payload(raw)
    goal_status = _normalize_goal_status(parsed.get("goal_status"), action=action)
    return (
        view_description,
        reasoning,
        action,
        summary,
        selected_cube,
        identification,
        {"goal_status": goal_status},
    )


def _normalize_goal_status(value: Any, *, action: str) -> Dict[str, str]:
    if action == "done":
        default_status = "unknown"
        default_reason = "No valid corridor-end explanation provided."
    else:
        default_status = "not-at-goal"
        default_reason = "Still navigating toward the corridor end."
    if not isinstance(value, dict):
        value = {}
    status = value.get("status")
    if status not in _VALID_GOAL_STATUSES:
        status = default_status
    reason = str(value.get("reason", "")).strip() or default_reason
    confidence = value.get("confidence")
    if confidence not in _VALID_CONFIDENCES:
        confidence = "low"
    return {
        "status": status,
        "reason": reason,
        "confidence": confidence,
    }


def _stop_zone_target(scene_spec: SceneSpec) -> tuple[str, float, set[float]]:
    tile_len = scene_spec.length / _E4_ROWS_PER_SIDE
    tile_width = scene_spec.width / 3
    if scene_spec.mirror_enabled:
        target_zone = "mirror-end"
        target_x = scene_spec.length / 2 - tile_len / 2
    else:
        target_zone = "long-end"
        target_x = (scene_spec.mirror_plane_x or 0.0) + scene_spec.length - tile_len / 2
    z_targets = {
        round(-scene_spec.width / 2 + tile_width / 2 + lane * tile_width, 2)
        for lane in range(3)
    }
    return target_zone, round(target_x, 2), z_targets


def _resolve_e4_variant(scene_spec: SceneSpec, run_setup: Optional[RunSetup]) -> str:
    metadata = run_setup.metadata if run_setup is not None else {}
    variant = str(metadata.get("e4_variant", "")).strip().lower()
    if variant in {"mirror", "long"}:
        return variant
    return "mirror" if scene_spec.mirror_enabled else "long"


def _resolve_expected_done_step(scene_spec: SceneSpec, run_setup: Optional[RunSetup]) -> int:
    default = 10 if scene_spec.mirror_enabled else 20
    metadata = run_setup.metadata if run_setup is not None else {}
    raw = metadata.get("expected_done_step")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value in {10, 20} else default


def _position_matches_last_row(
    stop_position: Optional[list[float]],
    *,
    target_x: float,
    z_targets: set[float],
    x_tol: float = 0.25,
    z_tol: float = 0.25,
) -> bool:
    if stop_position is None:
        return False
    x, z = stop_position
    if abs(float(x) - float(target_x)) > x_tol:
        return False
    return any(abs(float(z) - lane_z) <= z_tol for lane_z in z_targets)


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    scene_spec = (
        run_setup.scene_spec
        if run_setup is not None
        else SceneSpec(scene_id="narrow_corridor_mirror", mirror_enabled=True, layout_kind="corridor")
    )
    target_zone, target_x, z_targets = _stop_zone_target(scene_spec)
    variant = _resolve_e4_variant(scene_spec, run_setup)
    expected_done_step = _resolve_expected_done_step(scene_spec, run_setup)
    default_goal_status = {
        "status": "unknown",
        "reason": "Agent never called done.",
        "confidence": "low",
    }
    default_output = {
        "success": False,
        "e4_variant": variant,
        "expected_done_step": expected_done_step,
        "actual_done_step": None,
        "step_match": False,
        "position_match": False,
        "terminal_decision": "none",
        "failure_reason": "Agent never called done.",
        "target_zone": target_zone,
        "stop_position": None,
        "goal_status": default_goal_status,
    }
    if not step_records:
        return default_output

    last = step_records[-1]
    position = last.get("position")
    stop_position: Optional[list[float]] = None
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        stop_position = [float(position[0]), float(position[1])]

    if bool(last.get("e4_terminal_violation")):
        return {
            **default_output,
            "actual_done_step": (
                int(last.get("step"))
                if isinstance(last.get("step"), (int, float))
                else None
            ),
            "position_match": bool(last.get("e4_terminal_row") is True),
            "terminal_decision": "move-at-terminal-row",
            "failure_reason": "Movement action used at terminal row; expected done.",
            "stop_position": stop_position,
            "goal_status": _normalize_goal_status(last.get("goal_status"), action=str(last.get("action", "w"))),
        }

    if not completed:
        return default_output

    if last.get("action") != "done":
        return default_output

    goal_status = _normalize_goal_status(last.get("goal_status"), action="done")
    actual_done_step = last.get("step")
    if isinstance(actual_done_step, float):
        actual_done_step = int(actual_done_step)
    if not isinstance(actual_done_step, int):
        actual_done_step = None
    step_match = actual_done_step == expected_done_step
    position_match = _position_matches_last_row(
        stop_position,
        target_x=target_x,
        z_targets=z_targets,
    )
    success = bool(step_match and position_match)

    return {
        "success": success,
        "e4_variant": variant,
        "expected_done_step": expected_done_step,
        "actual_done_step": actual_done_step,
        "step_match": step_match,
        "position_match": position_match,
        "terminal_decision": "done",
        "failure_reason": None if success else "Done called at wrong step and/or outside terminal row.",
        "target_zone": target_zone,
        "stop_position": stop_position,
        "goal_status": goal_status,
    }


CORRIDOR_TASK_SPEC = TaskSpec(
    task_id="corridor_task",
    build_instruction=build_instruction,
    response_schema=RESPONSE_SCHEMA,
    parse_reply=parse_reply,
    build_system_prompt=build_system_prompt,
    build_user_prompt=build_user_prompt,
    capabilities=TaskCapabilities(
        supports_heatmap=False,
        supports_judge=False,
        supports_summary_metrics=False,
        extract_task_output=_extract_task_output,
    ),
)
