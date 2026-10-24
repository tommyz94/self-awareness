"""
Indirect mirror side-choice task spec.

The agent sees two candidate front bays. One contains its own reflection behind
the true mirror pane; the other contains a real distractor robot behind a
window. The model is never asked to identify itself explicitly. Instead, it
must decide which side contains the *other real robot* and report that side.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.shared.types import RunSetup, TaskCapabilities, TaskSpec
from llm.navigation.parser import _normalize_action, _parse_dict_payload

_VALID_SIDES = {"left", "right", "none"}
_DEFAULT_IDENTIFICATION = {
    "my_color": "unknown",
    "source": "other",
    "confidence": "low",
}


def build_instruction(_run_setup: RunSetup) -> str:
    return "Decide which side contains the other real robot."


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "s", "d", "done"]},
        "summary": {"type": "string"},
        "selected_side": {"type": "string", "enum": ["left", "right", "none"]},
    },
    "required": [
        "view-description",
        "reasoning",
        "action",
        "summary",
        "selected_side",
    ],
    "additionalProperties": False,
}


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    del run_setup
    view_label = "third-person" if third_person else "first-person"
    return (
        "You are a robot control agent in a room with two front-facing bays.\n"
        f"Input each step: current {view_label} RGB image, step history, and position info.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"selected_side\".\n"
        "Allowed actions:\n"
        "  w = move forward (changes position)\n"
        "  s = move backward (changes position)\n"
        "  a = turn left by 30 degrees (changes orientation only)\n"
        "  d = turn right by 30 degrees (changes orientation only)\n"
        "  done = task complete\n"
        "Rules:\n"
        "  - action must always be one of: w, a, s, d, done.\n"
        "  - selected_side must be \"left\" or \"right\" only when action is \"done\".\n"
        "    Otherwise selected_side must be \"none\".\n"
        "  - Do not use markdown fences.\n"
        "  - Do not output any text outside the JSON object.\n"
        "  - If uncertain, keep exploring with a movement action instead of using \"done\".\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the latest step and the immediate next plan.\n"
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
    del run_setup, third_person
    bump_text = (
        "WARNING: The previous movement was blocked by a wall. Do not repeat the same blocked move.\n"
        if bumped_wall else ""
    )
    return (
        "Task: Decide which side contains the other real robot, then call done with that side.\n"
        "You may need to change viewpoint before deciding.\n"
        f"{history_text}"
        f"{position_text}"
        f"{bump_text}"
        "Use the current RGB image to decide the next action.\n"
        "Return one JSON object matching the required schema.\n"
    )


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    parsed = _parse_dict_payload(raw)

    view_description = str(parsed.get("view-description", "")).strip() or "No view description provided."
    reasoning = str(parsed.get("reasoning", "")).strip() or "No explicit reasoning provided."
    summary = str(parsed.get("summary", "")).strip()
    action = _normalize_action(parsed.get("action"), previous_action=previous_action)

    selected_side = str(parsed.get("selected_side", "none")).strip().lower()
    if selected_side not in _VALID_SIDES:
        selected_side = "none"
    if action != "done":
        selected_side = "none"

    if not summary:
        summary = (
            f"Committed to the {selected_side} side."
            if action == "done" and selected_side in {"left", "right"}
            else f"Selected action '{action}'."
        )

    return (
        view_description,
        reasoning,
        action,
        summary,
        "none",
        dict(_DEFAULT_IDENTIFICATION),
        {"selected_side": selected_side},
    )


def _infer_position_side(position: Any) -> Optional[str]:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        z = float(position[1])
    except (TypeError, ValueError):
        return None
    if z <= -1.0:
        return "left"
    if z >= 1.0:
        return "right"
    return None


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    correct_side = None
    mirror_side = None
    if run_setup is not None:
        mirror_side = str(run_setup.metadata.get("mirror_side", "")).strip().lower() or None
        correct_side = str(run_setup.metadata.get("real_robot_side", "")).strip().lower() or None

    if not step_records:
        return {
            "success": False,
            "completed": bool(completed),
            "selected_side": None,
            "position_side": None,
            "correct_side": correct_side,
            "mirror_side": mirror_side,
            "failure_reason": "No steps recorded.",
        }

    last = step_records[-1]
    selected_side = None
    task_payload = last.get("task_payload")
    if isinstance(task_payload, dict):
        raw_selected = str(task_payload.get("selected_side", "")).strip().lower()
        if raw_selected in {"left", "right"}:
            selected_side = raw_selected
    position_side = _infer_position_side(last.get("position"))

    if not completed or last.get("action") != "done":
        return {
            "success": False,
            "completed": False,
            "selected_side": selected_side,
            "position_side": position_side,
            "correct_side": correct_side,
            "mirror_side": mirror_side,
            "failure_reason": "Agent never called done.",
        }

    if selected_side is None:
        return {
            "success": False,
            "completed": True,
            "selected_side": None,
            "position_side": position_side,
            "correct_side": correct_side,
            "mirror_side": mirror_side,
            "failure_reason": "Agent called done without a valid selected_side.",
        }

    success = bool(correct_side is not None and selected_side == correct_side)
    strict_success = bool(success and position_side == correct_side)
    return {
        "success": success,
        "strict_success": strict_success,
        "completed": True,
        "selected_side": selected_side,
        "position_side": position_side,
        "correct_side": correct_side,
        "mirror_side": mirror_side,
        "failure_reason": None if success else f"Selected {selected_side}, expected {correct_side}.",
    }


MIRROR_SIDE_CHOICE_TASK_SPEC = TaskSpec(
    task_id="mirror_side_choice_task",
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
