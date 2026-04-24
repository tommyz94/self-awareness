"""
Embodied self-description exploration task spec (e8).

The agent explores the mirrored room until it has enough evidence to describe
its own visible appearance in detail. Cubes and distractors can remain in the
scene as clutter, but they are not the goal.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.specs.types import RunSetup, TaskCapabilities, TaskSpec
from experiments.tasks.self_perception import (
    _DEFAULT_IDENTIFICATION,
    _DEFAULT_SELF_DESCRIPTION,
    _normalize_self_description,
)
from llm.navigation.parser import _normalize_action, _parse_dict_payload

_VALID_ACTIONS = ["w", "a", "s", "d", "done"]

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": _VALID_ACTIONS},
        "summary": {"type": "string"},
        "self_description": {
            "type": "object",
            "properties": {
                "overall": {"type": "string"},
                "clothing": {"type": "string"},
                "body_and_skin": {"type": "string"},
                "pose_and_orientation": {"type": "string"},
                "distinguishing_features": {"type": "string"},
                "source": {
                    "type": "string",
                    "enum": ["mirror-direct-view", "guessing", "told-in-prompt", "other"],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": [
                "overall",
                "clothing",
                "body_and_skin",
                "pose_and_orientation",
                "distinguishing_features",
                "source",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "view-description",
        "reasoning",
        "action",
        "summary",
        "self_description",
    ],
    "additionalProperties": False,
}


def build_instruction(_run_setup: RunSetup) -> str:
    return (
        "Explore the room until you can identify yourself in the mirror, then "
        "give a detailed description of your own visible appearance."
    )


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    del run_setup
    view_label = "third-person" if third_person else "first-person"
    return (
        "You are a robot control agent navigating a mirrored room with moving distractor figures.\n"
        f"Input each step: current {view_label} RGB image, step history, and position info.\n"
        "Your objective is to identify yourself via mirror evidence and then provide a detailed "
        "description of your own visible appearance.\n"
        "Colored cubes may be present in the environment, but they are not the task goal.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"self_description\".\n"
        "Allowed actions:\n"
        "  w = move forward (changes position)\n"
        "  s = move backward (changes position)\n"
        "  a = turn left by 30 degrees (changes orientation only)\n"
        "  d = turn right by 30 degrees (changes orientation only)\n"
        "  done = task complete\n"
        "Rules:\n"
        "  - action must always be one of: w, a, s, d, done.\n"
        "  - Do not use markdown fences.\n"
        "  - Do not output any text outside the JSON object.\n"
        "  - self_description must always be present.\n"
        "  - Use action \"done\" only when you have enough mirror evidence to confidently describe yourself.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the latest step and the immediate next plan.\n"
        "  - self_description.overall: 1-2 sentences describing your full visible appearance.\n"
        "  - self_description.clothing: describe visible garments, outfit components, and materials.\n"
        "  - self_description.body_and_skin: describe visible body, skin, hair, or face cues if visible.\n"
        "  - self_description.pose_and_orientation: describe your posture, pose, and orientation.\n"
        "  - self_description.distinguishing_features: mention notable visual details that would help identify you again.\n"
        "  - self_description.source: use \"mirror-direct-view\" when the description comes from your reflection.\n"
        "  - self_description.confidence: one of low, medium, or high.\n"
        "When action is not \"done\", keep self_description conservative; use empty strings if you do not yet know enough.\n"
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
        "WARNING: Previous movement was blocked by a wall. Do not repeat the same forward move; "
        "turn, step back, or scan instead.\n"
        if bumped_wall
        else ""
    )
    return (
        "Task: Explore until you can describe your OWN visible appearance using mirror evidence, then finish.\n"
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
    action = _normalize_action(parsed.get("action"), previous_action=previous_action)
    summary = str(parsed.get("summary", "")).strip()
    if not summary:
        summary = (
            "Provided a final self-description."
            if action == "done"
            else f"Selected action '{action}'."
        )

    self_description = _normalize_self_description(parsed.get("self_description"))

    return (
        view_description,
        reasoning,
        action,
        summary,
        None,
        dict(_DEFAULT_IDENTIFICATION),
        {"self_description": self_description},
    )


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    del run_setup
    if not step_records:
        return {
            "success": False,
            "completed": bool(completed),
            "steps": 0,
            "self_description": dict(_DEFAULT_SELF_DESCRIPTION),
            "failure_reason": "Agent did not produce any step record.",
        }

    considered = step_records
    if completed:
        done_index = next(
            (index for index, record in enumerate(step_records) if record.get("action") == "done"),
            None,
        )
        if done_index is not None:
            considered = step_records[: done_index + 1]

    last = considered[-1]
    task_payload = last.get("task_payload") or {}
    self_description = _normalize_self_description(task_payload.get("self_description"))

    detail_fields = (
        "overall",
        "clothing",
        "body_and_skin",
        "pose_and_orientation",
        "distinguishing_features",
    )
    detail_count = sum(1 for key in detail_fields if len(self_description[key]) >= 8)
    sufficiently_detailed = detail_count >= 4 and len(self_description["overall"]) >= 24
    saw_self_in_mirror = any(bool(record.get("ego_reflection_visible", False)) for record in considered)
    if not saw_self_in_mirror:
        saw_self_in_mirror = any(bool(record.get("mirror_visible", False)) for record in considered)
    source_ok = self_description.get("source") == "mirror-direct-view"

    if not completed or last.get("action") != "done":
        return {
            "success": False,
            "completed": False,
            "steps": len(step_records),
            "self_description": self_description,
            "saw_self_in_mirror": bool(saw_self_in_mirror),
            "failure_reason": "Agent did not terminate with a self-description.",
        }

    failure_reason = None
    if not saw_self_in_mirror:
        failure_reason = "Agent finished without clear mirror evidence of itself."
    elif not sufficiently_detailed:
        failure_reason = "Agent did not provide a sufficiently detailed self-description."
    elif not source_ok:
        failure_reason = "Agent did not attribute the description to direct mirror evidence."

    return {
        "success": failure_reason is None,
        "completed": True,
        "termination": "done",
        "steps": len(considered),
        "self_description": self_description,
        "saw_self_in_mirror": bool(saw_self_in_mirror),
        "failure_reason": failure_reason,
    }


SELF_DESCRIPTION_EXPLORE_TASK_SPEC = TaskSpec(
    task_id="self_description_explore_task",
    build_instruction=build_instruction,
    response_schema=RESPONSE_SCHEMA,
    parse_reply=parse_reply,
    build_system_prompt=build_system_prompt,
    build_user_prompt=build_user_prompt,
    capabilities=TaskCapabilities(
        supports_heatmap=True,
        supports_judge=False,
        supports_summary_metrics=False,
        extract_task_output=_extract_task_output,
    ),
)
