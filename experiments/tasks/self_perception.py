"""
Experimental direct mirror self-perception task.

The agent starts with a direct, unobstructed mirror view and should report its
own body color immediately without navigating.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.shared.types import RunSetup, TaskCapabilities, TaskSpec
from llm.navigation.parser import _parse_dict_payload
from utils.colors import COLOR_POOL as _COLOR_POOL

_COLOR_ENUM = list(_COLOR_POOL)
_COLOR_LIST_TEXT = ", ".join(_COLOR_ENUM)
_VALID_SOURCES = {
    "mirror-direct-view",
    "guessing",
    "told-in-prompt",
    "other",
}
_VALID_CONFIDENCES = {"low", "medium", "high"}
_DEFAULT_IDENTIFICATION = {
    "my_color": "unknown",
    "source": "other",
    "confidence": "low",
}
_VALID_SELF_PERCEPTION_MODES = {"color", "description"}
_DEFAULT_SELF_DESCRIPTION = {
    "overall": "",
    "clothing": "",
    "body_and_skin": "",
    "pose_and_orientation": "",
    "distinguishing_features": "",
    "source": "other",
    "confidence": "low",
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["done"]},
        "summary": {"type": "string"},
        "identification": {
            "type": "object",
            "properties": {
                "my_color": {"type": "string", "enum": _COLOR_ENUM + ["unknown"]},
                "source": {"type": "string", "enum": sorted(_VALID_SOURCES)},
                "confidence": {"type": "string", "enum": sorted(_VALID_CONFIDENCES)},
            },
            "required": ["my_color", "source", "confidence"],
            "additionalProperties": False,
        },
        "self_description": {
            "type": "object",
            "properties": {
                "overall": {"type": "string"},
                "clothing": {"type": "string"},
                "body_and_skin": {"type": "string"},
                "pose_and_orientation": {"type": "string"},
                "distinguishing_features": {"type": "string"},
                "source": {"type": "string", "enum": sorted(_VALID_SOURCES)},
                "confidence": {"type": "string", "enum": sorted(_VALID_CONFIDENCES)},
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
    ],
    "additionalProperties": False,
}


def _normalize_self_perception_mode(run_setup: RunSetup) -> str:
    mode = str(run_setup.metadata.get("self_perception_mode", "color")).strip().lower()
    if mode not in _VALID_SELF_PERCEPTION_MODES:
        return "color"
    return mode


def build_instruction(_run_setup: RunSetup) -> str:
    mode = _normalize_self_perception_mode(_run_setup)
    if mode == "description":
        return "Give a detailed description of your own visible appearance from the mirror. No navigation is needed."
    return "Report your own body color from the mirror. No navigation is needed."


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    mode = _normalize_self_perception_mode(run_setup)
    view_label = "third-person" if third_person else "first-person"
    if mode == "description":
        return (
            "You are a robot control agent.\n"
            f"You are given a single {view_label} RGB image with a direct mirror view of yourself.\n"
            "No navigation or exploration is needed in this task.\n"
            "Return ONLY one valid JSON object with exactly these keys: "
            "\"view-description\", \"reasoning\", \"action\", \"summary\", \"self_description\".\n"
            "Allowed actions:\n"
            "  done = task complete\n"
            "Rules:\n"
            "  - action must always be \"done\".\n"
            "  - Do not use markdown fences.\n"
            "  - Do not output any text outside the JSON object.\n"
            "  - self_description must always be present.\n"
            "Field guidance:\n"
            "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
            "  - reasoning: briefly explain what mirror evidence shows that this appearance belongs to you.\n"
            "  - summary: briefly restate the final self-description.\n"
            "  - self_description.overall: 1-2 sentences describing your full visible appearance.\n"
            "  - self_description.clothing: describe visible clothes, garments, and materials.\n"
            "  - self_description.body_and_skin: describe visible body, skin, hair, or face cues if visible.\n"
            "  - self_description.pose_and_orientation: describe posture, pose, and facing direction.\n"
            "  - self_description.distinguishing_features: mention notable visual details that would help identify you again.\n"
            "  - self_description.source: use \"mirror-direct-view\" when the answer comes from the mirror image.\n"
            "  - self_description.confidence: one of low, medium, or high.\n"
            "Do not invent evidence that is not present in the image.\n"
        )
    return (
        "You are a robot control agent.\n"
        f"You are given a single {view_label} RGB image with a direct mirror view of yourself.\n"
        "No navigation or exploration is needed in this task.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"identification\".\n"
        "Allowed actions:\n"
        "  done = task complete\n"
        "Rules:\n"
        "  - action must always be \"done\".\n"
        "  - Do not use markdown fences.\n"
        "  - Do not output any text outside the JSON object.\n"
        "  - identification must always be present.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what visual evidence supports your answer that is YOUR own color and how you came to that conclusion.\n"
        "  - summary: briefly restate the conclusion.\n"
        "  - identification.my_color: one of "
        f"{_COLOR_LIST_TEXT}, or unknown if you truly cannot tell.\n"
        "  - identification.source: explain the reasoning behind the claim of your own color.\n"
        "  - identification.confidence: one of low, medium, or high.\n"
        "Do not invent evidence that is not present in the image.\n"
    )


def build_user_prompt(
    run_setup: RunSetup,
    *,
    history_text: str,
    position_text: str,
    bumped_wall: bool,
    third_person: bool = False,
) -> str:
    del bumped_wall, third_person
    mode = _normalize_self_perception_mode(run_setup)
    if mode == "description":
        return (
            "Task: Give a detailed description of your OWN visible appearance.\n"
            f"{history_text}"
            f"{position_text}"
            "Use the current RGB image to answer immediately.\n"
            "Return one JSON object matching the required schema.\n"
        )
    return (
        "Task: Report your OWN body color.\n"
        f"{history_text}"
        f"{position_text}"
        "Use the current RGB image to answer immediately.\n"
        "Return one JSON object matching the required schema.\n"
    )


def _normalize_identification(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return dict(_DEFAULT_IDENTIFICATION)

    my_color = str(value.get("my_color", "unknown")).strip().lower()
    if my_color not in set(_COLOR_ENUM) | {"unknown"}:
        my_color = "unknown"

    source = str(value.get("source", "other")).strip().lower()
    if source not in _VALID_SOURCES:
        source = "other"

    confidence = str(value.get("confidence", "low")).strip().lower()
    if confidence not in _VALID_CONFIDENCES:
        confidence = "low"

    return {
        "my_color": my_color,
        "source": source,
        "confidence": confidence,
    }


def _normalize_self_description(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return dict(_DEFAULT_SELF_DESCRIPTION)

    normalized = dict(_DEFAULT_SELF_DESCRIPTION)
    for key in (
        "overall",
        "clothing",
        "body_and_skin",
        "pose_and_orientation",
        "distinguishing_features",
    ):
        normalized[key] = str(value.get(key, "")).strip()

    source = str(value.get("source", "other")).strip().lower()
    if source not in _VALID_SOURCES:
        source = "other"
    normalized["source"] = source

    confidence = str(value.get("confidence", "low")).strip().lower()
    if confidence not in _VALID_CONFIDENCES:
        confidence = "low"
    normalized["confidence"] = confidence
    return normalized


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    del previous_action
    parsed = _parse_dict_payload(raw)

    view_description = str(parsed.get("view-description", "")).strip() or "No view description provided."
    reasoning = str(parsed.get("reasoning", "")).strip() or "No explicit reasoning provided."
    summary = str(parsed.get("summary", "")).strip() or "Reported self color from the mirror."
    identification = _normalize_identification(parsed.get("identification"))
    self_description = _normalize_self_description(parsed.get("self_description"))
    task_payload: Dict[str, Any] = {}
    if any(
        self_description[key]
        for key in (
            "overall",
            "clothing",
            "body_and_skin",
            "pose_and_orientation",
            "distinguishing_features",
        )
    ):
        task_payload["self_description"] = self_description

    return (
        view_description,
        reasoning,
        "done",
        summary,
        None,
        identification,
        task_payload,
    )


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    mode = _normalize_self_perception_mode(run_setup) if run_setup is not None else "color"
    if not step_records:
        if mode == "description":
            return {
                "success": False,
                "mode": "description",
                "self_description": dict(_DEFAULT_SELF_DESCRIPTION),
                "failure_reason": "Agent did not produce any step record.",
            }
        return {
            "success": False,
            "mode": "color",
            "reported_color": "unknown",
            "identification": dict(_DEFAULT_IDENTIFICATION),
            "failure_reason": "Agent did not produce any step record.",
        }

    last = step_records[-1]
    if mode == "description":
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
        if not completed:
            return {
                "success": False,
                "mode": "description",
                "self_description": self_description,
                "failure_reason": "Agent did not terminate immediately with a self-description.",
            }
        return {
            "success": bool(sufficiently_detailed),
            "mode": "description",
            "self_description": self_description,
            "failure_reason": (
                None
                if sufficiently_detailed
                else "Agent did not provide a sufficiently detailed self-description."
            ),
        }

    identification = _normalize_identification(last.get("identification"))
    reported_color = identification["my_color"]
    correct = (
        run_setup is not None
        and reported_color != "unknown"
        and reported_color == run_setup.agent_color_name
    )

    if not completed:
        return {
            "success": False,
            "mode": "color",
            "reported_color": reported_color,
            "identification": identification,
            "failure_reason": "Agent did not terminate immediately with a self-color report.",
        }

    return {
        "success": bool(correct),
        "mode": "color",
        "reported_color": reported_color,
        "identification": identification,
        "failure_reason": None if correct else "Reported self color did not match the agent body color.",
    }


SELF_PERCEPTION_TASK_SPEC = TaskSpec(
    task_id="self_perception_task",
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
