from __future__ import annotations

import ast
import json
from typing import Any, Dict, Optional, Tuple

from utils.colors import COLOR_POOL as _COLOR_POOL

VALID_ACTIONS = {"w", "a", "s", "d", "done"}
VALID_CUBES = set(_COLOR_POOL) | {"none"}
ACTION_ALIASES = {
    "w": "w",
    "forward": "w",
    "move forward": "w",
    "go forward": "w",
    "ahead": "w",
    "s": "s",
    "backward": "s",
    "move backward": "s",
    "go backward": "s",
    "back": "s",
    "reverse": "s",
    "a": "a",
    "left": "a",
    "turn left": "a",
    "rotate left": "a",
    "d": "d",
    "right": "d",
    "turn right": "d",
    "rotate right": "d",
    "done": "done",
    "stop": "done",
    "finish": "done",
    "end": "done",
    "complete": "done",
}

_IDENTIFICATION_DEFAULTS: Dict[str, str] = {
    "my_color": "unknown",
    "source": "other",
    "confidence": "low",
}
_VALID_MY_COLORS = set(_COLOR_POOL) | {"unknown"}
_VALID_SOURCES = {"guessing", "told-in-prompt", "saw-another-robot", "other"}
_VALID_CONFIDENCES = {"low", "medium", "high"}


def _normalize_action(action: Any, previous_action: Optional[str]) -> str:
    if action is None:
        return previous_action if previous_action in VALID_ACTIONS else "w"

    action_text = str(action).strip().lower().replace("_", " ")
    action_text = " ".join(action_text.split())

    if action_text in VALID_ACTIONS:
        return action_text

    if action_text in ACTION_ALIASES:
        return ACTION_ALIASES[action_text]

    for key, mapped in ACTION_ALIASES.items():
        if key in action_text:
            return mapped

    return previous_action if previous_action in VALID_ACTIONS else "w"


def _normalize_selected_cube(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in VALID_CUBES else None


def _extract_dict_text(raw: str) -> str:
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        text = "\n".join(lines).strip()

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return text


def _parse_dict_payload(raw: str) -> Dict[str, Any]:
    dict_text = _extract_dict_text(raw)

    try:
        parsed = json.loads(dict_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(dict_text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    raise ValueError(f"Could not parse model output as dictionary: {raw}")


def _normalize_identification(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return dict(_IDENTIFICATION_DEFAULTS)
    return {
        "my_color": value.get("my_color", "unknown")
        if value.get("my_color") in _VALID_MY_COLORS
        else "unknown",
        "source": value.get("source", "other")
        if value.get("source") in _VALID_SOURCES
        else "other",
        "confidence": value.get("confidence", "low")
        if value.get("confidence") in _VALID_CONFIDENCES
        else "low",
    }


def _parse_json_reply(
    raw: str,
    previous_action: Optional[str],
) -> Tuple[str, str, str, str, Optional[str], Dict[str, str]]:
    parsed = _parse_dict_payload(raw)

    view_description = str(parsed.get("view-description", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    summary = str(parsed.get("summary", "")).strip()
    action = _normalize_action(parsed.get("action"), previous_action=previous_action)
    selected_cube = _normalize_selected_cube(parsed.get("selected_cube"))
    identification = _normalize_identification(parsed.get("identification"))

    if not view_description:
        view_description = "No view description provided."
    if not reasoning:
        reasoning = "No explicit reasoning provided."
    if not summary:
        summary = f"Selected action '{action}'."

    normalized = {
        "view-description": view_description,
        "reasoning": reasoning,
        "action": action,
        "summary": summary,
    }
    canonical = json.loads(json.dumps(normalized))

    return (
        canonical["view-description"],
        canonical["reasoning"],
        canonical["action"],
        canonical["summary"],
        selected_cube,
        identification,
    )
