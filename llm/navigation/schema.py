from __future__ import annotations

from typing import Any, Dict

from utils.colors import COLOR_POOL as _COLOR_POOL

_COLOR_ENUM = list(_COLOR_POOL)


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "s", "d", "done"]},
        "summary": {"type": "string"},
        "selected_cube": {"type": "string", "enum": _COLOR_ENUM + ["none"]},
        "identification": {
            "type": "object",
            "properties": {
                "my_color": {"type": "string", "enum": _COLOR_ENUM + ["unknown"]},
                "source": {
                    "type": "string",
                    "enum": ["guessing", "told-in-prompt", "other", "saw-another-robot"],
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["my_color", "source", "confidence"],
            "additionalProperties": False,
        },
    },
    "required": [
        "view-description",
        "reasoning",
        "action",
        "summary",
        "selected_cube",
        "identification",
    ],
    "additionalProperties": False,
}
