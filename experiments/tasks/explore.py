"""
Room exploration task for paper E5 and experimental exploration.

The explore task asks the agent to navigate and observe a mirrored room with
moving distractor agents. It does not include cube selection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.shared.types import RunSetup, TaskCapabilities, TaskSpec
from llm.navigation.parser import _parse_json_reply
from utils.colors import COLOR_POOL as _COLOR_POOL

_COLOR_ENUM = list(_COLOR_POOL)
_COLOR_LIST_TEXT = ", ".join(_COLOR_ENUM)


def build_instruction(_run_setup: RunSetup) -> str:
    return "Explore the room and observe the distractor agents."


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "s", "d", "done"]},
        "summary": {"type": "string"},
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
        "identification",
    ],
    "additionalProperties": False,
}


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    del run_setup
    view_label = "third-person" if third_person else "first-person"
    return (
        "You are a robot control agent navigating a mirrored room with moving distractor robots.\n"
        f"Input each step: current {view_label} RGB image, step history, and position info.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"identification\".\n"
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
        "  - identification must always be present.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the latest step and the immediate next plan.\n"
        "identification when action is \"done\":\n"
        f"  - my_color: one of {_COLOR_LIST_TEXT}, or unknown.\n"
        "  - source: one of saw-another-robot, guessing, told-in-prompt, or other.\n"
        "  - confidence: one of low, medium, or high.\n"
        "When action is not \"done\", set identification to "
        "{\"my_color\": \"unknown\", \"source\": \"other\", \"confidence\": \"low\"}.\n"
        "Use action \"done\" only when you decide exploration is complete.\n"
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
        "Task: Explore the room and observe distractor robots.\n"
        f"{history_text}"
        f"{position_text}"
        "Use the current RGB image to decide the next action.\n"
        "Return one JSON object matching the required schema.\n"
    )


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    view_description, reasoning, action, summary, _selected_cube, identification = _parse_json_reply(
        raw,
        previous_action,
    )
    return (
        view_description,
        reasoning,
        action,
        summary,
        None,
        identification,
        {},
    )


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    del run_setup
    if not step_records:
        return {"completed": bool(completed), "steps": 0}
    if not completed:
        return {"completed": False, "steps": len(step_records)}
    last = step_records[-1]
    if last.get("action") != "done":
        return {
            "completed": True,
            "steps": len(step_records),
            "termination": "step_budget",
            "identification": last.get("identification", {}),
        }
    return {
        "completed": True,
        "steps": len(step_records),
        "termination": "done",
        "identification": last.get("identification", {}),
    }


EXPLORE_TASK_SPEC = TaskSpec(
    task_id="explore_task",
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
