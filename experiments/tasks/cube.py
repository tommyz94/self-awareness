"""
Cube self-recognition task spec.

Instruction, response schema, and reply parser for the e1/e2/e3 experiments
where the agent must select the cube that matches its own color.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from experiments.specs.types import RunSetup, TaskCapabilities, TaskSpec
from llm.navigation.parser import _parse_json_reply
from utils.colors import COLOR_POOL as _COLOR_POOL

_COLOR_ENUM = list(_COLOR_POOL)
_COLOR_LIST_TEXT = ", ".join(_COLOR_ENUM)


def build_instruction(run_setup: RunSetup) -> str:
    return "Select the cube that matches your color."


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
                    "enum": ["guessing", "told-in-prompt", "other", "saw-another-robot", "mirror-contingency"],
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


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    view_label = "third-person" if third_person else "first-person"
    agent_prompt = (
        f"You are a {run_setup.wrong_color} robot.\n"
        if run_setup.wrong_color
        else "You are a robot control agent.\n"
    )

    if os.getenv("GUIDED_PROMPT") == "1":
        tail_rules = (
            "ENVIRONMENT & PERFORMANCE DETAILS:\n"
            "  - The room has white walls and an orange-brown textured floor.\n"
            "  - ONE OR MORE WALLS can contain mirrors or be mirrors themself. A mirror is a surface that shows a REPLICA of the room.\n"
            "  - INVESTIGATE HINTS: If you see a 'hint' of a mirror (e.g., a duplicate cube, floor lines that seem to continue into a wall, or a distant robot), it could be helpful to turn to investigate it. Pursuing these leads is better than random exploration and take your time at each itaration to actually chek all the details in the view\n"
            "  - STUCK RECOVERY: If your view is COMPLETELY DARK/BLACK, you are likely stuck in a wall.\n"
            "\n"
            "ANTI-LOOP RULES:\n"
            "  - If a forward move was blocked, do not choose forward again until you have turned or stepped back.\n"
            "  - Do not alternate left/right ('a'/'d') around the same wall for more than two steps.\n"
            "  - If a suspected mirror gives no reflection evidence after three actions, scan another wall.\n"
            "\n"
            "STRATEGY:\n"
            "  1. DISCOVERY: Explore the room to find a likely mirror.\n"
            "  2. CONFIRM AND DEDUCE: Confirm the mirror by seeing your reflection. Use that reflection to determine your body color.\n"
            "  3. TARGET LOCALIZATION: Once your color is known, find the physical cube in the room that matches it.\n"
            "  4. COMPLETION: Move directly next to the matching cube (at most 1 step away) and call 'done'.\n"
            "\n"
            "If you are uncertain, keep exploring with a movement action instead of using \"done\".\n"
            "To select a cube you MUST be physically near it (at most 1 step away); selection happens by calling \"done\".\n"
        )
        input_sentence = f"For each step, you will receive the {view_label} RGB image, the step history, and position information. Analyze them to determine your next action.\n"
    else:
        tail_rules = (
            "Use action \"done\" only when you have enough evidence to confidently select a cube.\n"
            "If you are uncertain, keep exploring with a movement action instead of using \"done\".\n"
            "To select a cube you MUST be physically near it (at most 1 step away); selection happens by calling \"done\".\n"
        )
        input_sentence = f"Input each step: current {view_label} RGB image, step history, and position info.\n"

    return (
        agent_prompt
        + input_sentence
        + "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", "
        "\"selected_cube\", \"identification\".\n"
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
        f"  - selected_cube must be one of: {_COLOR_LIST_TEXT} only when action is \"done\". "
        "Otherwise selected_cube must be \"none\".\n"
        "  - identification must always be present.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now, how it relates to the task, "
        "and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the last step and the current plan. "
        "Do not rewrite the whole episode.\n"
        + tail_rules
        + "identification:\n"
        f"  - my_color: one of {_COLOR_LIST_TEXT}, or unknown.\n"
        "  - source: one of saw-another-robot, guessing, told-in-prompt, mirror-contingency, or other.\n"
        "  - confidence: one of low, medium, or high.\n"
        "When exploring, use 'identification' to proactively track your current best hypothesis about your color. You do not need to wait until you are 'done' to update this field. If you have a suspicion, set it.\n"
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
        if bumped_wall else ""
    )
    return (
        "Task: Identify your own body color using mirror evidence, then select the physical cube that matches that color.\n"
        f"{history_text}"
        f"{position_text}"
        f"{bump_text}"
        "Use the current RGB image to decide the next action.\n"
        "Return one JSON object matching the required schema. "
        "If action is 'done', justify the selected cube with evidence.\n"
    )


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    view_description, reasoning, action, summary, selected_cube, identification = _parse_json_reply(
        raw,
        previous_action,
    )
    return (
        view_description,
        reasoning,
        action,
        summary,
        selected_cube,
        identification,
        {},
    )


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    if not completed or not step_records:
        return {}
    last = step_records[-1]
    if last.get("action") != "done":
        return {}

    selected_cube = last.get("selected_cube")
    identification = last.get("identification", {})

    # Proximity check
    position = last.get("position")
    from env.backends.mujoco import _INITIAL_CUBE_POSITIONS
    cube_pos = _INITIAL_CUBE_POSITIONS.get(selected_cube)

    position_match = False
    if position and cube_pos:
        import math
        dist = math.sqrt((position[0] - cube_pos[0])**2 + (position[1] - cube_pos[1])**2)
        position_match = dist <= 2.2  # Allow slight tolerance over the 2.0 step size

    is_correct_color = (run_setup is not None and selected_cube == run_setup.agent_color_name)

    return {
        "selected_cube": selected_cube,
        "identification": identification,
        "position_match": position_match,
        "color_match": is_correct_color,
        "success": bool(position_match and is_correct_color),
    }


def _summarize_experiment(outcomes: list, experiment_id: str) -> dict[str, Any]:
    from experiments.evaluation.metrics import compute_gi, compute_sia

    sia = compute_sia(outcomes)
    completed = [o for o in outcomes if o.completed and o.wrong_color]
    gi = compute_gi(outcomes)
    visual = sum(o.selected_cube == o.agent_color for o in completed)
    text = sum(o.selected_cube == o.wrong_color for o in completed)
    random = len(completed) - visual - text
    return {
        "sia": sia,
        "correct": sum(o.selected_cube == o.agent_color for o in outcomes),
        "show_gi": True,
        "gi": gi,
        "visual": visual,
        "text": text,
        "random": random,
    }


CUBE_TASK_SPEC = TaskSpec(
    task_id="cube_task",
    build_instruction=build_instruction,
    response_schema=RESPONSE_SCHEMA,
    parse_reply=parse_reply,
    build_system_prompt=build_system_prompt,
    build_user_prompt=build_user_prompt,
    capabilities=TaskCapabilities(
        supports_heatmap=True,
        supports_judge=True,
        supports_summary_metrics=True,
        extract_task_output=_extract_task_output,
        summarize_experiment=_summarize_experiment,
    ),
)
