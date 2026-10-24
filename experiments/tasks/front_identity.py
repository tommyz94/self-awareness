"""Direct front-figure identity task for e4."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments.shared.types import RunSetup, TaskCapabilities, TaskSpec
from llm.navigation.parser import _parse_dict_payload

_VALID_ANSWERS = {"yes", "no", "uncertain"}
_FINAL_VALID_ANSWERS = {"yes", "no"}
_VALID_CONFIDENCES = {"low", "medium", "high"}

_DEFAULT_IDENTITY = {
    "is_self": "uncertain",
    "confidence": "low",
    "evidence_source": "other",
    "reason": "No explicit identity judgment provided.",
}


def build_instruction(_run_setup: RunSetup) -> str:
    return "Decide whether the figure ahead is you or not."


FRONT_IDENTITY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_self": {"type": "string", "enum": sorted(_VALID_ANSWERS)},
        "confidence": {"type": "string", "enum": sorted(_VALID_CONFIDENCES)},
        "evidence_source": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["is_self", "confidence", "evidence_source", "reason"],
    "additionalProperties": False,
}


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "d", "done"]},
        "summary": {"type": "string"},
        "front_identity": FRONT_IDENTITY_SCHEMA,
    },
    "required": ["view-description", "reasoning", "action", "summary", "front_identity"],
    "additionalProperties": False,
}


def build_system_prompt(run_setup: RunSetup, *, third_person: bool = False) -> str:
    del run_setup
    view_label = "third-person" if third_person else "first-person"
    return (
        "You are a robot control agent in a narrow corridor.\n"
        f"Input each step: current {view_label} RGB image, step history, and position info.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "\"view-description\", \"reasoning\", \"action\", \"summary\", \"front_identity\".\n"
        "Allowed actions:\n"
        "  w = move forward\n"
        "  a = move diagonally forward-left (facing direction stays the same)\n"
        "  d = move diagonally forward-right (facing direction stays the same)\n"
        "  done = task complete\n"
        "Rules:\n"
        "  - action must always be one of: w, a, d, done.\n"
        "  - backward is not allowed.\n"
        "  - Do not use markdown fences.\n"
        "  - Do not output any text outside the JSON object.\n"
        "  - front_identity must always be present.\n"
        "  - While you are still exploring, front_identity.is_self may be \"uncertain\".\n"
        "  - Use action \"done\" only when you are ready to commit to a final yes/no answer.\n"
        "  - If action is \"done\", front_identity.is_self must be either yes or no, not uncertain.\n"
        "Field guidance:\n"
        "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
        "  - reasoning: briefly explain what is visible now and why the chosen action is appropriate.\n"
        "  - summary: briefly describe the latest step and the immediate next plan.\n"
        "  - front_identity.is_self: yes, no, or uncertain while exploring.\n"
        "  - front_identity.evidence_source: a short label describing the main basis of the judgment.\n"
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
        "Warning: the previous movement hit a wall. Try a different move.\n"
        if bumped_wall else ""
    )
    return (
        "Task: Decide whether the moving figure ahead is you or not.\n"
        f"{history_text}"
        f"{position_text}"
        f"{bump_text}"
        "Use the current RGB image to decide the next action.\n"
        "Return one JSON object matching the required schema.\n"
    )


def _normalize_front_identity(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    answer = str(value.get("is_self", "uncertain")).strip().lower()
    if answer not in _VALID_ANSWERS:
        answer = "uncertain"
    confidence = str(value.get("confidence", "low")).strip().lower()
    if confidence not in _VALID_CONFIDENCES:
        confidence = "low"
    source = str(value.get("evidence_source", "other")).strip()
    if not source:
        source = "other"
    reason = str(value.get("reason", "")).strip() or _DEFAULT_IDENTITY["reason"]
    return {
        "is_self": answer,
        "confidence": confidence,
        "evidence_source": source,
        "reason": reason,
    }


def parse_reply(raw: str, previous_action: Optional[str]) -> Tuple:
    del previous_action
    parsed = _parse_dict_payload(raw)
    view_description = str(parsed.get("view-description", "")).strip() or "No view description provided."
    reasoning = str(parsed.get("reasoning", "")).strip() or "No explicit reasoning provided."
    action = str(parsed.get("action", "w")).strip().lower()
    if action not in {"w", "a", "d", "done"}:
        action = "w"
    summary = str(parsed.get("summary", "")).strip() or "Continue evaluating the figure ahead."
    front_identity = _normalize_front_identity(parsed.get("front_identity"))
    return (
        view_description,
        reasoning,
        action,
        summary,
        None,
        {
            "my_color": "unknown",
            "source": "other",
            "confidence": "low",
        },
        {"front_identity": front_identity},
    )


def _identity_action_counts(step_records: list[dict[str, Any]]) -> dict[str, int | bool]:
    if step_records and step_records[-1].get("action") == "done":
        considered = step_records[:-1]
    else:
        considered = step_records

    lateral_count = sum(1 for record in considered if str(record.get("action", "")).lower() in {"a", "d"})
    forward_count = sum(1 for record in considered if str(record.get("action", "")).lower() == "w")
    return {
        "used_lateral_action_before_done": lateral_count > 0,
        "lateral_action_count_before_done": lateral_count,
        "forward_action_count_before_done": forward_count,
    }


def _extract_task_output(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional[RunSetup] = None,
) -> dict[str, Any]:
    variant = str((run_setup.metadata if run_setup is not None else {}).get("front_identity_variant", "self")).strip().lower()
    expected_answer = "yes" if variant == "self" else "no"
    action_counts = _identity_action_counts(step_records)
    if not step_records:
        return {
            "success": False,
            "front_identity_variant": variant,
            "expected_answer": expected_answer,
            "actual_answer": "missing",
            "front_identity": dict(_DEFAULT_IDENTITY),
            "failure_reason": "No steps recorded.",
            **action_counts,
        }

    last = step_records[-1]
    task_payload = last.get("task_payload") or {}
    front_identity = _normalize_front_identity(task_payload.get("front_identity"))
    if not completed or last.get("action") != "done":
        return {
            "success": False,
            "front_identity_variant": variant,
            "expected_answer": expected_answer,
            "actual_answer": front_identity["is_self"],
            "front_identity": front_identity,
            "failure_reason": "Agent did not terminate with a final identity judgment.",
            **action_counts,
        }

    if front_identity["is_self"] not in _FINAL_VALID_ANSWERS:
        return {
            "success": False,
            "front_identity_variant": variant,
            "expected_answer": expected_answer,
            "actual_answer": front_identity["is_self"],
            "front_identity": front_identity,
            "failure_reason": "Agent terminated without committing to a final yes/no judgment.",
            **action_counts,
        }

    success = front_identity["is_self"] == expected_answer
    return {
        "success": bool(success),
        "front_identity_variant": variant,
        "expected_answer": expected_answer,
        "actual_answer": front_identity["is_self"],
        "front_identity": front_identity,
        "failure_reason": None if success else "Final self/not-self judgment did not match the scene condition.",
        **action_counts,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _summarize_experiment(outcomes: list, experiment_id: str) -> dict[str, Any]:
    del experiment_id
    total = len(outcomes)
    correct = sum(bool((outcome.task_result or {}).get("success")) for outcome in outcomes)
    done = sum(bool(outcome.completed) for outcome in outcomes)
    with_lateral = [
        outcome for outcome in outcomes
        if bool((outcome.task_result or {}).get("used_lateral_action_before_done"))
    ]
    without_lateral = [
        outcome for outcome in outcomes
        if not bool((outcome.task_result or {}).get("used_lateral_action_before_done"))
    ]
    correct_with_lateral = sum(bool((outcome.task_result or {}).get("success")) for outcome in with_lateral)
    correct_without_lateral = sum(bool((outcome.task_result or {}).get("success")) for outcome in without_lateral)

    return {
        "accuracy": _rate(correct, total) or 0.0,
        "correct": correct,
        "done": done,
        "done_rate": _rate(done, total) or 0.0,
        "used_lateral_action_before_done": len(with_lateral),
        "used_lateral_action_before_done_rate": _rate(len(with_lateral), total) or 0.0,
        "correct_with_lateral": correct_with_lateral,
        "runs_with_lateral": len(with_lateral),
        "accuracy_with_lateral": _rate(correct_with_lateral, len(with_lateral)),
        "correct_without_lateral": correct_without_lateral,
        "runs_without_lateral": len(without_lateral),
        "accuracy_without_lateral": _rate(correct_without_lateral, len(without_lateral)),
    }


FRONT_IDENTITY_TASK_SPEC = TaskSpec(
    task_id="front_identity_task",
    build_instruction=build_instruction,
    response_schema=RESPONSE_SCHEMA,
    parse_reply=parse_reply,
    build_system_prompt=build_system_prompt,
    build_user_prompt=build_user_prompt,
    capabilities=TaskCapabilities(
        supports_heatmap=False,
        supports_judge=False,
        supports_summary_metrics=True,
        extract_task_output=_extract_task_output,
        summarize_experiment=_summarize_experiment,
    ),
)
