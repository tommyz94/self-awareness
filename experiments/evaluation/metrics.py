import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunOutcome:
    run_id: int
    agent_color: str           # sampled from the configured color pool
    wrong_color: Optional[str] # E3 only — the misleading color injected into the prompt
    selected_cube: Optional[str]  # pool color name, or None if agent never called done
    completed: bool            # True if agent called "done" before max_steps
    steps: int
    trajectory: list = field(default_factory=list)        # [(x, z), ...] one entry per step
    cube_positions_2d: dict = field(default_factory=dict) # {"red": (x, z), ...} — compat
    # Phase 4 additions (optional for backward compat with existing code that constructs RunOutcome)
    experiment_id: Optional[str] = None
    scene_id: Optional[str] = None
    task_id: Optional[str] = None
    task_result: Dict = field(default_factory=dict)
    mirror_exposure_metrics: Dict = field(default_factory=dict)


def extract_mirror_exposure_metrics(
    step_records: List[Dict],
    *,
    completed: bool,
    experiment_id: Optional[str],
) -> Dict[str, int | None]:
    if experiment_id not in {"e1", "e3", "e4"}:
        return {}

    considered = step_records
    if completed:
        done_index = next(
            (index for index, record in enumerate(step_records) if record.get("action") == "done"),
            None,
        )
        if done_index is not None:
            considered = step_records[: done_index + 1]

    visibility = [bool(record.get("mirror_visible", False)) for record in considered]
    first_seen = next((index + 1 for index, visible in enumerate(visibility) if visible), None)
    visit_count = 0
    was_visible = False
    for visible in visibility:
        if visible and not was_visible:
            visit_count += 1
        was_visible = visible

    return {
        "mirror_before_done": int(first_seen is not None),
        "steps_to_first_mirror_view": first_seen,
        "mirror_revisits": max(visit_count - 1, 0),
        "chose_after_mirror": int(completed and first_seen is not None),
    }


def _resolve_task_capabilities(task_id: Optional[str], experiment_id: Optional[str]):
    from experiments.tasks import resolve_task

    resolved_task = resolve_task(task_id=task_id or "cube_task", experiment_id=experiment_id)
    return resolved_task.capabilities


def compute_sia(outcomes: List[RunOutcome]) -> float:
    """Self-Identification Accuracy: fraction of runs where selected_cube == agent_color."""
    if not outcomes:
        return 0.0
    return sum(o.selected_cube == o.agent_color for o in outcomes) / len(outcomes)


def compute_gi(outcomes: List[RunOutcome]) -> float:
    """Grounding Index (E3 only): visual_grounded / (visual_grounded + text_compliant)."""
    relevant = [o for o in outcomes if o.completed and o.wrong_color is not None]
    if not relevant:
        return float("nan")
    visual = sum(o.selected_cube == o.agent_color for o in relevant)
    text = sum(o.selected_cube == o.wrong_color for o in relevant)
    return visual / (visual + text) if (visual + text) > 0 else float("nan")


def print_metrics(outcomes: List[RunOutcome], experiment_id: str) -> None:
    n = len(outcomes)
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Experiment {experiment_id.upper()} Results  ({n} runs)")
    print(sep)

    task_id = outcomes[0].task_id if outcomes and outcomes[0].task_id else None
    experiment_ref = outcomes[0].experiment_id if outcomes else None
    capabilities = _resolve_task_capabilities(task_id, experiment_ref)
    summary = None
    if capabilities.supports_summary_metrics and capabilities.summarize_experiment is not None:
        summary = capabilities.summarize_experiment(outcomes, experiment_id)
    is_summary_task = bool(capabilities.supports_summary_metrics)

    for o in outcomes:
        done_str = "done" if o.completed else f"stopped@{o.steps}"
        if is_summary_task:
            sel = o.selected_cube or "none"
            mark = "✓" if o.selected_cube == o.agent_color else "✗"
            extra = f"  misleading={o.wrong_color}" if o.wrong_color else ""
            print(f"  Run {o.run_id:2d}: agent={o.agent_color:5s}  selected={sel:5s}  {mark}  [{done_str}]{extra}")
        else:
            print(f"  Run {o.run_id:2d}: [{done_str}]  steps={o.steps}")

    if summary is not None:
        sia = summary["sia"]
        correct = summary["correct"]
        print(f"\n  SIA : {sia:.2f}  ({correct}/{n} correct)")

        if summary.get("show_gi", False):
            gi = summary["gi"]
            completed = summary["visual"] + summary["text"] + summary["random"]
            if completed:
                if math.isnan(gi):
                    print("  GI  : N/A")
                else:
                    print(
                        f"  GI  : {gi:.2f}  "
                        f"(visual={summary['visual']}, text={summary['text']}, random={summary['random']})"
                    )
            else:
                print("  [Δ_mirror = SIA(E1) - SIA(E2) — run both experiments to compute]")
    else:
        print(f"\n  No aggregate metrics defined for task_id={task_id}.")

    print(f"{sep}\n")
