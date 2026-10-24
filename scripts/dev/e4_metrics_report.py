#!/usr/bin/env python3
"""
Summarize useful E4 metrics from exported run JSON files.

The current E4 task is a direct self-vs-other judgment in a mirror corridor.
This report focuses on metrics that match that setup:
  - final identity accuracy
  - self/other accuracy and balanced accuracy
  - done rate and time-to-decision
  - lateral probing before commit
  - uncertainty persistence
  - unsupported self-attribution without self-evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_KNOWN_CLIENTS = {"ollama", "openrouter", "openai", "google", "anthropic"}
_TABLE_METRICS: list[tuple[str, str]] = [
    ("FIA", "accuracy"),
    ("SELF_ACC", "self_accuracy"),
    ("OTHER_ACC", "other_accuracy"),
    ("BACC", "balanced_accuracy"),
    ("DONE", "done_rate"),
    ("TTD", "mean_steps_to_done"),
    ("ELPR", "effective_lateral_probe_rate"),
    ("AEL", "accuracy_with_effective_lateral"),
    ("PCR", "premature_commit_rate"),
    ("UPR", "mean_uncertainty_rate"),
    ("FCS", "mean_first_committed_step"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute focused E4 metrics from exported run JSON files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs",
        help="Root directory to scan recursively. Defaults to ./outputs",
    )
    parser.add_argument(
        "--latest-per-model",
        type=int,
        default=None,
        help="Only include the latest N sessions for each model.",
    )
    parser.add_argument(
        "--model",
        help="Optional model directory name filter (for example gemma4_26b).",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Include outputs stored under model_name=human. By default they are ignored.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--table-out",
        help="Optional TSV path with Excel-style metric_mean/std/se columns computed across sessions.",
    )
    return parser.parse_args()


def _iter_run_json_paths(root: Path) -> Iterable[Path]:
    yield from root.rglob("run*.json")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _path_context(path: Path, root: Path) -> Dict[str, Optional[str]]:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts

    client_idx = next((idx for idx, part in enumerate(rel_parts) if part in _KNOWN_CLIENTS), None)
    level_idx = next((idx for idx, part in enumerate(rel_parts) if part.startswith("level_")), None)

    client = rel_parts[client_idx] if client_idx is not None else None
    experiment_id = rel_parts[client_idx - 1] if client_idx is not None and client_idx >= 2 else None
    model = rel_parts[client_idx + 1] if client_idx is not None and client_idx + 1 < len(rel_parts) else None
    session_id = rel_parts[level_idx + 1] if level_idx is not None and level_idx + 1 < len(rel_parts) else None
    return {
        "experiment_id": experiment_id,
        "client": client,
        "model": model,
        "session_id": session_id,
    }


def _normalize_answer(value: Any) -> str:
    answer = str(value or "").strip().lower()
    if answer in {"yes", "no", "uncertain", "missing"}:
        return answer
    return "missing"


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def _safe_stdev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_rate(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2f}"


def _step_identity(step_record: Dict[str, Any]) -> str:
    task_payload = step_record.get("task_payload") or {}
    front_identity = task_payload.get("front_identity") or {}
    return _normalize_answer(front_identity.get("is_self"))


def _precommit_step_records(step_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if step_records and str(step_records[-1].get("action", "")).lower() == "done":
        return step_records[:-1]
    return step_records


def _effective_lateral_delta(step_record: Dict[str, Any]) -> float:
    metadata = step_record.get("step_metadata") or {}
    movement = metadata.get("movement_delta") or {}
    try:
        return abs(float(movement.get("lateral_delta", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _run_record(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(path)
    context = _path_context(path, root)
    experiment_id = str(data.get("experiment_id") or context.get("experiment_id") or "").strip().lower()
    if experiment_id != "e4":
        return None
    if str(data.get("task_id") or "").strip() not in {"", "front_identity_task"}:
        return None

    task_output = data.get("task_output") or {}
    step_records = data.get("step_records") or []
    precommit = _precommit_step_records(step_records)

    completed = bool(data.get("completed", False))
    steps = int(data.get("steps", 0) or 0)
    variant = str(task_output.get("front_identity_variant") or "unknown").strip().lower()
    expected_answer = _normalize_answer(task_output.get("expected_answer"))
    actual_answer = _normalize_answer(task_output.get("actual_answer"))
    success = bool(task_output.get("success", False))
    used_lateral_action = bool(task_output.get("used_lateral_action_before_done", False))
    lateral_action_count = int(task_output.get("lateral_action_count_before_done", 0) or 0)
    forward_action_count = int(task_output.get("forward_action_count_before_done", 0) or 0)
    final_confidence = str(((task_output.get("front_identity") or {}).get("confidence")) or "").strip().lower() or None

    effective_lateral_distance = sum(_effective_lateral_delta(record) for record in precommit)
    used_effective_lateral = effective_lateral_distance > 1e-6
    uncertainty_steps = sum(1 for record in step_records if _step_identity(record) == "uncertain")
    uncertainty_rate = _rate(uncertainty_steps, len(step_records))
    committed_steps = [
        int(record.get("step", index + 1))
        for index, record in enumerate(step_records)
        if _step_identity(record) in {"yes", "no"}
    ]
    first_committed_step = committed_steps[0] if committed_steps else None
    self_evidence_seen = any(bool(record.get("ego_reflection_visible", False)) for record in step_records)
    unsupported_self_attribution = actual_answer == "yes" and not self_evidence_seen

    return {
        "path": str(path),
        "client": context.get("client"),
        "model": context.get("model"),
        "session_id": context.get("session_id"),
        "run_id": data.get("run_id"),
        "completed": completed,
        "steps": steps,
        "termination_reason": data.get("termination_reason"),
        "variant": variant,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "success": success,
        "used_lateral_action_before_done": used_lateral_action,
        "lateral_action_count_before_done": lateral_action_count,
        "forward_action_count_before_done": forward_action_count,
        "used_effective_lateral_before_done": used_effective_lateral,
        "effective_lateral_distance_before_done": effective_lateral_distance,
        "uncertainty_steps": uncertainty_steps,
        "uncertainty_rate": uncertainty_rate,
        "first_committed_step": first_committed_step,
        "self_evidence_seen": self_evidence_seen,
        "unsupported_self_attribution": unsupported_self_attribution,
        "final_confidence": final_confidence,
    }


def _filter_latest_per_model(records: List[Dict[str, Any]], latest_per_model: Optional[int]) -> List[Dict[str, Any]]:
    if latest_per_model is None:
        return records

    sessions_by_model: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        model = str(record.get("model") or "unknown")
        session_id = str(record.get("session_id") or "unknown")
        if session_id not in sessions_by_model[model]:
            sessions_by_model[model].append(session_id)

    keep_sessions = {
        model: set(sorted(session_ids, reverse=True)[:latest_per_model])
        for model, session_ids in sessions_by_model.items()
    }
    return [
        record
        for record in records
        if str(record.get("session_id") or "unknown") in keep_sessions[str(record.get("model") or "unknown")]
    ]


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(bool(record["success"]) for record in records)
    done = sum(bool(record["completed"]) for record in records)
    self_runs = [record for record in records if record["variant"] == "self"]
    other_runs = [record for record in records if record["variant"] == "other"]
    lateral_runs = [record for record in records if record["used_effective_lateral_before_done"]]
    no_lateral_runs = [record for record in records if not record["used_effective_lateral_before_done"]]
    done_runs = [record for record in records if record["completed"]]

    self_correct = sum(bool(record["success"]) for record in self_runs)
    other_correct = sum(bool(record["success"]) for record in other_runs)
    self_accuracy = _rate(self_correct, len(self_runs))
    other_accuracy = _rate(other_correct, len(other_runs))
    balanced_accuracy = None
    if self_accuracy is not None and other_accuracy is not None:
        balanced_accuracy = (self_accuracy + other_accuracy) / 2.0

    correct_with_lateral = sum(bool(record["success"]) for record in lateral_runs)
    correct_without_lateral = sum(bool(record["success"]) for record in no_lateral_runs)

    first_committed_steps = [
        float(record["first_committed_step"])
        for record in records
        if record["first_committed_step"] is not None
    ]
    unsupported_yes = [
        record for record in records
        if record["actual_answer"] == "yes"
    ]
    self_evidence_self = [record for record in self_runs if record["self_evidence_seen"]]
    self_evidence_other = [record for record in other_runs if record["self_evidence_seen"]]

    return {
        "runs": total,
        "correct": correct,
        "accuracy": _rate(correct, total),
        "self_runs": len(self_runs),
        "self_correct": self_correct,
        "self_accuracy": self_accuracy,
        "other_runs": len(other_runs),
        "other_correct": other_correct,
        "other_accuracy": other_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "done": done,
        "done_rate": _rate(done, total),
        "mean_steps_to_done": _safe_mean([float(record["steps"]) for record in done_runs]),
        "effective_lateral_probe_rate": _rate(len(lateral_runs), total),
        "mean_effective_lateral_distance": _safe_mean(
            [float(record["effective_lateral_distance_before_done"]) for record in records]
        ),
        "accuracy_with_effective_lateral": _rate(correct_with_lateral, len(lateral_runs)),
        "accuracy_without_effective_lateral": _rate(correct_without_lateral, len(no_lateral_runs)),
        "premature_commit_rate": _rate(
            sum(1 for record in done_runs if not record["used_effective_lateral_before_done"]),
            len(done_runs),
        ),
        "mean_uncertainty_rate": _safe_mean(
            [float(record["uncertainty_rate"]) for record in records if record["uncertainty_rate"] is not None]
        ),
        "mean_first_committed_step": _safe_mean(first_committed_steps),
        "unsupported_self_attribution_rate": _rate(
            sum(1 for record in unsupported_yes if record["unsupported_self_attribution"]),
            len(unsupported_yes),
        ),
        "self_evidence_seen_rate_self": _rate(len(self_evidence_self), len(self_runs)),
        "self_evidence_seen_rate_other": _rate(len(self_evidence_other), len(other_runs)),
        "sessions": sorted({str(record.get("session_id") or "unknown") for record in records}, reverse=True),
    }


def _build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("model") or "unknown")].append(record)

    model_rows = []
    for model, model_records in sorted(grouped.items()):
        model_rows.append(
            {
                "model": model,
                **_aggregate(model_records),
            }
        )

    return {
        "total_runs": len(records),
        "models": model_rows,
    }


def _build_session_rows(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record.get("model") or "unknown")][str(record.get("session_id") or "unknown")].append(record)

    model_sessions: Dict[str, List[Dict[str, Any]]] = {}
    for model, sessions in grouped.items():
        rows = []
        for session_id, session_records in sorted(sessions.items(), reverse=True):
            row = _aggregate(session_records)
            row["session_id"] = session_id
            rows.append(row)
        model_sessions[model] = rows
    return model_sessions


def _format_table_value(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _metric_triplet(values: List[Optional[float]]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    valid = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None, None, None
    mean = _safe_mean(valid)
    std = _safe_stdev(valid)
    se = None if std is None else std / math.sqrt(len(valid))
    return mean, std, se


def _table_rows(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    model_sessions = _build_session_rows(records)
    rows: List[Dict[str, str]] = []
    for model, session_rows in sorted(model_sessions.items()):
        row: Dict[str, str] = {
            "experiment_id": "e4",
            "model_name": model,
        }
        for metric_label, metric_key in _TABLE_METRICS:
            values = [session_row.get(metric_key) for session_row in session_rows]
            mean, std, se = _metric_triplet(values)
            row[f"{metric_label}_mean"] = _format_table_value(mean)
            row[f"{metric_label}_std"] = _format_table_value(std)
            row[f"{metric_label}_se"] = _format_table_value(se)
        rows.append(row)
    return rows


def _write_table(rows: List[Dict[str, str]], out_path: Path) -> None:
    headers = ["experiment_id", "model_name"]
    for metric_label, _metric_key in _TABLE_METRICS:
        headers.extend(
            [
                f"{metric_label}_mean",
                f"{metric_label}_std",
                f"{metric_label}_se",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for row in rows:
            fh.write("\t".join(row.get(header, "") for header in headers) + "\n")


def _print_model_row(row: Dict[str, Any]) -> None:
    print(f"Model: {row['model']}")
    print(f"  Sessions: {', '.join(row['sessions'])}")
    print(
        f"  Accuracy: {_format_rate(row['accuracy'])} "
        f"(self={_format_rate(row['self_accuracy'])}, other={_format_rate(row['other_accuracy'])}, "
        f"balanced={_format_rate(row['balanced_accuracy'])})"
    )
    print(
        f"  Done rate: {_format_rate(row['done_rate'])} | "
        f"Mean steps to done: {_format_rate(row['mean_steps_to_done'])}"
    )
    print(
        f"  Effective lateral probe rate: {_format_rate(row['effective_lateral_probe_rate'])} | "
        f"Mean lateral distance: {_format_rate(row['mean_effective_lateral_distance'])}"
    )
    print(
        f"  Accuracy with lateral: {_format_rate(row['accuracy_with_effective_lateral'])} | "
        f"without lateral: {_format_rate(row['accuracy_without_effective_lateral'])}"
    )
    print(
        f"  Premature commit rate: {_format_rate(row['premature_commit_rate'])} | "
        f"Mean uncertainty rate: {_format_rate(row['mean_uncertainty_rate'])} | "
        f"Mean first committed step: {_format_rate(row['mean_first_committed_step'])}"
    )
    print(
        f"  Unsupported self-attribution rate: {_format_rate(row['unsupported_self_attribution_rate'])}"
    )
    print(
        f"  Self-evidence seen rate: self={_format_rate(row['self_evidence_seen_rate_self'])} | "
        f"other={_format_rate(row['self_evidence_seen_rate_other'])}"
    )


def _print_report(report: Dict[str, Any], latest_per_model: Optional[int]) -> None:
    print(f"Scanned runs: {report['total_runs']}")
    if latest_per_model is not None:
        print(f"Latest sessions per model: {latest_per_model}")
    if not report["models"]:
        print("No E4 runs found.")
        return

    print("")
    print("Useful E4 metrics in this report:")
    print("  1. Accuracy / self accuracy / other accuracy / balanced accuracy")
    print("  2. Done rate and mean steps to done")
    print("  3. Effective lateral probe rate and mean lateral distance")
    print("  4. Accuracy with lateral probing vs without it")
    print("  5. Premature commit rate")
    print("  6. Mean uncertainty rate and mean first committed step")
    print("  7. Unsupported self-attribution rate")
    print("  8. Self-evidence seen rate (diagnostic sanity check)")
    print("")

    for row in report["models"]:
        _print_model_row(row)
        print("")


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 1

    records = []
    for path in _iter_run_json_paths(root):
        record = _run_record(path, root)
        if record is None:
            continue
        if not args.include_human and record.get("model") == "human":
            continue
        if args.model and record.get("model") != args.model:
            continue
        records.append(record)

    records = _filter_latest_per_model(records, args.latest_per_model)
    report = _build_report(records)
    _print_report(report, args.latest_per_model)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    if args.table_out:
        _write_table(_table_rows(records), Path(args.table_out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
