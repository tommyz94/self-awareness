#!/usr/bin/env python3
"""
Scan exported experiment sessions and flag structural inconsistencies.

This report is stricter than the inventory/readiness reports. It is meant to
catch corrupted session folders, especially when parallel jobs accidentally
write into the same session directory.

Checks per session:
- mixed seeds across run*.json files
- summary seed != run seeds
- summary num_runs != number of run*.json files
- duplicate or missing run_id values
- run filename != embedded run_id
- summary run entry != corresponding run JSON payload (basic fields)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_RUN_JSON_RE = re.compile(r"run(\d+)\.json$")
_RUN_DIR_RE = re.compile(r"run_(\d+)$")
_KNOWN_CLIENTS = {"ollama", "openrouter", "openai", "google", "anthropic"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report corrupted or inconsistent exported session folders."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs",
        help="Root directory to scan recursively. Defaults to ./outputs",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--only-issues",
        action="store_true",
        help="Print only sessions with at least one detected issue.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _iter_run_json_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("run*.json"):
        if _RUN_JSON_RE.fullmatch(path.name):
            yield path


def _path_context(session_dir: Path, root: Path) -> Dict[str, Optional[str]]:
    try:
        rel_parts = session_dir.relative_to(root).parts
    except ValueError:
        rel_parts = session_dir.parts

    client_idx = None
    for idx, part in enumerate(rel_parts):
        if part in _KNOWN_CLIENTS:
            client_idx = idx
            break

    level_idx = next((i for i, part in enumerate(rel_parts) if part.startswith("level_")), None)
    client = rel_parts[client_idx] if client_idx is not None else None
    experiment_id = rel_parts[client_idx - 1] if client_idx is not None and client_idx >= 2 else None
    model = rel_parts[client_idx + 1] if client_idx is not None and client_idx + 1 < len(rel_parts) else None
    session_id = rel_parts[level_idx + 1] if level_idx is not None and level_idx + 1 < len(rel_parts) else None
    variant = None
    if client_idx is not None and level_idx is not None and client_idx + 2 < level_idx:
        variant = "/".join(rel_parts[client_idx + 2 : level_idx]) or None
    return {
        "experiment_id": experiment_id,
        "client": client,
        "model": model,
        "variant": variant,
        "session_id": session_id,
    }


def _run_brief(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    run_setup = data.get("run_setup") or {}

    filename_match = _RUN_JSON_RE.fullmatch(path.name)
    filename_run_id = int(filename_match.group(1)) if filename_match else None

    run_dir_name = path.parent.name
    run_dir_match = _RUN_DIR_RE.fullmatch(run_dir_name)
    run_dir_id = int(run_dir_match.group(1)) if run_dir_match else None

    return {
        "path": str(path),
        "run_id": data.get("run_id"),
        "filename_run_id": filename_run_id,
        "run_dir_id": run_dir_id,
        "experiment_id": data.get("experiment_id"),
        "scene_id": data.get("scene_id"),
        "task_id": data.get("task_id"),
        "completed": bool(data.get("completed", False)),
        "steps": data.get("steps"),
        "termination_reason": data.get("termination_reason"),
        "agent_color": data.get("agent_color"),
        "wrong_color": data.get("wrong_color"),
        "selected_cube": data.get("selected_cube"),
        "seed": run_setup.get("seed"),
    }


def _compare_summary_entry(summary_entry: Dict[str, Any], run_record: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    fields = [
        "completed",
        "steps",
        "termination_reason",
        "agent_color",
        "wrong_color",
        "selected_cube",
    ]
    for field in fields:
        if field in summary_entry and summary_entry.get(field) != run_record.get(field):
            mismatches.append(
                f"summary.{field}={summary_entry.get(field)!r} != run.{field}={run_record.get(field)!r}"
            )

    if "correct" in summary_entry:
        agent_color = run_record.get("agent_color")
        selected_cube = run_record.get("selected_cube")
        expected_correct = bool(agent_color is not None and selected_cube == agent_color)
        if summary_entry.get("correct") != expected_correct:
            mismatches.append(
                f"summary.correct={summary_entry.get('correct')!r} != expected_correct={expected_correct!r}"
            )
    return mismatches


def _session_report(session_dir: Path, root: Path) -> Dict[str, Any]:
    run_paths = sorted(session_dir.glob("run_*/run*.json"))
    runs = [_run_brief(path) for path in run_paths]
    context = _path_context(session_dir, root)
    summary_path = session_dir / "experiment_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else None

    issues: List[Dict[str, Any]] = []
    seeds = [run["seed"] for run in runs]
    seed_counts = Counter("unknown" if seed in (None, "") else str(seed) for seed in seeds)
    distinct_seeds = [seed for seed in seed_counts if seed != "unknown"]
    run_ids = [run["run_id"] for run in runs if isinstance(run["run_id"], int)]
    run_id_counts = Counter(run_ids)

    if len(distinct_seeds) > 1:
        issues.append(
            {
                "type": "mixed_seeds",
                "details": {
                    "seed_counts": dict(seed_counts),
                },
            }
        )

    for run in runs:
        if run["run_id"] != run["filename_run_id"]:
            issues.append(
                {
                    "type": "filename_run_id_mismatch",
                    "details": {
                        "path": run["path"],
                        "run_id": run["run_id"],
                        "filename_run_id": run["filename_run_id"],
                    },
                }
            )
        if run["run_id"] != run["run_dir_id"]:
            issues.append(
                {
                    "type": "run_dir_run_id_mismatch",
                    "details": {
                        "path": run["path"],
                        "run_id": run["run_id"],
                        "run_dir_id": run["run_dir_id"],
                    },
                }
            )

    duplicates = {run_id: count for run_id, count in run_id_counts.items() if count > 1}
    if duplicates:
        issues.append({"type": "duplicate_run_ids", "details": {"counts": duplicates}})

    if run_ids:
        max_run_id = max(run_ids)
        expected_ids = set(range(1, max_run_id + 1))
        missing_ids = sorted(expected_ids - set(run_ids))
        if missing_ids:
            issues.append({"type": "missing_run_ids", "details": {"missing": missing_ids}})

    if summary is None:
        issues.append({"type": "missing_summary", "details": {}})
        summary_runs = []
    else:
        summary_seed = summary.get("seed")
        summary_num_runs = summary.get("num_runs")
        summary_runs = summary.get("runs") or []

        if summary_num_runs != len(runs):
            issues.append(
                {
                    "type": "summary_num_runs_mismatch",
                    "details": {
                        "summary_num_runs": summary_num_runs,
                        "run_json_count": len(runs),
                    },
                }
            )

        run_seed_values = {run["seed"] for run in runs if run["seed"] not in (None, "")}
        if summary_seed not in (None, "") and run_seed_values and summary_seed not in run_seed_values:
            issues.append(
                {
                    "type": "summary_seed_not_present_in_runs",
                    "details": {
                        "summary_seed": summary_seed,
                        "run_seeds": sorted(run_seed_values),
                    },
                }
            )
        elif summary_seed not in (None, ""):
            mismatched_runs = [
                {"run_id": run["run_id"], "run_seed": run["seed"]}
                for run in runs
                if run["seed"] != summary_seed
            ]
            if mismatched_runs:
                issues.append(
                    {
                        "type": "summary_seed_mismatch",
                        "details": {
                            "summary_seed": summary_seed,
                            "mismatched_runs": mismatched_runs,
                        },
                    }
                )

        summary_by_run_id = {
            row.get("run_id"): row
            for row in summary_runs
            if isinstance(row, dict) and isinstance(row.get("run_id"), int)
        }
        if len(summary_by_run_id) != len(summary_runs):
            issues.append(
                {
                    "type": "summary_duplicate_or_invalid_run_ids",
                    "details": {"summary_run_count": len(summary_runs)},
                }
            )
        for run in runs:
            run_id = run["run_id"]
            if not isinstance(run_id, int):
                issues.append(
                    {
                        "type": "invalid_run_id",
                        "details": {"path": run["path"], "run_id": run_id},
                    }
                )
                continue
            summary_entry = summary_by_run_id.get(run_id)
            if summary_entry is None:
                issues.append(
                    {
                        "type": "summary_missing_run_entry",
                        "details": {"run_id": run_id, "path": run["path"]},
                    }
                )
                continue
            entry_mismatches = _compare_summary_entry(summary_entry, run)
            if entry_mismatches:
                issues.append(
                    {
                        "type": "summary_run_mismatch",
                        "details": {
                            "run_id": run_id,
                            "path": run["path"],
                            "mismatches": entry_mismatches,
                        },
                    }
                )

    return {
        "session_dir": str(session_dir),
        **context,
        "run_count": len(runs),
        "seed_counts": dict(seed_counts),
        "issues": issues,
    }


def _build_report(root: Path) -> Dict[str, Any]:
    sessions: dict[Path, List[Path]] = defaultdict(list)
    for run_path in _iter_run_json_paths(root):
        session_dir = run_path.parent.parent
        sessions[session_dir].append(run_path)

    session_rows = []
    issue_counts: Counter[str] = Counter()
    for session_dir in sorted(sessions):
        row = _session_report(session_dir, root)
        session_rows.append(row)
        for issue in row["issues"]:
            issue_counts[issue["type"]] += 1

    bad_sessions = [row for row in session_rows if row["issues"]]
    return {
        "root": str(root),
        "total_sessions": len(session_rows),
        "sessions_with_issues": len(bad_sessions),
        "issue_counts": dict(sorted(issue_counts.items())),
        "sessions": session_rows,
    }


def _print_report(report: Dict[str, Any], *, only_issues: bool) -> None:
    print(f"Scanned sessions: {report['total_sessions']}")
    print(f"Sessions with issues: {report['sessions_with_issues']}")
    if report["issue_counts"]:
        print("Issue counts:")
        for issue_type, count in report["issue_counts"].items():
            print(f"  {issue_type}: {count}")

    rows = report["sessions"]
    if only_issues:
        rows = [row for row in rows if row["issues"]]

    if not rows:
        print("No sessions to print.")
        return

    print("")
    print("Sessions:")
    for row in rows:
        header = (
            f"  {row['experiment_id']} | {row['client']} | {row['model']} | "
            f"session={row['session_id']} | runs={row['run_count']}"
        )
        if row["issues"]:
            header += " | ISSUES"
        print(header)
        if row["seed_counts"]:
            seed_text = ", ".join(f"{seed}={count}" for seed, count in sorted(row["seed_counts"].items()))
            print(f"    seeds: {seed_text}")
        for issue in row["issues"]:
            print(f"    - {issue['type']}: {issue['details']}")


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    report = _build_report(root)
    _print_report(report, only_issues=bool(args.only_issues))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print("")
        print(f"Wrote JSON report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
