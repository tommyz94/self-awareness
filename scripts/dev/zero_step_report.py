#!/usr/bin/env python3
"""
Scan exported run JSON files and report zero-step failed runs.

Zero-step failures are runs with:
  - completed == false
  - steps == 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_RUN_JSON_RE = re.compile(r"run\d+\.json$")
_KNOWN_CLIENTS = {"ollama", "openrouter", "openai", "google", "anthropic"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize exported runs and flag zero-step failures."
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
        "--date",
        help="Optional session date filter in YYYYMMDD format.",
    )
    parser.add_argument(
        "--latest-date",
        action="store_true",
        help="Restrict the report to the most recent session date found in the scanned runs.",
    )
    return parser.parse_args()


def _iter_run_json_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("run*.json"):
        if _RUN_JSON_RE.fullmatch(path.name):
            yield path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _path_context(path: Path, root: Path) -> Dict[str, Optional[str]]:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts

    client_idx = None
    for idx, part in enumerate(rel_parts):
        if part in _KNOWN_CLIENTS:
            client_idx = idx
            break

    level_idx = next((i for i, part in enumerate(rel_parts) if part.startswith("level_")), None)

    client = rel_parts[client_idx] if client_idx is not None else None
    experiment_id = rel_parts[client_idx - 1] if client_idx is not None and client_idx >= 2 else None
    model = rel_parts[client_idx + 1] if client_idx is not None and client_idx + 1 < len(rel_parts) else None
    variant = None
    if client_idx is not None and level_idx is not None and client_idx + 2 < level_idx:
        variant = "/".join(rel_parts[client_idx + 2 : level_idx]) or None
    session_id = (
        rel_parts[level_idx + 1]
        if level_idx is not None and level_idx + 1 < len(rel_parts)
        else None
    )
    return {
        "experiment_id": experiment_id,
        "client": client,
        "model": model,
        "variant": variant,
        "session_id": session_id,
    }


def _run_record(path: Path, root: Path) -> Dict[str, Any]:
    data = _load_json(path)
    context = _path_context(path, root)

    completed = bool(data.get("completed", False))
    raw_steps = data.get("steps", 0)
    try:
        steps = int(raw_steps)
    except (TypeError, ValueError):
        steps = 0

    run_setup = data.get("run_setup") or {}
    record = {
        "path": str(path),
        "experiment_id": data.get("experiment_id") or context.get("experiment_id"),
        "client": context.get("client"),
        "model": context.get("model"),
        "variant": context.get("variant"),
        "session_id": context.get("session_id"),
        "session_date": _session_date(context.get("session_id")),
        "run_id": data.get("run_id"),
        "seed": run_setup.get("seed"),
        "completed": completed,
        "steps": steps,
        "zero_step_failed": (not completed and steps == 0),
    }
    return record


def _group_key(record: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("experiment_id") or "unknown"),
        str(record.get("client") or "unknown"),
        str(record.get("model") or "unknown"),
    )


def _session_key(record: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("experiment_id") or "unknown"),
        str(record.get("client") or "unknown"),
        str(record.get("model") or "unknown"),
        str(record.get("session_id") or "unknown"),
    )


def _session_date(session_id: Optional[str]) -> Optional[str]:
    if session_id is None:
        return None
    match = re.match(r"^(\d{8})_\d{6}(?:_\d+)?$", session_id)
    if match:
        return match.group(1)
    return None


def _date_key(record: Dict[str, Any]) -> str:
    return str(record.get("session_date") or "unknown")


def _build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    zero_step = [record for record in records if record["zero_step_failed"]]
    groups: Dict[tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"total_runs": 0, "zero_step_failed_runs": 0}
    )
    sessions: Dict[tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"total_runs": 0, "zero_step_failed_runs": 0, "zero_step_run_ids": []}
    )
    dates: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total_runs": 0, "zero_step_failed_runs": 0}
    )

    for record in records:
        group = groups[_group_key(record)]
        group["total_runs"] += 1
        if record["zero_step_failed"]:
            group["zero_step_failed_runs"] += 1

        date_bucket = dates[_date_key(record)]
        date_bucket["total_runs"] += 1
        if record["zero_step_failed"]:
            date_bucket["zero_step_failed_runs"] += 1

        session = sessions[_session_key(record)]
        session["total_runs"] += 1
        if record["zero_step_failed"]:
            session["zero_step_failed_runs"] += 1
            session["zero_step_run_ids"].append(record.get("run_id"))

    grouped = [
        {
            "experiment_id": experiment_id,
            "client": client,
            "model": model,
            **payload,
        }
        for (experiment_id, client, model), payload in sorted(groups.items())
    ]
    session_rows = [
        {
            "experiment_id": experiment_id,
            "client": client,
            "model": model,
            "session_id": session_id,
            **payload,
        }
        for (experiment_id, client, model, session_id), payload in sorted(sessions.items())
        if payload["zero_step_failed_runs"] > 0
    ]
    date_rows = [
        {
            "session_date": session_date,
            **payload,
        }
        for session_date, payload in sorted(dates.items(), reverse=True)
    ]

    return {
        "root": None,
        "total_runs": len(records),
        "zero_step_failed_runs": len(zero_step),
        "by_date": date_rows,
        "grouped": grouped,
        "affected_sessions": session_rows,
        "zero_step_runs": zero_step,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"Scanned runs: {report['total_runs']}")
    print(f"Zero-step failed runs: {report['zero_step_failed_runs']}")

    if not report["zero_step_runs"]:
        print("No zero-step failed runs found.")
        return

    print("")
    print("By date:")
    for row in report["by_date"]:
        if row["zero_step_failed_runs"] <= 0:
            continue
        print(
            "  "
            f"{row['session_date']}: {row['zero_step_failed_runs']}/{row['total_runs']} zero-step failed"
        )

    print("")
    print("By experiment/model:")
    for row in report["grouped"]:
        if row["zero_step_failed_runs"] <= 0:
            continue
        print(
            "  "
            f"{row['experiment_id']} | {row['client']} | {row['model']}: "
            f"{row['zero_step_failed_runs']}/{row['total_runs']} zero-step failed"
        )

    print("")
    print("Affected sessions:")
    for row in report["affected_sessions"]:
        run_ids = ", ".join(str(run_id) for run_id in row["zero_step_run_ids"])
        print(
            "  "
            f"{row['experiment_id']} | {row['client']} | {row['model']} | "
            f"session={row['session_id']}: {row['zero_step_failed_runs']}/{row['total_runs']} "
            f"zero-step failed | runs={run_ids}"
        )

    print("")
    print("Zero-step failed runs:")
    for row in report["zero_step_runs"]:
        variant = row.get("variant")
        variant_text = f" | variant={variant}" if variant else ""
        print(
            "  "
            f"exp={row['experiment_id']} | client={row['client']} | model={row['model']}"
            f"{variant_text} | date={row['session_date']} | session={row['session_id']} | run={row['run_id']} "
            f"| seed={row['seed']} | path={row['path']}"
        )


def _apply_date_filter(
    records: List[Dict[str, Any]],
    *,
    selected_date: Optional[str],
    latest_date: bool,
) -> List[Dict[str, Any]]:
    if selected_date and latest_date:
        raise SystemExit("Use either --date or --latest-date, not both.")
    if selected_date:
        if not re.fullmatch(r"\d{8}", selected_date):
            raise SystemExit("--date must use YYYYMMDD format.")
        return [record for record in records if record.get("session_date") == selected_date]
    if latest_date:
        known_dates = sorted(
            {
                record["session_date"]
                for record in records
                if record.get("session_date") not in (None, "unknown")
            }
        )
        if not known_dates:
            return []
        selected = known_dates[-1]
        return [record for record in records if record.get("session_date") == selected]
    return records


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    for path in sorted(_iter_run_json_paths(root)):
        try:
            records.append(_run_record(path, root))
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON: {path} ({exc})", file=sys.stderr)

    records = _apply_date_filter(
        records,
        selected_date=args.date,
        latest_date=args.latest_date,
    )
    report = _build_report(records)
    report["root"] = str(root)
    _print_report(report)

    if args.json_out:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2))
        print("")
        print(f"Saved JSON report to: {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
