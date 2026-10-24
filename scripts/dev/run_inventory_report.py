#!/usr/bin/env python3
"""
Scan exported run JSON files and summarize run counts by provider/model.

The report is intentionally inventory-focused:
- total runs per client/model
- per-experiment counts inside each model bucket
- per-seed counts inside each model bucket
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
        description="Summarize exported run counts by provider/model and seed."
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
    parser.add_argument(
        "--clients",
        help="Optional comma-separated client filter.",
    )
    parser.add_argument(
        "--models",
        help="Optional comma-separated model filter.",
    )
    parser.add_argument(
        "--experiments",
        help="Optional comma-separated experiment filter.",
    )
    parser.add_argument(
        "--no-seed-breakdown",
        action="store_true",
        help="Suppress the per-seed section in the text report.",
    )
    return parser.parse_args()


def _comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    values = [part for part in parts if part]
    return values or None


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
    session_id = (
        rel_parts[level_idx + 1]
        if level_idx is not None and level_idx + 1 < len(rel_parts)
        else None
    )
    return {
        "experiment_id": experiment_id,
        "client": client,
        "model": model,
        "session_id": session_id,
    }


def _session_date(session_id: Optional[str]) -> Optional[str]:
    if session_id is None:
        return None
    match = re.match(r"^(\d{8})_\d{6}(?:_\d+)?$", session_id)
    if match:
        return match.group(1)
    return None


def _run_record(path: Path, root: Path) -> Dict[str, Any]:
    data = _load_json(path)
    context = _path_context(path, root)
    run_setup = data.get("run_setup") or {}
    return {
        "path": str(path),
        "experiment_id": data.get("experiment_id") or context.get("experiment_id"),
        "client": context.get("client"),
        "model": context.get("model"),
        "session_id": context.get("session_id"),
        "session_date": _session_date(context.get("session_id")),
        "run_id": data.get("run_id"),
        "seed": run_setup.get("seed"),
        "completed": bool(data.get("completed", False)),
    }


def _apply_filters(
    records: List[Dict[str, Any]],
    *,
    selected_date: Optional[str],
    latest_date: bool,
    clients: Optional[List[str]],
    models: Optional[List[str]],
    experiments: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if selected_date and latest_date:
        raise SystemExit("Use either --date or --latest-date, not both.")
    filtered = list(records)

    if selected_date:
        if not re.fullmatch(r"\d{8}", selected_date):
            raise SystemExit("--date must use YYYYMMDD format.")
        filtered = [record for record in filtered if record.get("session_date") == selected_date]
    elif latest_date:
        known_dates = sorted(
            {
                record["session_date"]
                for record in filtered
                if record.get("session_date") not in (None, "unknown")
            }
        )
        if not known_dates:
            return []
        selected = known_dates[-1]
        filtered = [record for record in filtered if record.get("session_date") == selected]

    if clients:
        allowed = set(clients)
        filtered = [record for record in filtered if record.get("client") in allowed]
    if models:
        allowed = set(models)
        filtered = [record for record in filtered if record.get("model") in allowed]
    if experiments:
        allowed = set(experiments)
        filtered = [record for record in filtered if record.get("experiment_id") in allowed]
    return filtered


def _sorted_counter(mapping: Dict[str, int]) -> List[Dict[str, Any]]:
    return [
        {"name": name, "count": mapping[name]}
        for name in sorted(mapping)
    ]


def _seed_sort_key(seed_value: str) -> tuple[int, Any]:
    if seed_value == "unknown":
        return (1, seed_value)
    try:
        return (0, int(seed_value))
    except ValueError:
        return (0, seed_value)


def _build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "total_runs": 0,
            "completed_runs": 0,
            "session_ids": set(),
            "experiments": defaultdict(int),
            "seeds": defaultdict(lambda: {"count": 0, "session_ids": set(), "experiments": defaultdict(int)}),
        }
    )

    for record in records:
        key = (
            str(record.get("client") or "unknown"),
            str(record.get("model") or "unknown"),
        )
        bucket = by_model[key]
        bucket["total_runs"] += 1
        if record.get("completed"):
            bucket["completed_runs"] += 1
        session_id = str(record.get("session_id") or "unknown")
        bucket["session_ids"].add(session_id)

        experiment_id = str(record.get("experiment_id") or "unknown")
        bucket["experiments"][experiment_id] += 1

        raw_seed = record.get("seed")
        seed_key = "unknown" if raw_seed in (None, "") else str(raw_seed)
        seed_bucket = bucket["seeds"][seed_key]
        seed_bucket["count"] += 1
        seed_bucket["session_ids"].add(session_id)
        seed_bucket["experiments"][experiment_id] += 1

    grouped: List[Dict[str, Any]] = []
    for (client, model), payload in sorted(by_model.items()):
        seed_rows = []
        for seed_key in sorted(payload["seeds"], key=_seed_sort_key):
            seed_payload = payload["seeds"][seed_key]
            seed_rows.append(
                {
                    "seed": None if seed_key == "unknown" else seed_key,
                    "count": seed_payload["count"],
                    "session_count": len(seed_payload["session_ids"]),
                    "experiments": _sorted_counter(seed_payload["experiments"]),
                }
            )

        grouped.append(
            {
                "client": client,
                "model": model,
                "total_runs": payload["total_runs"],
                "completed_runs": payload["completed_runs"],
                "incomplete_runs": payload["total_runs"] - payload["completed_runs"],
                "session_count": len(payload["session_ids"]),
                "seed_count": len(payload["seeds"]),
                "experiments": _sorted_counter(payload["experiments"]),
                "seeds": seed_rows,
            }
        )

    return {
        "root": None,
        "total_runs": len(records),
        "model_buckets": len(grouped),
        "grouped": grouped,
    }


def _fmt_counts(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['name']}={row['count']}" for row in rows)


def _print_report(report: Dict[str, Any], *, show_seed_breakdown: bool) -> None:
    print(f"Scanned runs: {report['total_runs']}")
    print(f"Model buckets: {report['model_buckets']}")

    if not report["grouped"]:
        print("No matching run JSON files found.")
        return

    print("")
    print("By model:")
    for row in report["grouped"]:
        print(
            "  "
            f"{row['client']} | {row['model']}: "
            f"runs={row['total_runs']} | completed={row['completed_runs']} | "
            f"incomplete={row['incomplete_runs']} | sessions={row['session_count']} | "
            f"seeds={row['seed_count']}"
        )
        print(f"    experiments: {_fmt_counts(row['experiments'])}")
        if not show_seed_breakdown:
            continue
        print("    seeds:")
        for seed_row in row["seeds"]:
            seed_value = seed_row["seed"] if seed_row["seed"] is not None else "unknown"
            experiment_counts = ", ".join(
                f"{item['name']}={item['count']}" for item in seed_row["experiments"]
            )
            print(
                "      "
                f"{seed_value}: runs={seed_row['count']} | sessions={seed_row['session_count']} "
                f"| experiments={experiment_counts or 'none'}"
            )


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

    records = _apply_filters(
        records,
        selected_date=args.date,
        latest_date=args.latest_date,
        clients=_comma_list(args.clients),
        models=_comma_list(args.models),
        experiments=_comma_list(args.experiments),
    )

    report = _build_report(records)
    report["root"] = str(root)
    _print_report(report, show_seed_breakdown=not args.no_seed_breakdown)

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
