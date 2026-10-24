#!/usr/bin/env python3
"""
Check whether models have enough repeated seeded runs to support error-bar metrics.

Default readiness rule:
- target experiments: e1,e2,e3,e5,e7
- each experiment needs at least 3 distinct seeds
- each counted seed must contribute exactly 7 runs

This matches the common "3 batches x 7 runs" setup used for accuracy/error-bar
aggregation in this repo.
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
_DEFAULT_EXPERIMENTS = ["e1", "e2", "e3", "e5", "e7"]
_FAMILY_PATTERNS = [
    (re.compile(r"gpt-5\.1", re.IGNORECASE), "gpt-5.1"),
    (re.compile(r"claude-sonnet-4\.6", re.IGNORECASE), "claude-sonnet-4.6"),
    (re.compile(r"gemini-2\.5-flash", re.IGNORECASE), "gemini-2.5-flash"),
    (re.compile(r"gemini-2\.5-pro", re.IGNORECASE), "gemini-2.5-pro"),
    (re.compile(r"(gemma4_26b|gemma-4-26b)", re.IGNORECASE), "gemma-4-26b"),
    (re.compile(r"(gemma4_31b|gemma-4-31b)", re.IGNORECASE), "gemma-4-31b"),
    (re.compile(r"(ministral-3_14b|ministral-14b)", re.IGNORECASE), "ministral-14b"),
    (re.compile(r"ministral-3_8b", re.IGNORECASE), "ministral-3-8b"),
    (re.compile(r"llava-phi3_3\.8b", re.IGNORECASE), "llava-phi3-3.8b"),
    (re.compile(r"qwen3\.6-plus", re.IGNORECASE), "qwen3.6-plus"),
    (re.compile(r"qwen-vl-max", re.IGNORECASE), "qwen-vl-max"),
    (re.compile(r"kimi-k2\.6", re.IGNORECASE), "kimi-k2.6"),
    (re.compile(r"^human$", re.IGNORECASE), "human"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report which models are ready for 3x7 seeded metric aggregation."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs",
        help="Root directory to scan recursively. Defaults to ./outputs",
    )
    parser.add_argument(
        "--group-by",
        choices=["exact", "family"],
        default="exact",
        help="Group runs by exact client/model or provider-agnostic model family. Default: exact.",
    )
    parser.add_argument(
        "--seed-rule",
        choices=["at-least", "exact"],
        default="at-least",
        help=(
            "How to count a valid seed block: 'at-least' means a seed counts if it has "
            "at least N runs and extras can be dropped; 'exact' requires exactly N runs. "
            "Default: at-least."
        ),
    )
    parser.add_argument(
        "--experiments",
        default=",".join(_DEFAULT_EXPERIMENTS),
        help="Comma-separated target experiments. Default: e1,e2,e3,e5,e7.",
    )
    parser.add_argument(
        "--runs-per-seed",
        type=int,
        default=7,
        help="Required number of runs in each counted seed block. Default: 7.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=3,
        help="Minimum number of valid seed blocks per experiment. Default: 3.",
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
        "--json-out",
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--mixed-seed-policy",
        choices=["run_json", "summary_if_available"],
        default="run_json",
        help=(
            "How to handle sessions whose run*.json files contain multiple seeds. "
            "'run_json' trusts the run files directly. "
            "'summary_if_available' overrides all runs in that session to the seed "
            "stored in experiment_summary.json when present."
        ),
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


def _model_family_name(model: Optional[str]) -> str:
    if model is None:
        return "unknown"
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.search(model):
            return family
    return model


def _run_record(path: Path, root: Path) -> Dict[str, Any]:
    data = _load_json(path)
    context = _path_context(path, root)
    run_setup = data.get("run_setup") or {}
    return {
        "path": str(path),
        "session_dir": str(path.parent.parent),
        "experiment_id": data.get("experiment_id") or context.get("experiment_id"),
        "client": context.get("client"),
        "model": context.get("model"),
        "session_id": context.get("session_id"),
        "session_date": _session_date(context.get("session_id")),
        "run_id": data.get("run_id"),
        "seed": run_setup.get("seed"),
        "completed": bool(data.get("completed", False)),
    }


def _apply_mixed_seed_policy(
    records: List[Dict[str, Any]],
    *,
    policy: str,
) -> List[Dict[str, Any]]:
    if policy == "run_json":
        return records

    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_session[str(record.get("session_dir") or "")].append(record)

    normalized: List[Dict[str, Any]] = []
    for session_dir, session_records in by_session.items():
        non_empty_seeds = {
            record.get("seed")
            for record in session_records
            if record.get("seed") not in (None, "")
        }
        if len(non_empty_seeds) <= 1:
            normalized.extend(session_records)
            continue

        summary_path = Path(session_dir) / "experiment_summary.json"
        if not summary_path.exists():
            normalized.extend(session_records)
            continue

        try:
            summary = _load_json(summary_path)
        except json.JSONDecodeError:
            normalized.extend(session_records)
            continue

        summary_seed = summary.get("seed")
        if summary_seed in (None, ""):
            normalized.extend(session_records)
            continue

        for record in session_records:
            patched = dict(record)
            patched["seed"] = summary_seed
            patched["seed_source"] = "summary"
            normalized.append(patched)

    return normalized


def _apply_filters(
    records: List[Dict[str, Any]],
    *,
    selected_date: Optional[str],
    latest_date: bool,
    clients: Optional[List[str]],
    models: Optional[List[str]],
    experiments: List[str],
) -> List[Dict[str, Any]]:
    if selected_date and latest_date:
        raise SystemExit("Use either --date or --latest-date, not both.")
    filtered = [record for record in records if record.get("experiment_id") in experiments]

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
    return filtered


def _seed_sort_key(seed_value: str) -> tuple[int, Any]:
    if seed_value == "unknown":
        return (1, seed_value)
    try:
        return (0, int(seed_value))
    except ValueError:
        return (0, seed_value)


def _build_report(
    records: List[Dict[str, Any]],
    *,
    group_by: str,
    seed_rule: str,
    experiments: List[str],
    runs_per_seed: int,
    min_seeds: int,
) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "completed_runs": 0,
            "seed_counts": defaultdict(int),
            "providers": set(),
            "exact_models": set(),
        }
    )
    for record in records:
        client = str(record.get("client") or "unknown")
        model = str(record.get("model") or "unknown")
        if group_by == "family":
            group_name = _model_family_name(record.get("model"))
        else:
            group_name = model
        key = (group_name, str(record.get("experiment_id") or "unknown"))
        bucket = grouped[key]
        bucket["runs"] += 1
        if record.get("completed"):
            bucket["completed_runs"] += 1
        bucket["providers"].add(client)
        bucket["exact_models"].add(f"{client} | {model}")
        raw_seed = record.get("seed")
        seed_key = "unknown" if raw_seed in (None, "") else str(raw_seed)
        bucket["seed_counts"][seed_key] += 1

    per_group: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "experiments": {},
            "ready_experiments": 0,
            "providers": set(),
            "exact_models": set(),
        }
    )

    for (group_name, experiment_id), payload in sorted(grouped.items()):
        seed_counts = dict(sorted(payload["seed_counts"].items(), key=lambda kv: _seed_sort_key(kv[0])))
        if seed_rule == "exact":
            valid_seed_blocks = [seed for seed, count in seed_counts.items() if count == runs_per_seed]
        else:
            valid_seed_blocks = [seed for seed, count in seed_counts.items() if count >= runs_per_seed]
        extra_runs = sum(max(0, count - runs_per_seed) for seed, count in seed_counts.items() if seed in valid_seed_blocks)
        usable_runs = sum(min(count, runs_per_seed) for seed, count in seed_counts.items() if seed in valid_seed_blocks)
        issues: List[str] = []
        if payload["runs"] < min_seeds * runs_per_seed:
            issues.append(
                f"only {payload['runs']} total runs (need >= {min_seeds * runs_per_seed})"
            )
        if len(seed_counts) < min_seeds:
            issues.append(f"only {len(seed_counts)} distinct seeds (need >= {min_seeds})")
        if len(valid_seed_blocks) < min_seeds:
            comparator = "==" if seed_rule == "exact" else ">="
            issues.append(
                f"only {len(valid_seed_blocks)} seeds with {comparator}{runs_per_seed} runs"
            )
        experiment_row = {
            "experiment": experiment_id,
            "runs": payload["runs"],
            "completed_runs": payload["completed_runs"],
            "incomplete_runs": payload["runs"] - payload["completed_runs"],
            "distinct_seeds": len(seed_counts),
            "valid_seed_block_count": len(valid_seed_blocks),
            "valid_seeds": [None if seed == "unknown" else seed for seed in valid_seed_blocks],
            "seed_counts": {None if seed == "unknown" else seed: count for seed, count in seed_counts.items()},
            "usable_runs_after_trimming": usable_runs,
            "droppable_extra_runs": extra_runs,
            "issues": issues,
            "metric_ready": len(valid_seed_blocks) >= min_seeds,
        }
        group_bucket = per_group[group_name]
        group_bucket["experiments"][experiment_id] = experiment_row
        group_bucket["ready_experiments"] += int(experiment_row["metric_ready"])
        group_bucket["providers"].update(payload["providers"])
        group_bucket["exact_models"].update(payload["exact_models"])

    rows: List[Dict[str, Any]] = []
    for group_name, payload in sorted(per_group.items()):
        experiment_rows = []
        for experiment_id in experiments:
            experiment_rows.append(
                payload["experiments"].get(
                    experiment_id,
                    {
                        "experiment": experiment_id,
                        "runs": 0,
                        "completed_runs": 0,
                        "incomplete_runs": 0,
                        "distinct_seeds": 0,
                        "valid_seed_block_count": 0,
                        "valid_seeds": [],
                        "seed_counts": {},
                        "usable_runs_after_trimming": 0,
                        "droppable_extra_runs": 0,
                        "issues": [
                            f"only 0 total runs (need >= {min_seeds * runs_per_seed})",
                            f"only 0 distinct seeds (need >= {min_seeds})",
                            f"only 0 seeds with {'==' if seed_rule == 'exact' else '>='}{runs_per_seed} runs",
                        ],
                        "metric_ready": False,
                    },
                )
            )
        row = {
            "ready_experiments": payload["ready_experiments"],
            "target_experiments": len(experiments),
            "fully_ready": payload["ready_experiments"] == len(experiments),
            "experiments": experiment_rows,
        }
        if group_by == "family":
            row["family"] = group_name
            row["providers"] = sorted(payload["providers"])
            row["exact_models"] = sorted(payload["exact_models"])
        else:
            provider_model = sorted(payload["exact_models"])
            if provider_model:
                client, model = provider_model[0].split(" | ", 1)
            else:
                client, model = "unknown", group_name
            row["client"] = client
            row["model"] = model
        rows.append(row)

    return {
        "root": None,
        "group_by": group_by,
        "seed_rule": seed_rule,
        "target_experiments": experiments,
        "runs_per_seed": runs_per_seed,
        "min_seeds": min_seeds,
        "models": rows,
        "fully_ready_models": (
            [
                {
                    "family": row["family"],
                    "providers": row["providers"],
                    "exact_models": row["exact_models"],
                }
                for row in rows
                if row["fully_ready"]
            ]
            if group_by == "family"
            else [
                {"client": row["client"], "model": row["model"]}
                for row in rows
                if row["fully_ready"]
            ]
        ),
    }


def _print_report(report: Dict[str, Any]) -> None:
    comparator = "==" if report["seed_rule"] == "exact" else ">="
    print(
        "Readiness rule: "
        f"{report['min_seeds']} seeds x {comparator}{report['runs_per_seed']} runs "
        f"for each of {', '.join(report['target_experiments'])}"
    )
    print("")
    print("By model:")
    for row in report["models"]:
        if report["group_by"] == "family":
            providers = ",".join(row["providers"])
            print(
                "  "
                f"{row['family']} | providers={providers} | "
                f"exact_variants={len(row['exact_models'])} | "
                f"ready_experiments={row['ready_experiments']}/{row['target_experiments']}"
            )
        else:
            print(
                "  "
                f"{row['client']} | {row['model']} | "
                f"ready_experiments={row['ready_experiments']}/{row['target_experiments']}"
            )
        parts = []
        for experiment_row in row["experiments"]:
            marker = " READY" if experiment_row["metric_ready"] else ""
            parts.append(
                f"{experiment_row['experiment']}:"
                f"{experiment_row['runs']}runs/"
                f"{experiment_row['distinct_seeds']}seeds/"
                f"{experiment_row['valid_seed_block_count']}x{comparator}{report['runs_per_seed']}"
                f"{marker}"
            )
        print("    " + " ; ".join(parts))
        for experiment_row in row["experiments"]:
            if experiment_row["metric_ready"]:
                if experiment_row["droppable_extra_runs"] > 0:
                    print(
                        "      "
                        f"{experiment_row['experiment']}: droppable_extra_runs="
                        f"{experiment_row['droppable_extra_runs']} "
                        f"(usable_runs={experiment_row['usable_runs_after_trimming']})"
                    )
            elif experiment_row["issues"]:
                print("      " + f"{experiment_row['experiment']}: " + "; ".join(experiment_row["issues"]))

    print("")
    print("Fully ready models:")
    if not report["fully_ready_models"]:
        print("  none")
        return
    for row in report["fully_ready_models"]:
        if report["group_by"] == "family":
            providers = ",".join(row["providers"])
            print(f"  {row['family']} | providers={providers}")
        else:
            print(f"  {row['client']} | {row['model']}")


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 1
    if args.runs_per_seed <= 0:
        raise SystemExit("--runs-per-seed must be positive.")
    if args.min_seeds <= 0:
        raise SystemExit("--min-seeds must be positive.")

    target_experiments = _comma_list(args.experiments) or []
    if not target_experiments:
        raise SystemExit("No target experiments selected.")

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
        experiments=target_experiments,
    )
    records = _apply_mixed_seed_policy(
        records,
        policy=args.mixed_seed_policy,
    )

    report = _build_report(
        records,
        group_by=args.group_by,
        seed_rule=args.seed_rule,
        experiments=target_experiments,
        runs_per_seed=args.runs_per_seed,
        min_seeds=args.min_seeds,
    )
    report["root"] = str(root)
    report["mixed_seed_policy"] = args.mixed_seed_policy
    _print_report(report)

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
