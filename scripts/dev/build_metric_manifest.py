#!/usr/bin/env python3
"""
Build a deterministic manifest of runs to use for final metrics.

This script does not modify raw outputs. Instead, it applies a reproducible
selection policy so downstream metrics can read an explicit list of kept runs.

Default policy:
- root: ./final_outputs
- experiments: e1,e2,e3,e5,e7
- group models provider-agnostically by family
- treat mixed-seed sessions using experiment_summary.json when available
- require at least 7 runs for a seed to be usable
- keep exactly the first 7 runs from each selected seed, ordered
  deterministically by session id and run_id
- keep at most 3 valid seeds per model/experiment
- when more than 3 valid seeds exist, pick the first 3 by deterministic
  seed order only

This intentionally avoids quality-based run selection so the final subset does
not drift into post-hoc curation or cherry-picking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.metric_readiness_report import (
    _apply_filters,
    _apply_mixed_seed_policy,
    _comma_list,
    _iter_run_json_paths,
    _model_family_name,
    _run_record,
    _seed_sort_key,
)


_DEFAULT_EXPERIMENTS = ["e1", "e2", "e3", "e5", "e7"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic manifest of metric-eligible runs."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="final_outputs",
        help="Root directory to scan recursively. Defaults to ./final_outputs",
    )
    parser.add_argument(
        "--group-by",
        choices=["exact", "family"],
        default="family",
        help="Group by exact provider/model or provider-agnostic family. Default: family.",
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
        help="Number of runs to keep from each selected seed. Default: 7.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=3,
        help=(
            "Maximum number of valid seeds to keep per model/experiment. "
            "Use 0 to keep all valid seeds. Default: 3."
        ),
    )
    parser.add_argument(
        "--seed-selection",
        choices=["minimal-trimming", "seed-order"],
        default="seed-order",
        help=(
            "How to choose seeds when more than --max-seeds are valid. "
            "'minimal-trimming' prefers seeds closest to the target run count; "
            "'seed-order' uses deterministic seed ordering only. Default: seed-order."
        ),
    )
    parser.add_argument(
        "--mixed-seed-policy",
        choices=["run_json", "summary_if_available"],
        default="summary_if_available",
        help=(
            "How to handle sessions whose run*.json files contain multiple seeds. "
            "Default: summary_if_available."
        ),
    )
    parser.add_argument(
        "--date",
        help="Optional session date filter in YYYYMMDD format.",
    )
    parser.add_argument(
        "--latest-date",
        action="store_true",
        help="Restrict the manifest to the most recent session date found.",
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
        help="Optional path for a machine-readable JSON manifest/report.",
    )
    parser.add_argument(
        "--csv-out",
        help="Optional path for a flat CSV manifest of kept runs.",
    )
    return parser.parse_args()


def _run_id_sort_key(run_id: Any) -> Tuple[int, Any]:
    if run_id is None:
        return (1, "")
    try:
        return (0, int(run_id))
    except (TypeError, ValueError):
        return (0, str(run_id))


def _run_sort_key(record: Dict[str, Any]) -> Tuple[str, Tuple[int, Any], str]:
    return (
        str(record.get("session_id") or ""),
        _run_id_sort_key(record.get("run_id")),
        str(record.get("path") or ""),
    )


def _session_recency_sort_value(session_id: Optional[str]) -> int:
    raw = str(session_id or "").replace("_", "")
    try:
        return int(raw)
    except ValueError:
        return -1


def _selected_seed_sort_key(
    seed_key: str,
    summary: Dict[str, Any],
    *,
    runs_per_seed: int,
    strategy: str,
) -> Tuple[Any, ...]:
    if strategy == "seed-order":
        return _seed_sort_key(seed_key)
    return (
        -int(summary["usable_kept_count"]),
        -_session_recency_sort_value(summary["latest_session_id"]),
        max(0, int(summary["available_runs"]) - runs_per_seed),
        *_seed_sort_key(seed_key),
    )


def _group_identity(record: Dict[str, Any], group_by: str) -> Tuple[str, Dict[str, Any]]:
    client = str(record.get("client") or "unknown")
    model = str(record.get("model") or "unknown")
    if group_by == "family":
        family = _model_family_name(record.get("model"))
        return family, {
            "family": family,
            "providers": {client},
            "exact_models": {f"{client} | {model}"},
        }
    return f"{client} | {model}", {
        "client": client,
        "model": model,
        "providers": {client},
        "exact_models": {f"{client} | {model}"},
    }


def _build_manifest(
    records: List[Dict[str, Any]],
    *,
    group_by: str,
    experiments: List[str],
    runs_per_seed: int,
    max_seeds: int,
    seed_selection: str,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    group_meta: Dict[str, Dict[str, Any]] = {}

    for record in records:
        group_key, meta = _group_identity(record, group_by)
        if group_key not in group_meta:
            group_meta[group_key] = meta
        else:
            group_meta[group_key]["providers"].update(meta["providers"])
            group_meta[group_key]["exact_models"].update(meta["exact_models"])

        experiment_id = str(record.get("experiment_id") or "unknown")
        raw_seed = record.get("seed")
        seed_key = "unknown" if raw_seed in (None, "") else str(raw_seed)
        grouped[group_key][experiment_id][seed_key].append(record)

    manifest_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []

    for group_key in sorted(grouped):
        meta = group_meta[group_key]
        model_row: Dict[str, Any]
        if group_by == "family":
            model_row = {
                "family": meta["family"],
                "providers": sorted(meta["providers"]),
                "exact_models": sorted(meta["exact_models"]),
                "experiments": [],
            }
        else:
            model_row = {
                "client": meta["client"],
                "model": meta["model"],
                "experiments": [],
            }

        for experiment_id in experiments:
            seed_buckets = grouped[group_key].get(experiment_id, {})
            seed_summaries: Dict[str, Dict[str, Any]] = {}
            for seed_key, records_for_seed in seed_buckets.items():
                ordered_for_selection = sorted(records_for_seed, key=_run_sort_key)
                candidate_kept = ordered_for_selection[:runs_per_seed]
                seed_summaries[seed_key] = {
                    "available_runs": len(records_for_seed),
                    "usable_kept_count": sum(int(bool(record.get("usable", False))) for record in candidate_kept),
                    "latest_session_id": max(
                        (str(record.get("session_id") or "") for record in records_for_seed),
                        default="",
                    ),
                    "ordered_for_selection": ordered_for_selection,
                }

            seed_counts = {
                seed_key: summary["available_runs"]
                for seed_key, summary in seed_summaries.items()
            }
            valid_seed_keys = [
                seed_key
                for seed_key, count in seed_counts.items()
                if count >= runs_per_seed
            ]
            valid_seed_keys = sorted(
                valid_seed_keys,
                key=lambda seed_key: _selected_seed_sort_key(
                    seed_key,
                    seed_summaries[seed_key],
                    runs_per_seed=runs_per_seed,
                    strategy=seed_selection,
                ),
            )
            if max_seeds > 0:
                selected_seed_keys = valid_seed_keys[:max_seeds]
                dropped_valid_seed_keys = valid_seed_keys[max_seeds:]
            else:
                selected_seed_keys = valid_seed_keys
                dropped_valid_seed_keys = []

            selected_seed_rows: List[Dict[str, Any]] = []
            selected_run_total = 0
            dropped_extra_runs = 0

            for selected_seed_index, seed_key in enumerate(selected_seed_keys, start=1):
                ordered_records = seed_summaries[seed_key]["ordered_for_selection"]
                kept_records = ordered_records[:runs_per_seed]
                dropped_records = ordered_records[runs_per_seed:]
                selected_run_total += len(kept_records)
                dropped_extra_runs += len(dropped_records)

                kept_run_rows = []
                for kept_index, record in enumerate(kept_records, start=1):
                    manifest_row = {
                        "group_by": group_by,
                        "experiment_id": experiment_id,
                        "seed": None if seed_key == "unknown" else seed_key,
                        "seed_rank": selected_seed_index,
                        "kept_index": kept_index,
                        "run_id": record.get("run_id"),
                        "session_id": record.get("session_id"),
                        "session_dir": record.get("session_dir"),
                        "path": record.get("path"),
                        "client": record.get("client"),
                        "model": record.get("model"),
                        "completed": bool(record.get("completed", False)),
                    }
                    if group_by == "family":
                        manifest_row["family"] = group_key
                    else:
                        manifest_row["provider_model"] = group_key
                    manifest_rows.append(manifest_row)
                    kept_run_rows.append(
                        {
                            "run_id": record.get("run_id"),
                            "session_id": record.get("session_id"),
                            "path": record.get("path"),
                            "completed": bool(record.get("completed", False)),
                        }
                    )

                selected_seed_rows.append(
                    {
                        "seed": None if seed_key == "unknown" else seed_key,
                        "available_runs": seed_counts[seed_key],
                        "kept_runs": len(kept_records),
                        "dropped_runs": len(dropped_records),
                        "usable_kept_runs": seed_summaries[seed_key]["usable_kept_count"],
                        "latest_session_id": seed_summaries[seed_key]["latest_session_id"],
                        "kept_run_ids": [record.get("run_id") for record in kept_records],
                        "kept_runs_manifest": kept_run_rows,
                    }
                )

            ignored_short_seeds = [
                {
                    "seed": None if seed_key == "unknown" else seed_key,
                    "count": count,
                }
                for seed_key, count in sorted(seed_counts.items(), key=lambda kv: _seed_sort_key(kv[0]))
                if count < runs_per_seed
            ]
            dropped_valid_seeds = [
                {
                    "seed": None if seed_key == "unknown" else seed_key,
                    "available_runs": seed_counts[seed_key],
                }
                for seed_key in dropped_valid_seed_keys
            ]

            experiment_row = {
                "experiment": experiment_id,
                "distinct_seeds_seen": len(seed_counts),
                "seed_counts": {
                    None if seed_key == "unknown" else seed_key: count
                    for seed_key, count in sorted(seed_counts.items(), key=lambda kv: _seed_sort_key(kv[0]))
                },
                "valid_seed_count": len(valid_seed_keys),
                "selected_seed_count": len(selected_seed_keys),
                "selected_run_count": selected_run_total,
                "dropped_extra_runs": dropped_extra_runs,
                "dropped_valid_seeds": dropped_valid_seeds,
                "ignored_short_seeds": ignored_short_seeds,
                "selected_seeds": selected_seed_rows,
            }
            model_row["experiments"].append(experiment_row)

        model_rows.append(model_row)

    return {
        "group_by": group_by,
        "target_experiments": experiments,
        "runs_per_seed": runs_per_seed,
        "max_seeds": max_seeds,
        "seed_selection": seed_selection,
        "models": model_rows,
        "manifest_rows": manifest_rows,
    }


def _print_report(report: Dict[str, Any]) -> None:
    max_seeds = report["max_seeds"]
    seed_text = "all valid seeds" if max_seeds == 0 else f"at most {max_seeds} seeds"
    print(
        "Selection rule: "
        f"{seed_text}, exactly {report['runs_per_seed']} runs per kept seed, "
        f"experiments={','.join(report['target_experiments'])}, "
        f"seed_selection={report['seed_selection']}"
    )
    print("")
    for model_row in report["models"]:
        if report["group_by"] == "family":
            providers = ",".join(model_row["providers"])
            print(f"{model_row['family']} | providers={providers}")
        else:
            print(f"{model_row['client']} | {model_row['model']}")
        for experiment_row in model_row["experiments"]:
            dropped_valid = len(experiment_row["dropped_valid_seeds"])
            short = len(experiment_row["ignored_short_seeds"])
            print(
                "  "
                f"{experiment_row['experiment']}: "
                f"selected={experiment_row['selected_run_count']} runs "
                f"({experiment_row['selected_seed_count']} seeds x {report['runs_per_seed']}) | "
                f"valid_seeds={experiment_row['valid_seed_count']} | "
                f"dropped_extra_runs={experiment_row['dropped_extra_runs']} | "
                f"dropped_valid_seeds={dropped_valid} | "
                f"ignored_short_seeds={short}"
            )
            if experiment_row["selected_seeds"]:
                seed_text = ", ".join(
                    f"{seed_row['seed']}={seed_row['kept_runs']}/{seed_row['available_runs']}"
                    for seed_row in experiment_row["selected_seeds"]
                )
                print(f"    kept seeds: {seed_text}")
            if experiment_row["dropped_valid_seeds"]:
                dropped_text = ", ".join(
                    f"{seed_row['seed']}={seed_row['available_runs']}"
                    for seed_row in experiment_row["dropped_valid_seeds"]
                )
                print(f"    dropped valid seeds: {dropped_text}")
            if experiment_row["ignored_short_seeds"]:
                short_text = ", ".join(
                    f"{seed_row['seed']}={seed_row['count']}"
                    for seed_row in experiment_row["ignored_short_seeds"]
                )
                print(f"    ignored short seeds: {short_text}")
        print("")


def _write_csv_manifest(path: Path, manifest_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_by",
        "family",
        "provider_model",
        "experiment_id",
        "seed",
        "seed_rank",
        "kept_index",
        "run_id",
        "session_id",
        "session_dir",
        "path",
        "client",
        "model",
        "completed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            output_row = {name: row.get(name) for name in fieldnames}
            writer.writerow(output_row)


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 1
    if args.runs_per_seed <= 0:
        raise SystemExit("--runs-per-seed must be positive.")
    if args.max_seeds < 0:
        raise SystemExit("--max-seeds cannot be negative.")

    target_experiments = _comma_list(args.experiments) or []
    if not target_experiments:
        raise SystemExit("No target experiments selected.")

    records: List[Dict[str, Any]] = []
    for path in sorted(_iter_run_json_paths(root)):
        try:
            record = _run_record(path, root)
            data = json.loads(path.read_text())
            step_records = list(data.get("step_records") or [])
            record["step_record_count"] = len(step_records)
            record["usable"] = bool(step_records)
            records.append(record)
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
    records = _apply_mixed_seed_policy(records, policy=args.mixed_seed_policy)

    report = _build_manifest(
        records,
        group_by=args.group_by,
        experiments=target_experiments,
        runs_per_seed=args.runs_per_seed,
        max_seeds=args.max_seeds,
        seed_selection=args.seed_selection,
    )
    report["root"] = str(root)
    report["mixed_seed_policy"] = args.mixed_seed_policy
    report["scanned_runs"] = len(records)

    _print_report(report)

    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON manifest: {json_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser().resolve()
        _write_csv_manifest(csv_path, report["manifest_rows"])
        print(f"Wrote CSV manifest: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
