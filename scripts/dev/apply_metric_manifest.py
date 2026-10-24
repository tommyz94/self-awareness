#!/usr/bin/env python3
"""
Apply a metric manifest to an outputs tree in place.

Intended use:
- start from a disposable copy such as ./final_outputs
- build a manifest with scripts/dev/build_metric_manifest.py
- apply it here so directory-walking scripts only see the selected runs

What it does:
- removes run_* directories not present in the manifest
- rewrites mixed-seed kept runs to the summary seed when needed
- rewrites experiment_summary.json to match the kept runs for touched sessions
- deletes empty session directories after pruning
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a metric manifest to an outputs tree in place."
    )
    parser.add_argument(
        "manifest",
        help="Path to metric_manifest.json",
    )
    parser.add_argument(
        "--root",
        help="Optional outputs root override. Defaults to the root stored in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _summary_seed(summary_path: Path) -> Optional[Any]:
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except json.JSONDecodeError:
        return None
    return summary.get("seed")


def _recompute_summary_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    runs = list(summary.get("runs") or [])
    summary["num_runs"] = len(runs)

    if runs:
        correct = sum(bool(run.get("correct")) for run in runs)
        summary["sia"] = round(correct / len(runs), 4)
    else:
        summary["sia"] = 0.0

    relevant = [
        run
        for run in runs
        if run.get("completed") and run.get("wrong_color") is not None
    ]
    if not relevant:
        summary["gi"] = None
        return summary

    visual = sum(run.get("selected_cube") == run.get("agent_color") for run in relevant)
    text = sum(run.get("selected_cube") == run.get("wrong_color") for run in relevant)
    if visual + text == 0:
        summary["gi"] = None
    else:
        summary["gi"] = round(visual / (visual + text), 4)
    return summary


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"Manifest does not exist: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_json(manifest_path)
    root = Path(args.root).expanduser().resolve() if args.root else Path(manifest["root"]).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 1

    kept_paths: Set[Path] = set()
    kept_run_ids_by_session: Dict[Path, Set[int]] = defaultdict(set)
    kept_rows_by_session: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest.get("manifest_rows", []):
        run_path = Path(row["path"]).resolve()
        session_dir = Path(row["session_dir"]).resolve()
        kept_paths.add(run_path)
        run_id = row.get("run_id")
        if run_id is not None:
            kept_run_ids_by_session[session_dir].add(int(run_id))
        kept_rows_by_session[session_dir].append(row)

    all_run_json_paths = {
        path.resolve()
        for path in root.rglob("run*.json")
        if path.name.startswith("run") and path.name.endswith(".json")
    }
    remove_run_json_paths = sorted(all_run_json_paths - kept_paths)

    touched_sessions: Set[Path] = set()
    removed_run_dirs = 0
    patched_run_jsons = 0
    rewritten_summaries = 0
    removed_session_dirs = 0

    for run_json_path in remove_run_json_paths:
        run_dir = run_json_path.parent
        session_dir = run_dir.parent
        touched_sessions.add(session_dir)
        if args.dry_run:
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
            removed_run_dirs += 1

    for session_dir, rows in kept_rows_by_session.items():
        touched_sessions.add(session_dir)
        summary_path = session_dir / "experiment_summary.json"
        summary_seed = _summary_seed(summary_path)
        if summary_seed is None:
            continue

        run_paths = []
        seed_values = set()
        for row in rows:
            run_path = Path(row["path"]).resolve()
            if run_path.exists():
                run_paths.append(run_path)
                try:
                    run_data = _load_json(run_path)
                except json.JSONDecodeError:
                    continue
                raw_seed = (run_data.get("run_setup") or {}).get("seed")
                if raw_seed not in (None, ""):
                    seed_values.add(raw_seed)

        if len(seed_values) <= 1 and (not seed_values or summary_seed in seed_values):
            continue

        for run_path in run_paths:
            if args.dry_run:
                continue
            try:
                run_data = _load_json(run_path)
            except json.JSONDecodeError:
                continue
            run_setup = dict(run_data.get("run_setup") or {})
            if run_setup.get("seed") == summary_seed:
                continue
            run_setup["seed"] = summary_seed
            run_data["run_setup"] = run_setup
            _write_json(run_path, run_data)
            patched_run_jsons += 1

    for session_dir in sorted(touched_sessions):
        remaining_run_jsons = sorted(
            path
            for path in session_dir.rglob("run*.json")
            if path.name.startswith("run") and path.name.endswith(".json")
        )
        if not remaining_run_jsons:
            if args.dry_run:
                continue
            if session_dir.exists():
                shutil.rmtree(session_dir)
                removed_session_dirs += 1
            continue

        summary_path = session_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
        except json.JSONDecodeError:
            continue

        kept_run_ids = kept_run_ids_by_session.get(session_dir)
        if not kept_run_ids:
            continue
        filtered_runs = [
            run
            for run in list(summary.get("runs") or [])
            if run.get("run_id") in kept_run_ids
        ]
        summary["runs"] = filtered_runs
        summary = _recompute_summary_metrics(summary)
        if args.dry_run:
            continue
        _write_json(summary_path, summary)
        rewritten_summaries += 1

    print(f"Root: {root}")
    print(f"Kept run JSONs: {len(kept_paths)}")
    print(f"Removed extra run dirs: {removed_run_dirs}{' (dry-run)' if args.dry_run else ''}")
    print(f"Patched mixed-seed run JSONs: {patched_run_jsons}{' (dry-run)' if args.dry_run else ''}")
    print(f"Rewritten summaries: {rewritten_summaries}{' (dry-run)' if args.dry_run else ''}")
    print(f"Removed empty session dirs: {removed_session_dirs}{' (dry-run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
