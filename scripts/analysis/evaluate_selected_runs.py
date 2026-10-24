#!/usr/bin/env python3
"""Calculate benchmark metrics for episodes listed in a selection manifest.

Only raw JSON files named in the manifest, plus an optional explicitly selected
additional cohort, enter the calculation. Originals are copied byte for byte
into a temporary view; experiment.py supplies every metric formula.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.evaluation.data_ids import CALCULATOR_TO_PAPER, PAPER_TO_CALCULATOR, calculator_id

CALCULATOR = ROOT / "experiment.py"
DATA_ADAPTER = ROOT / "experiments/evaluation/data_ids.py"
EXPERIMENTS = {"e1", "e2", "e3", "e5", "e7"}
SOURCE_EXPERIMENTS = EXPERIMENTS | set(PAPER_TO_CALCULATOR)
CORE_METRICS = ("TSA", "TTD", "MCR", "MTATO", "CAAL", "CONFABULATION")
RUN_NAME = re.compile(r"run\d+\.json")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def select_runs(manifest, *, data_root, additional_root=None):
    """Validate input files before the calculator can silently skip bad JSON."""
    document = json.loads(manifest.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Manifest must be a JSON object containing manifest_rows")
    rows = document.get("manifest_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty manifest_rows list")
    selected = []
    seen = set()

    def add(path, experiment, cohort, expected_hash=None, scheme=None):
        path = path.resolve(strict=True)
        if path in seen:
            raise ValueError(f"Duplicate selected run: {path}")
        if experiment not in SOURCE_EXPERIMENTS or not RUN_NAME.fullmatch(path.name):
            raise ValueError(f"Unsupported experiment or run filename: {path}")
        exp_dir = next((p for p in path.parents if p.name == experiment), None)
        if exp_dir is None:
            raise ValueError(f"Missing {experiment} directory in run path: {path}")
        parts = path.relative_to(exp_dir).parts
        if len(parts) < 3:
            raise ValueError(f"Expected <experiment>/<provider>/<model>/.../runNN.json: {path}")
        content = path.read_bytes()
        sha256 = digest(content)
        if expected_hash is not None and sha256 != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {path}")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Run JSON must be an object: {path}")
        if data.get("experiment_id", experiment) != experiment:
            raise ValueError(f"Run experiment_id disagrees with selection: {path}")
        scoring_id = calculator_id(experiment, data, scheme=scheme)
        truth = data.get("agent_color")
        steps = data.get("step_records")
        if not isinstance(truth, str) or truth in {"", "unknown", "null"}:
            raise ValueError(f"Missing ground-truth agent_color: {path}")
        if not isinstance(steps, list) or any(
            not isinstance(step, dict)
            or not isinstance(step.get("identification", {}), dict)
            for step in steps
        ):
            raise ValueError(f"Malformed step_records: {path}")
        seen.add(path)
        selected.append({
            "path": str(path), "experiment_id": experiment,
            "calculator_experiment_id": scoring_id,
            "paper_experiment_id": CALCULATOR_TO_PAPER[scoring_id],
            "provider": parts[0], "model_directory": parts[1],
            "cohort": cohort, "sha256": sha256,
            "source_experiment_directory": str(exp_dir),
            "zero_steps": not steps,
        })

    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not isinstance(row.get("experiment_id"), str)):
            raise ValueError("Each manifest row needs string path and experiment_id fields")
        path = Path(row["path"])
        add(path if path.is_absolute() else data_root / path,
            row["experiment_id"], "primary", row.get("sha256"),
            row.get("experiment_id_scheme", document.get("experiment_id_scheme")))
    if additional_root is not None:
        additional_root = additional_root.resolve(strict=True)
        if not additional_root.is_dir():
            raise ValueError("--additional-root must be a dedicated run directory")
        paths = [p for p in sorted(additional_root.rglob("run*.json"))
                 if RUN_NAME.fullmatch(p.name)]
        if not paths:
            raise ValueError(f"No runNN.json files in additional cohort: {additional_root}")
        for path in paths:
            exp_dir = next((p for p in path.parents if p.name in SOURCE_EXPERIMENTS), None)
            if exp_dir is None:
                raise ValueError(f"Cannot resolve experiment for additional run: {path}")
            add(path, exp_dir.name, "additional")
    return selected


def evaluate(manifest, *, data_root, out_dir=None, additional_root=None, dry_run=False):
    manifest = manifest.resolve(strict=True)
    manifest_hash = digest(manifest.read_bytes())
    calculator_hash = digest(CALCULATOR.read_bytes())
    adapter_hash = digest(DATA_ADAPTER.read_bytes())
    selected = select_runs(manifest, data_root=data_root, additional_root=additional_root)
    summary = {
        "metric_convention": "historical; all formulas supplied by experiment.py",
        "calculator_sha256": calculator_hash,
        "data_id_adapter_sha256": adapter_hash,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "additional_root": str(additional_root.resolve()) if additional_root else None,
        "selected_runs": len(selected),
        "cohort_counts": dict(Counter(r["cohort"] for r in selected)),
        "experiment_counts": dict(sorted(Counter(r["experiment_id"] for r in selected).items())),
        "paper_experiment_counts": dict(sorted(Counter(r["paper_experiment_id"] for r in selected).items())),
        "zero_step_runs": sum(r["zero_steps"] for r in selected),
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    if out_dir is None:
        raise ValueError("--out-dir is required unless --dry-run is used")
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new directory: {out_dir}")
    input_dirs = {Path(r["source_experiment_directory"]) for r in selected}
    if additional_root is not None:
        input_dirs.add(additional_root.resolve())
    if any(out_dir.is_relative_to(path) for path in input_dirs):
        raise ValueError("Output directory must be outside the original run directories")

    with tempfile.TemporaryDirectory(prefix="self-awareness-evaluation-") as temporary:
        temporary = Path(temporary)
        view = temporary / "selected"
        for index, record in enumerate(selected):
            source = Path(record["path"])
            content = source.read_bytes()
            if digest(content) != record["sha256"]:
                raise ValueError(f"Input changed during evaluation: {source}")
            target = (view / record["calculator_experiment_id"] / record["provider"]
                      / record["model_directory"] / str(index) / source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        spec = importlib.util.spec_from_file_location("historical_calculator", CALCULATOR)
        calculator = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(calculator)
        except ModuleNotFoundError as error:
            raise ValueError("Install requirements.txt before evaluating runs") from error
        log = io.StringIO()
        result = temporary / "metrics.csv"
        with contextlib.redirect_stdout(log):
            calculator.compute_metrics(
                base_dir=str(view), experiments_to_process=sorted(EXPERIMENTS),
                csv_out=str(result),
            )
        if not result.is_file():
            raise ValueError("Calculator produced no rows; check selected model directories")
        with result.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            columns = reader.fieldnames or []
        if sum(int(r["total_runs"]) for r in rows) != len(selected):
            raise ValueError("Calculator did not account for every selected run")
        required = {f"{m}_{s}" for m in CORE_METRICS for s in ("mean", "se")}
        if not required.issubset(columns):
            raise ValueError("Calculator output is missing core metric columns")
        unchanged = all(digest(Path(r["path"]).read_bytes()) == r["sha256"] for r in selected)
        unchanged = (unchanged and digest(manifest.read_bytes()) == manifest_hash
                     and digest(CALCULATOR.read_bytes()) == calculator_hash
                     and digest(DATA_ADAPTER.read_bytes()) == adapter_hash)
        if not unchanged:
            raise ValueError("An original input or the calculator changed during evaluation")
        summary.update({
            "model_condition_pairs": len(rows), "input_integrity_verified": True,
            "undefined_values": "Empty mean is undefined, including MTATO with no successes. Legacy std/SE=0 does not define that mean.",
        })
        # Publish only after calculation and validation; never overwrite a report.
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "metrics.csv").write_bytes(result.read_bytes())
        # A public table uses E1-E5. Metric values are copied as strings unchanged.
        with (out_dir / "paper_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*columns, "calculator_experiment_id", "experiment_id_scheme"])
            writer.writeheader()
            for row in rows:
                old_id = row["experiment_id"]
                writer.writerow({**row, "experiment_id": CALCULATOR_TO_PAPER[old_id],
                                 "calculator_experiment_id": old_id, "experiment_id_scheme": "paper-v1"})
        (out_dir / "calculator.log").write_text(log.getvalue())
        (out_dir / "selected_runs.json").write_text(json.dumps(selected, indent=2) + "\n")
        (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with manifest_rows; never rewritten")
    parser.add_argument("--data-root", type=Path, default=Path.cwd(), help="Base for relative manifest paths; default: current directory")
    parser.add_argument("--additional-root", type=Path, help="Optional dedicated cohort directory; all runNN.json files under it are selected")
    parser.add_argument("--out-dir", type=Path, help="New report directory; existing directories are refused")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and print counts without writing outputs or importing pandas")
    args = parser.parse_args()
    try:
        summary = evaluate(**vars(args))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
