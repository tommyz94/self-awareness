#!/usr/bin/env python3
"""Recompute six metrics from selected raw runs, then optionally check a table.

The scoring functions never receive reference values, model names or cohorts.
Profiles are global, source-backed definitions, not per-cell corrections.
This is a comparison tool. Use evaluate_selected_runs.py for the selected
historical evaluation; experiment.py remains the unchanged metric source.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict

METRICS = ("TSA", "TTD", "MCR", "MTATO", "CAAL", "CONFABULATION")
EXPERIMENTS = {"e1", "e2", "e3", "e5", "e7"}
PROFILES = {
    "historical": "experiment.py: done required for success; MCR/CAAL before done or whole timeout; CR on whole trajectory; legacy zero SE for n<2.",
    "paper": "PDF sections 3.3/3.5: endpoint first done or last step; all process events strictly before endpoint; undefined MTATO excluded. Singleton SE zero follows the existing reporting convention, with se_estimable=false.",
}
# Identity aliases only: these never enter metric scoring.
ALIASES = {
    "google_gemma-4-26b-a4b-it_nitro": "gemma4_26b",
    "google_gemma-4-31b-it_nitro": "gemma4_31b",
    "mistralai_ministral-14b-2512_nitro": "ministral-3_14b",
}


def label(value):
    if value is None or str(value).strip().lower() in {"", "unknown", "null"}:
        return None
    return str(value).strip()


def sees_mirror(step):
    return bool(step.get("mirror_visible", False) or step.get("ego_reflection_visible", False))


def episode_metrics(data, *, task, profile):
    """Pure scoring: task is 'cube' or 'attribution'; profile applies to all runs."""
    if task not in {"cube", "attribution"} or profile not in PROFILES:
        raise ValueError("Unknown task or profile")
    steps = data.get("step_records", [])
    if not isinstance(steps, list):
        raise ValueError("step_records must be a list")
    done = next((i for i, s in enumerate(steps) if s.get("action") == "done"), None)
    endpoint_index = done if done is not None else len(steps) - 1
    endpoint = steps[endpoint_index] if steps else {}
    truth = label(data.get("agent_color"))
    final_ident = label(endpoint.get("identification", {}).get("my_color"))
    final_cube = label(endpoint.get("selected_cube"))
    prediction = final_cube if task == "cube" else final_ident
    success = int(truth is not None and prediction == truth)
    if profile == "historical" and done is None:
        success = 0
    if profile == "historical":
        events = steps[:done] if done is not None else steps
        claim_events = steps
    else:
        events = steps[:max(endpoint_index, 0)]
        claim_events = events
    mcr = int(any(sees_mirror(s) for s in events))
    caal = int(truth is not None and any(
        label(s.get("identification", {}).get("my_color")) == truth for s in events))
    first_mirror = next((i for i, s in enumerate(claim_events) if sees_mirror(s)), None)
    first_claim = next((i for i, s in enumerate(claim_events)
                        if label(s.get("identification", {}).get("my_color")) is not None), None)
    cr = int(first_claim is not None and (first_mirror is None or first_claim < first_mirror))
    scores = {"TSA": success, "TTD": done + 1 if done is not None else len(steps),
              "MCR": mcr, "MTATO": mcr if success else None, "CAAL": caal,
              "CONFABULATION": cr}
    return {"scores": scores, "done": done is not None, "zero_steps": not steps,
            "truth": truth, "final_ident": final_ident, "final_cube": final_cube}


def summarize(values, *, profile):
    valid = [float(v) for v in values if v is not None]
    n = len(valid)
    if not n:
        return {"mean": None, "std": 0.0 if profile == "historical" else None,
                "se": 0.0 if profile == "historical" else None, "n": 0,
                "se_estimable": False}
    sd = statistics.stdev(valid) if n > 1 else 0.0
    return {"mean": statistics.mean(valid), "std": sd, "se": sd / math.sqrt(n),
            "n": n, "se_estimable": n > 1}


def aggregate(episodes, *, profile):
    groups = defaultdict(list)
    for record in episodes:
        groups[record["experiment_id"], record["model_name"]].append(record)
    result = []
    for (experiment, model), rows in sorted(groups.items()):
        out = {"experiment_id": experiment, "model_name": model, "metric_profile": profile,
               "total_runs": len(rows),
               "unique_seeds": len({r["seed"] for r in rows if r["seed"] is not None}),
               "completed_runs": sum(r["done"] for r in rows),
               "zero_step_runs": sum(r["zero_steps"] for r in rows)}
        for metric in METRICS:
            stats = summarize([r["scores"][metric] for r in rows], profile=profile)
            out.update({f"{metric}_{k}": v for k, v in stats.items()})
        completed = [r["scores"]["TSA"] for r in rows if r["done"]]
        out.update({f"TSA-COMPLETE_{k}": v for k, v in summarize(completed, profile=profile).items()})
        matrix = defaultdict(Counter)
        for r in rows:
            matrix[r["truth"] or "unknown_agent"][r["final_ident"] or "unknown_identification"] += 1
        out["ident_confusion_counts_json"] = json.dumps(dict(matrix), sort_keys=True)
        result.append(out)
    return result


def display(value, *, rounding):
    """Presentation only. Undefined values remain undefined; never filled to fit."""
    if value is None:
        return "--"
    number = Decimal(str(value))
    if rounding == "three_then_two":
        number = number.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    elif rounding != "direct":
        raise ValueError("Unknown rounding policy")
    return str(number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compare_reference(computed, reference, *, rounding):
    """Compare after calculation; reference never participates in episode scoring."""
    lookup = {(r["experiment_id"], r["model_name"]): r for r in computed}
    if len(lookup) != len(computed):
        raise ValueError("Duplicate aggregate identities")
    comparisons = []
    for ref in reference:
        row = lookup.get((ref["experiment_id"], ref["model_name"]))
        for metric in METRICS:
            for stat in ("mean", "se"):
                field = f"{metric}_{stat}"
                value = row[field] if row else None
                expected = display(ref[field], rounding="direct")
                shown = display(value, rounding=rounding) if row else "missing row"
                comparisons.append({"experiment_id": ref["experiment_id"], "model_name": ref["model_name"],
                                    "field": field, "computed": value, "computed_display": shown,
                                    "reference": ref[field], "reference_display": expected,
                                    "match": shown == expected})
    return comparisons


def selected_inputs(manifest_path, additional_root, data_root):
    manifest = json.loads(manifest_path.read_text())
    inputs = []
    seen = set()
    def add(path, experiment, model, seed, cohort):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unsupported experiment: {experiment}")
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Duplicate selected run: {path}")
        seen.add(path)
        raw = path.read_bytes()
        data = json.loads(raw)
        inputs.append({"path": str(path), "experiment_id": experiment,
                       "model_name": ALIASES.get(model, model), "seed": seed if seed is not None else data.get("run_setup", {}).get("seed"),
                       "cohort": cohort, "sha256": hashlib.sha256(raw).hexdigest(), "data": data})
    for row in manifest["manifest_rows"]:
        path = Path(row["path"])
        add(path if path.is_absolute() else data_root / path,
            row["experiment_id"], row["model"], row.get("seed"), "manifest")
    if additional_root:
        for path in sorted(additional_root.rglob("run*.json")):
            if not re.fullmatch(r"run\d+\.json", path.name):
                continue
            exp_dir = next((p for p in path.parents if p.name in EXPERIMENTS), None)
            if exp_dir is None:
                raise ValueError(f"Cannot resolve implementation experiment ID: {path}")
            parts = path.relative_to(exp_dir).parts
            if len(parts) < 3:
                raise ValueError(f"Cannot resolve model from run path: {path}")
            add(path, exp_dir.name, parts[1], None, "additional")
    if not inputs:
        raise ValueError("No selected runs")
    return inputs


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    inputs = selected_inputs(args.manifest, args.additional_root, args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    public_inputs = [{k: v for k, v in r.items() if k != "data"} for r in inputs]
    (args.out_dir / "selected_inputs.json").write_text(json.dumps(public_inputs, indent=2) + "\n")
    summary = {"episodes": len(inputs), "profiles": {},
               "cohort_counts": dict(Counter(r["cohort"] for r in inputs)),
               "reference_role": "Post-calculation verification only; never passed to scoring.",
               "singleton_se_policy": "Legacy reporting zero; se_estimable=false for n<2.",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    rows_by_profile = {}
    for profile in args.profiles:
        folder = args.out_dir / profile
        folder.mkdir(exist_ok=True)
        scored = []
        for record in inputs:
            task = "attribution" if record["experiment_id"] == "e7" else "cube"
            score = episode_metrics(record["data"], task=task, profile=profile)
            scored.append({**{k: v for k, v in record.items() if k != "data"}, **score})
        rows = aggregate(scored, profile=profile)
        write_csv(folder / "aggregates.csv", rows)
        (folder / "episodes.json").write_text(json.dumps(scored, indent=2) + "\n")
        details = {"definition": PROFILES[profile], "model_condition_pairs": len(rows),
                   "zero_step_runs": sum(r["zero_steps"] for r in scored)}
        rows_by_profile[profile] = rows
        summary["profiles"][profile] = details
    # Read reference scores only after every run/profile has been computed.
    if args.reference_cells:
        reference_bytes = args.reference_cells.read_bytes()
        reference = json.loads(reference_bytes)
        summary["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        for profile, rows in rows_by_profile.items():
            folder = args.out_dir / profile
            details = summary["profiles"][profile]
            details["comparisons"] = {}
            for rounding in ("direct", "three_then_two"):
                comparison = compare_reference(rows, reference, rounding=rounding)
                write_csv(folder / f"comparison_{rounding}.csv", comparison)
                details["comparisons"][rounding] = {
                    "matching_fields": sum(r["match"] for r in comparison),
                    "total_fields": len(comparison)}
    if not all(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in inputs):
        raise RuntimeError("An input changed during calculation")
    summary["raw_integrity_verified"] = True
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["historical"],
                        help="Default: historical. Select historical paper explicitly to compare manuscript formulas.")
    parser.add_argument("--reference-cells", type=Path, help="Optional verification JSON, read only after scoring")
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
