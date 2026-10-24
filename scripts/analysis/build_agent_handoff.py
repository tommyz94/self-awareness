#!/usr/bin/env python3
"""Build a private, structured handoff of the paper evaluation data.

The archive deliberately excludes image frames. It preserves the selected raw
run JSON files byte-for-byte, adds normalized archive paths, extracts a
transition-level episode table, audits field coverage, and bundles the relevant
logging/simulator code.

This is a private research handoff, not an anonymized conference supplement:
the raw JSON files and source manifest contain local absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PAPER_CONDITION = {
    "e1": "E1",
    "e2": "E2",
    "e3": "E3",
    "e5": "E4",
    "e7": "E5",
}

VIEWPOINT_ACTIONS = {"w", "a", "s", "d"}
UNKNOWN_COLORS = {"", "none", "null", "unknown"}

CODE_FILES = [
    "runtime/controller.py",
    "env/observation.py",
    "env/backends/mujoco.py",
    "experiments/evaluation/metrics.py",
    "experiments/tasks/cube.py",
    "experiments/tasks/self_perception.py",
    "experiments/tasks/explore.py",
    "experiments/shared/scenes.py",
    "experiments/paper/e1/experiment.py",
    "experiments/paper/e2/experiment.py",
    "experiments/paper/e3/experiment.py",
    "experiments/paper/e4/experiment.py",
    "experiments/paper/e5/experiment.py",
    "conf/experiment/e1.yaml",
    "conf/experiment/e2.yaml",
    "conf/experiment/e3.yaml",
    "conf/experiment/e5.yaml",
    "conf/experiment/e4.yaml",
]

AGGREGATE_FILES = [
    "result.csv",
    "outputs/reports/result_with_task_acc.csv",
    "notebooks/results_analysis.ipynb",
    "notebooks/paper_assets/appendix_extended_metrics_table.tex",
]


def normalized_color(value: Any) -> str:
    return str(value or "").strip().lower()


def session_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{8}_\d{6}", part):
            return part
    return "unknown_session"


def first_index(values: Iterable[bool]) -> int | None:
    for index, value in enumerate(values):
        if value:
            return index
    return None


def step_number(steps: list[dict[str, Any]], index: int | None) -> int | None:
    if index is None:
        return None
    return int(steps[index].get("step", index + 1))


def identification_color(step: dict[str, Any]) -> str:
    identification = step.get("identification") or {}
    return normalized_color(identification.get("my_color"))


def identification_confidence(step: dict[str, Any]) -> str:
    identification = step.get("identification") or {}
    return normalized_color(identification.get("confidence"))


def count_belief_changes(
    steps: list[dict[str, Any]],
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> int:
    colors = [
        identification_color(step)
        for step in steps[start_index:end_index]
        if identification_color(step) not in UNKNOWN_COLORS
    ]
    return sum(left != right for left, right in zip(colors, colors[1:]))


def first_non_unknown_index(steps: list[dict[str, Any]]) -> int | None:
    return first_index(
        identification_color(step) not in UNKNOWN_COLORS for step in steps
    )


def first_correct_index(
    steps: list[dict[str, Any]], agent_color: str
) -> int | None:
    return first_index(
        identification_color(step) == agent_color for step in steps
    )


def first_correction_index(
    steps: list[dict[str, Any]], agent_color: str
) -> int | None:
    saw_incorrect = False
    for index, step in enumerate(steps):
        color = identification_color(step)
        if color in UNKNOWN_COLORS:
            continue
        if color == agent_color and saw_incorrect:
            return index
        if color != agent_color:
            saw_incorrect = True
    return None


def first_belief_update_after(
    steps: list[dict[str, Any]], start_index: int
) -> int | None:
    if not steps:
        return None
    reference = identification_color(steps[max(0, start_index - 1)])
    for index in range(start_index, len(steps)):
        color = identification_color(steps[index])
        if color not in UNKNOWN_COLORS and color != reference:
            return index
    return None


def final_identification(run: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    task_identification = (run.get("task_output") or {}).get("identification")
    if isinstance(task_identification, dict) and task_identification:
        return task_identification
    if steps:
        identification = steps[-1].get("identification")
        if isinstance(identification, dict):
            return identification
    return {}


def task_success(
    experiment: str,
    run: dict[str, Any],
    final_color: str,
    agent_color: str,
) -> bool:
    if experiment == "e7":
        return final_color == agent_color
    selected = normalized_color(run.get("selected_cube"))
    return bool(selected) and selected == agent_color


def failure_stage(
    *,
    success: bool,
    ego_index: int | None,
    correct_index: int | None,
    final_correct: bool,
    cube_task: bool,
    selected_correct: bool,
) -> str:
    correct_after_ego = (
        ego_index is not None
        and correct_index is not None
        and correct_index >= ego_index
    )
    if success and correct_after_ego:
        return "success_after_ego_evidence"
    if success:
        return "success_without_prior_ego_evidence"
    if ego_index is None:
        return "failure_no_ego_evidence"
    if correct_index is None:
        return "failure_evidence_without_correct_attribution"
    if not final_correct:
        return "failure_correct_attribution_not_retained"
    if cube_task and not selected_correct:
        return "failure_correct_attribution_not_translated"
    return "failure_other"


def episode_row(
    *,
    family: str,
    experiment: str,
    seed: str,
    source_path: Path,
    archive_path: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    steps = list(run.get("step_records") or [])
    agent_color = normalized_color(run.get("agent_color"))
    selected_color = normalized_color(run.get("selected_cube"))
    wrong_color = normalized_color(run.get("wrong_color"))

    mirror_index = first_index(
        bool(step.get("mirror_visible")) for step in steps
    )
    ego_index = first_index(
        bool(step.get("ego_reflection_visible")) for step in steps
    )
    non_unknown_index = first_non_unknown_index(steps)
    correct_index = first_correct_index(steps, agent_color)
    correction_index = first_correction_index(steps, agent_color)

    final_ident = final_identification(run, steps)
    final_color = normalized_color(final_ident.get("my_color"))
    final_source = normalized_color(final_ident.get("source"))
    final_confidence = normalized_color(final_ident.get("confidence"))
    final_correct = final_color == agent_color
    success = task_success(experiment, run, final_color, agent_color)

    move_index = None
    reacquire_index = None
    update_index = None
    if ego_index is not None:
        move_index = next(
            (
                index
                for index in range(ego_index, len(steps))
                if normalized_color(steps[index].get("action")) in VIEWPOINT_ACTIONS
            ),
            None,
        )
    if move_index is not None:
        reacquire_index = next(
            (
                index
                for index in range(move_index + 1, len(steps))
                if bool(steps[index].get("ego_reflection_visible"))
            ),
            None,
        )
    if reacquire_index is not None:
        update_index = first_belief_update_after(steps, reacquire_index)

    pre_evidence_end = ego_index if ego_index is not None else len(steps)
    correct_start = correct_index if correct_index is not None else len(steps)

    first_attr_color = (
        identification_color(steps[non_unknown_index])
        if non_unknown_index is not None
        else ""
    )
    first_attr_confidence = (
        identification_confidence(steps[non_unknown_index])
        if non_unknown_index is not None
        else ""
    )
    confidence_at_first_correct = (
        identification_confidence(steps[correct_index])
        if correct_index is not None
        else ""
    )

    objects = list((run.get("run_setup") or {}).get("objects") or [])
    cube_objects = [obj for obj in objects if obj.get("kind") == "cube"]
    distractors = [
        obj for obj in objects if obj.get("kind") == "distractor_robot"
    ]
    candidate_colors = sorted(
        {
            normalized_color(obj.get("color"))
            for obj in cube_objects
            if normalized_color(obj.get("color"))
        }
    )
    distractor_colors = [
        normalized_color(obj.get("color")) for obj in distractors
    ]

    steps_after_correct = (
        len(steps) - correct_index - 1 if correct_index is not None else None
    )
    collisions_after_correct = (
        sum(bool(step.get("bumped_wall")) for step in steps[correct_index + 1 :])
        if correct_index is not None
        else None
    )
    moves_after_correct = (
        sum(
            normalized_color(step.get("action")) in VIEWPOINT_ACTIONS
            for step in steps[correct_index + 1 :]
        )
        if correct_index is not None
        else None
    )

    selected_correct = bool(selected_color) and selected_color == agent_color
    cube_task = experiment != "e7"
    action_agrees = (
        bool(selected_color)
        and final_color not in UNKNOWN_COLORS
        and selected_color == final_color
    )

    return {
        "paper_condition": PAPER_CONDITION[experiment],
        "code_experiment": experiment,
        "model_family": family,
        "seed": seed,
        "session_id": session_from_path(source_path),
        "run_id": run.get("run_id"),
        "archive_path": archive_path,
        "rich_step_metadata": all(
            isinstance(step.get("step_metadata"), dict) for step in steps
        ),
        "num_steps": len(steps),
        "agent_color": agent_color,
        "wrong_prompt_color": wrong_color,
        "candidate_cube_colors": "|".join(candidate_colors),
        "num_distractors": len(distractors),
        "distractor_colors": "|".join(distractor_colors),
        "same_color_distractor": agent_color in distractor_colors,
        "selected_color": selected_color,
        "final_attribution_color": final_color,
        "final_attribution_source": final_source,
        "final_confidence": final_confidence,
        "task_success": success,
        "completed": bool(run.get("completed")),
        "first_mirror_step": step_number(steps, mirror_index),
        "first_ego_reflection_step": step_number(steps, ego_index),
        "mirror_visible_steps": sum(
            bool(step.get("mirror_visible")) for step in steps
        ),
        "ego_reflection_visible_steps": sum(
            bool(step.get("ego_reflection_visible")) for step in steps
        ),
        "first_non_unknown_attribution_step": step_number(
            steps, non_unknown_index
        ),
        "first_non_unknown_attribution_color": first_attr_color,
        "first_non_unknown_confidence": first_attr_confidence,
        "first_correct_attribution_step": step_number(steps, correct_index),
        "first_correction_step": step_number(steps, correction_index),
        "confidence_at_first_correct": confidence_at_first_correct,
        "belief_latency_from_ego": (
            step_number(steps, correct_index) - step_number(steps, ego_index)
            if correct_index is not None and ego_index is not None
            else None
        ),
        "correct_attribution_ever": correct_index is not None,
        "correct_attribution_after_ego": (
            correct_index is not None
            and ego_index is not None
            and correct_index >= ego_index
        ),
        "correct_attribution_retained": (
            correct_index is not None and final_correct
        ),
        "final_attribution_correct": final_correct,
        "action_agrees_with_final_attribution": action_agrees,
        "correct_attribution_translated_to_action": (
            final_correct and selected_correct if cube_task else None
        ),
        "belief_changes_total": count_belief_changes(steps),
        "belief_changes_before_ego": count_belief_changes(
            steps, end_index=pre_evidence_end
        ),
        "belief_changes_after_ego": (
            count_belief_changes(steps, start_index=ego_index)
            if ego_index is not None
            else None
        ),
        "belief_changes_after_first_correct": (
            count_belief_changes(steps, start_index=correct_start)
            if correct_index is not None
            else None
        ),
        "unsupported_claim_before_ego": (
            non_unknown_index is not None
            and (ego_index is None or non_unknown_index < ego_index)
        ),
        "unsupported_initial_claim_persisted": (
            non_unknown_index is not None
            and (ego_index is None or non_unknown_index < ego_index)
            and first_attr_color == final_color
        ),
        "post_ego_viewpoint_change": move_index is not None,
        "ego_reflection_reacquired_after_change": reacquire_index is not None,
        "belief_updated_after_reacquisition": update_index is not None,
        "active_reflection_verification_proxy": (
            ego_index is not None
            and move_index is not None
            and reacquire_index is not None
            and update_index is not None
        ),
        "steps_after_first_correct": steps_after_correct,
        "moves_after_first_correct": moves_after_correct,
        "collisions_after_first_correct": collisions_after_correct,
        "failure_stage": failure_stage(
            success=success,
            ego_index=ego_index,
            correct_index=correct_index,
            final_correct=final_correct,
            cube_task=cube_task,
            selected_correct=selected_correct,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_coverage(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_keys: Counter[str] = Counter()
    step_keys: Counter[str] = Counter()
    nested_keys: Counter[str] = Counter()
    per_condition: defaultdict[str, Counter[str]] = defaultdict(Counter)
    total_steps = 0
    frame_paths = 0
    existing_frame_paths = 0
    sibling_frame_files = 0
    sibling_frame_runs = 0

    for record in run_records:
        experiment = record["experiment"]
        path: Path = record["source_path"]
        run = record["run"]
        steps = list(run.get("step_records") or [])
        total_steps += len(steps)
        per_condition[experiment]["runs"] += 1
        per_condition[experiment]["steps"] += len(steps)
        run_keys.update(run.keys())

        observation_dir = path.parent / "obs"
        images = list(observation_dir.glob("*.jpg")) if observation_dir.exists() else []
        if images:
            sibling_frame_runs += 1
            sibling_frame_files += len(images)

        for step in steps:
            step_keys.update(step.keys())
            metadata = step.get("step_metadata") or {}
            nested_keys.update(f"step_metadata.{key}" for key in metadata)
            llm_io = metadata.get("llm_io") or {}
            nested_keys.update(f"step_metadata.llm_io.{key}" for key in llm_io)
            identification = step.get("identification") or {}
            nested_keys.update(f"identification.{key}" for key in identification)
            image_path = llm_io.get("image_path") or metadata.get("image_path")
            if image_path:
                frame_paths += 1
                existing_frame_paths += Path(image_path).exists()

    return {
        "selected_runs": len(run_records),
        "total_steps": total_steps,
        "per_condition": {
            experiment: dict(counts)
            for experiment, counts in sorted(per_condition.items())
        },
        "run_key_coverage": dict(run_keys),
        "step_key_coverage": dict(step_keys),
        "nested_step_key_coverage": dict(nested_keys),
        "steps_with_explicit_image_path": frame_paths,
        "explicit_image_paths_existing": existing_frame_paths,
        "runs_with_sibling_observation_frames": sibling_frame_runs,
        "sibling_observation_frame_files": sibling_frame_files,
    }


def verified_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    e2 = [row for row in rows if row["code_experiment"] == "e2"]
    e3 = [row for row in rows if row["code_experiment"] == "e3"]
    e4 = [row for row in rows if row["code_experiment"] == "e5"]
    e5 = [row for row in rows if row["code_experiment"] == "e7"]

    def outcome_counts(group: list[dict[str, Any]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in group:
            selected = row["selected_color"]
            if not selected:
                category = "no_selection"
            elif selected == row["agent_color"]:
                category = "correct"
            elif selected == row["wrong_prompt_color"]:
                category = "false_prompt_color"
            elif selected in row["candidate_cube_colors"].split("|"):
                category = "other_present_color"
            else:
                category = "other_absent_color"
            counts[category] += 1
        return dict(counts)

    e2_blue = [row for row in e2 if row["agent_color"] == "blue"]
    e2_nonblue = [row for row in e2 if row["agent_color"] != "blue"]

    def same_color_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for present in (False, True):
            subset = [
                row for row in group if row["same_color_distractor"] is present
            ]
            result[str(present).lower()] = {
                "successes": sum(bool(row["task_success"]) for row in subset),
                "runs": len(subset),
            }
        return result

    cue_present = []
    cue_absent = []
    for row in e3:
        target = (
            cue_present
            if row["wrong_prompt_color"]
            in row["candidate_cube_colors"].split("|")
            else cue_absent
        )
        target.append(row)

    claude_e3 = [
        row for row in e3 if row["model_family"] == "claude-sonnet-4.6"
    ]

    return {
        "E2_blue_shortcut": {
            "blue_target": {
                "successes": sum(bool(row["task_success"]) for row in e2_blue),
                "runs": len(e2_blue),
            },
            "nonblue_target": {
                "successes": sum(
                    bool(row["task_success"]) for row in e2_nonblue
                ),
                "runs": len(e2_nonblue),
            },
            "blue_targets_by_model": dict(
                Counter(
                    row["model_family"]
                    for row in e2
                    if row["agent_color"] == "blue"
                )
            ),
        },
        "E3_outcomes": {
            "all_models": outcome_counts(e3),
            "claude": outcome_counts(claude_e3),
            "cue_present": {
                "cue_selections": sum(
                    row["selected_color"] == row["wrong_prompt_color"]
                    for row in cue_present
                ),
                "runs": len(cue_present),
            },
            "cue_absent": {
                "cue_selections": sum(
                    row["selected_color"] == row["wrong_prompt_color"]
                    for row in cue_absent
                ),
                "runs": len(cue_absent),
            },
        },
        "same_color_distractor": {
            "paper_E4": same_color_summary(e4),
            "paper_E5": same_color_summary(e5),
        },
        "failure_stage_counts": {
            condition: dict(
                Counter(
                    row["failure_stage"]
                    for row in rows
                    if row["paper_condition"] == condition
                )
            )
            for condition in PAPER_CONDITION.values()
        },
    }


def archive_path_for(
    family: str,
    experiment: str,
    seed: str,
    source_path: Path,
) -> str:
    return "/".join(
        [
            "runs",
            PAPER_CONDITION[experiment],
            family,
            str(seed),
            session_from_path(source_path),
            source_path.name,
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_readme(coverage: dict[str, Any]) -> str:
    return f"""# Self-awareness benchmark: private agent handoff

This archive contains the complete structured evaluation set selected by
`metric_manifest.json`. Image frames are deliberately excluded.

## Inventory

- Runs: {coverage['selected_runs']}
- Timesteps: {coverage['total_steps']}
- Sibling observation frames available in the source workspace:
  {coverage['sibling_observation_frame_files']} across
  {coverage['runs_with_sibling_observation_frames']} runs
- Steps with rich movement/prompt/raw-reply metadata:
  {coverage['nested_step_key_coverage'].get('step_metadata.llm_io.raw_reply', 0)}

## Condition mapping

- paper E1 = code e1
- paper E2 = code e2
- paper E3 = code e3
- paper E4 = code e5
- paper E5 = code e7

Do not mix paper E4 with the separate later `reports/e4_metrics_report.json`.

## Archive layout

- `manifests/metric_manifest.json`: original authoritative manifest
- `manifests/archive_index.csv`: source-to-archive path mapping
- `runs/`: all 840 raw selected run JSON files
- `derived/episode_transitions.csv`: one row per run with temporal transitions
- `derived/field_coverage.json`: exact field coverage audit
- `derived/verified_counts.json`: directly reproducible shortcut/failure counts
- `derived/sample_manifest.csv`: representative rich/legacy and contrast runs
- `code/`: relevant simulator, logger, task, metric, and experiment code
- `aggregate/`: existing aggregate results and analysis notebook
- `scripts/build_agent_handoff.py`: the builder/extractor used here

## Important limitations

The logs support behavioral active-verification proxies based on:

1. ego reflection visibility;
2. viewpoint-changing actions;
3. reflection retention/reacquisition;
4. subsequent belief updates.

They do not contain a reflected-ego bounding box, continuous reflected pose, or
per-step state for every moving distractor. Therefore, the data cannot directly
prove that the model used action-reflection synchrony. Stronger synchrony claims
require frame tracking, manual annotation, deterministic replay, or new logging.

## Privacy/anonymity warning

This is a private research handoff. Raw JSON and the original manifest contain
local absolute paths and have not been anonymized for conference submission.
"""


def select_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add(label: str, candidates: list[dict[str, Any]], count: int = 1) -> None:
        for row in candidates[:count]:
            selected.append(
                {
                    "sample_label": label,
                    "paper_condition": row["paper_condition"],
                    "model_family": row["model_family"],
                    "task_success": row["task_success"],
                    "rich_step_metadata": row["rich_step_metadata"],
                    "archive_path": row["archive_path"],
                }
            )

    add("rich_format", [row for row in rows if row["rich_step_metadata"]])
    add("legacy_format", [row for row in rows if not row["rich_step_metadata"]])

    for model in ("claude-sonnet-4.6", "ministral-14b"):
        base = [
            row
            for row in rows
            if row["paper_condition"] == "E4" and row["model_family"] == model
        ]
        add(f"E4_{model}_success", [row for row in base if row["task_success"]], 3)
        add(f"E4_{model}_failure", [row for row in base if not row["task_success"]], 3)

    for condition in ("E1", "E2"):
        for model in ("claude-sonnet-4.6", "ministral-14b"):
            base = [
                row
                for row in rows
                if row["paper_condition"] == condition
                and row["model_family"] == model
            ]
            add(
                f"{condition}_{model}_success",
                [row for row in base if row["task_success"]],
            )
            add(
                f"{condition}_{model}_failure",
                [row for row in base if not row["task_success"]],
            )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/agent_handoff"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (repo_root / args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = repo_root / "final_outputs/reports/metric_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    run_records: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for model in manifest["models"]:
        family = model["family"]
        for experiment_record in model["experiments"]:
            experiment = experiment_record["experiment"]
            if experiment not in PAPER_CONDITION:
                continue
            for seed_record in experiment_record["selected_seeds"]:
                seed = str(seed_record["seed"])
                for run_record in seed_record["kept_runs_manifest"]:
                    source_path = Path(run_record["path"])
                    if not source_path.exists():
                        raise FileNotFoundError(source_path)
                    archive_path = archive_path_for(
                        family, experiment, seed, source_path
                    )
                    run = json.loads(source_path.read_text(encoding="utf-8"))
                    record = {
                        "family": family,
                        "experiment": experiment,
                        "seed": seed,
                        "source_path": source_path,
                        "archive_path": archive_path,
                        "run": run,
                    }
                    run_records.append(record)
                    archive_rows.append(
                        {
                            "paper_condition": PAPER_CONDITION[experiment],
                            "code_experiment": experiment,
                            "model_family": family,
                            "seed": seed,
                            "source_path": str(source_path),
                            "archive_path": archive_path,
                            "source_sha256": sha256(source_path),
                        }
                    )
                    episode_rows.append(
                        episode_row(
                            family=family,
                            experiment=experiment,
                            seed=seed,
                            source_path=source_path,
                            archive_path=archive_path,
                            run=run,
                        )
                    )

    if len(run_records) != 840:
        raise ValueError(f"Expected 840 selected runs, found {len(run_records)}")

    coverage = audit_coverage(run_records)
    counts = verified_counts(episode_rows)
    samples = select_samples(episode_rows)

    archive_index_path = output_dir / "archive_index.csv"
    transition_path = output_dir / "episode_transitions.csv"
    sample_path = output_dir / "sample_manifest.csv"
    coverage_path = output_dir / "field_coverage.json"
    counts_path = output_dir / "verified_counts.json"
    readme_path = output_dir / "README.md"

    write_csv(archive_index_path, archive_rows)
    write_csv(transition_path, episode_rows)
    write_csv(sample_path, samples)
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    counts_path.write_text(
        json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    readme_path.write_text(build_readme(coverage), encoding="utf-8")

    zip_path = output_dir / "self_awareness_agent_handoff.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(
            manifest_path, "manifests/metric_manifest.json"
        )
        archive.write(archive_index_path, "manifests/archive_index.csv")
        archive.write(transition_path, "derived/episode_transitions.csv")
        archive.write(coverage_path, "derived/field_coverage.json")
        archive.write(counts_path, "derived/verified_counts.json")
        archive.write(sample_path, "derived/sample_manifest.csv")
        archive.write(readme_path, "README.md")
        archive.write(
            Path(__file__).resolve(), "scripts/build_agent_handoff.py"
        )

        for record in run_records:
            archive.write(record["source_path"], record["archive_path"])

        for relative in CODE_FILES:
            source = repo_root / relative
            if source.exists():
                archive.write(source, f"code/{relative}")

        for relative in AGGREGATE_FILES:
            source = repo_root / relative
            if source.exists():
                archive.write(source, f"aggregate/{relative}")

    checksum = sha256(zip_path)
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{checksum}  {zip_path.name}\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "archive": str(zip_path),
                "sha256": checksum,
                "archive_bytes": zip_path.stat().st_size,
                "runs": len(run_records),
                "transitions": len(episode_rows),
                "samples": len(samples),
                "coverage": coverage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
