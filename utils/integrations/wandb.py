from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Sequence

if TYPE_CHECKING:
    from experiments.evaluation.judge import JudgeResult


_CANONICAL_METRIC_CODES = ("M1", "M2", "M3", "M4")
_JUDGE_SCORE_COLUMNS = ["metric_code", "metric_name", "score", "score_norm", "justification"]
_COMPARISON_COLUMNS = [
    "run_stem",
    "agent_color",
    "selected_cube",
    "wrong_color",
    "correct",
    "completed",
    "steps",
    "total_score",
    "mean_score",
    "M1",
    "M2",
    "M3",
    "M4",
    "judge_model",
    "prompt_variant",
    "include_inner_reasoning",
]


@dataclass(frozen=True)
class RunContext:
    env: str
    nav_client: str
    evaluated_model: str
    experiment_type: str
    run_stem: str
    group: str
    name: str


def _derive_run_context(run_path: str | Path, run_name: str | None = None) -> RunContext:
    path = Path(run_path)
    parts = path.parts

    # Current layout: <root>/<env>/<experiment>/<client>/<model>/level_N/<timestamp>/run_NN/runNN.json.
    if len(parts) >= 8 and re.fullmatch(r"level_\d+", parts[-4]) and re.fullmatch(r"run_\d+", parts[-2]):
        env, experiment_type, nav_client, evaluated_model = parts[-8:-4]
    else:
        try:
            runs_idx = parts.index("runs")
        except ValueError as exc:
            raise ValueError(
                f"Could not derive W&B context from run path {path!s}: missing 'runs' segment."
            ) from exc

        outputs_idx = None
        for idx, part in enumerate(parts[:runs_idx]):
            if part == "outputs":
                outputs_idx = idx

        if outputs_idx is not None:
            between = parts[outputs_idx + 1 : runs_idx]
        else:
            between = ()

        tail = parts[runs_idx + 1 :]
        experiment_match = None
        if len(between) >= 4:
            candidate = between[-3]
            if re.fullmatch(r"e\d+", candidate):
                experiment_match = candidate

        if experiment_match is not None:
            env, experiment_type, nav_client, evaluated_model = between[-4:]
        elif len(between) >= 3:
            env, nav_client, evaluated_model = between[-3:]
        elif len(between) == 2:
            env, evaluated_model = between
            nav_client = "legacy"
        else:
            raise ValueError(f"Could not derive env/client/model from run path {path!s}.")

        if experiment_match is None and len(tail) >= 2 and not tail[0].endswith(".json"):
            experiment_type = tail[0]
        elif experiment_match is None:
            experiment_type = "single-run"
    run_stem = path.stem
    group = f"{evaluated_model}/{experiment_type}"
    name = run_name or f"{group}/{run_stem}"

    return RunContext(
        env=env,
        nav_client=nav_client,
        evaluated_model=evaluated_model,
        experiment_type=experiment_type,
        run_stem=run_stem,
        group=group,
        name=name,
    )


def _result_rows(results: Sequence["JudgeResult"]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "metric_code": result.metric_code,
                "metric_name": result.metric_name,
                "score": result.score,
                "score_norm": result.score / 5.0,
                "justification": result.justification,
            }
        )
    return rows


def _build_export_from_inputs(
    results: Sequence["JudgeResult"],
    run_data: Dict[str, Any],
    *,
    judge_model: str,
    prompt_variant: str,
    include_inner_reasoning: bool,
    run_path: str | Path,
) -> Dict[str, Any]:
    context = _derive_run_context(run_path)
    metric_codes = [result.metric_code for result in results]
    total_score = sum(result.score for result in results)
    mean_score = (total_score / len(results)) if results else 0.0

    agent_color = run_data.get("agent_color")
    selected_cube = run_data.get("selected_cube")
    correct = bool(selected_cube is not None and selected_cube == agent_color)

    return {
        "run_path": str(Path(run_path)),
        "judge_model": judge_model,
        "prompt_variant": prompt_variant,
        "include_inner_reasoning": include_inner_reasoning,
        "env": context.env,
        "nav_client": context.nav_client,
        "evaluated_model": context.evaluated_model,
        "experiment_type": context.experiment_type,
        "run_stem": context.run_stem,
        "agent_color": agent_color,
        "selected_cube": selected_cube,
        "wrong_color": run_data.get("wrong_color"),
        "completed": run_data.get("completed"),
        "steps": run_data.get("steps"),
        "correct": correct,
        "metric_codes": metric_codes,
        "total_score": total_score,
        "mean_score": mean_score,
        "results": [
            {
                "metric_code": result.metric_code,
                "metric_name": result.metric_name,
                "score": result.score,
                "justification": result.justification,
            }
            for result in results
        ],
    }


def _is_v2_judge_export(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("results"), list):
        return False

    required_keys = {
        "run_path",
        "judge_model",
        "prompt_variant",
        "include_inner_reasoning",
        "env",
        "nav_client",
        "evaluated_model",
        "experiment_type",
        "run_stem",
        "agent_color",
        "selected_cube",
        "wrong_color",
        "completed",
        "steps",
        "correct",
        "metric_codes",
        "total_score",
        "mean_score",
        "results",
    }
    if not required_keys.issubset(payload):
        return False

    for result in payload["results"]:
        if not isinstance(result, dict):
            return False
        if not {"metric_code", "metric_name", "score", "justification"}.issubset(result):
            return False
    return True


def _coerce_export(
    results: Sequence["JudgeResult"],
    run_data: Dict[str, Any],
    *,
    judge_model: str,
    prompt_variant: str,
    include_inner_reasoning: bool,
    run_path: str | Path,
    judge_export: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if judge_export is None:
        return _build_export_from_inputs(
            results,
            run_data,
            judge_model=judge_model,
            prompt_variant=prompt_variant,
            include_inner_reasoning=include_inner_reasoning,
            run_path=run_path,
        )
    if not _is_v2_judge_export(judge_export):
        raise ValueError("judge_export must be a v2 judge export object.")
    return dict(judge_export)


def _export_result_rows(export: Dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in export["results"]:
        rows.append(
            {
                "metric_code": result["metric_code"],
                "metric_name": result["metric_name"],
                "score": result["score"],
                "score_norm": result["score"] / 5.0,
                "justification": result["justification"],
            }
        )
    return rows


def _metric_score_map(export: Dict[str, Any]) -> Dict[str, int]:
    return {
        result["metric_code"]: int(result["score"])
        for result in export["results"]
    }


def _build_radar_image(export: Dict[str, Any]):
    from PIL import Image, ImageDraw, ImageFont

    scores = _metric_score_map(export)
    if not all(code in scores for code in _CANONICAL_METRIC_CODES):
        return None

    image = Image.new("RGBA", (480, 480), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    center_x = 240
    center_y = 240
    radius = 150
    axis_count = len(_CANONICAL_METRIC_CODES)
    angles = [
        (-math.pi / 2.0) + ((2.0 * math.pi * idx) / axis_count)
        for idx in range(axis_count)
    ]

    def _point(scale: float, angle: float) -> tuple[float, float]:
        return (
            center_x + math.cos(angle) * radius * scale,
            center_y + math.sin(angle) * radius * scale,
        )

    for ring in (0.2, 0.4, 0.6, 0.8, 1.0):
        ring_points = [_point(ring, angle) for angle in angles]
        draw.polygon(ring_points, outline="#d0d7de")

    for code, angle in zip(_CANONICAL_METRIC_CODES, angles):
        draw.line((center_x, center_y, *_point(1.0, angle)), fill="#8c959f", width=1)
        label_x, label_y = _point(1.16, angle)
        draw.text((label_x - 10, label_y - 8), code, fill="#24292f", font=font)

    polygon_points = [_point(scores[code] / 5.0, angle) for code, angle in zip(_CANONICAL_METRIC_CODES, angles)]
    draw.polygon(polygon_points, fill="#1f6feb33", outline="#1f6feb", width=3)
    draw.text((145, 20), "Judge scores radar", fill="#24292f", font=font)

    return image


def _load_saved_export(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not _is_v2_judge_export(payload):
        return None
    return payload


def _matching_config(candidate: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    return (
        candidate.get("judge_model") == reference.get("judge_model")
        and candidate.get("prompt_variant") == reference.get("prompt_variant")
        and candidate.get("include_inner_reasoning") == reference.get("include_inner_reasoning")
        and candidate.get("metric_codes") == reference.get("metric_codes")
    )


def _collect_experiment_exports(
    current_export: Dict[str, Any],
    run_path: str | Path,
) -> list[Dict[str, Any]]:
    run_dir = Path(run_path).parent
    exports_by_run = {str(current_export["run_stem"]): current_export}

    for candidate_path in sorted(run_dir.glob("run*_judge.json")):
        candidate = _load_saved_export(candidate_path)
        if candidate is None or not _matching_config(candidate, current_export):
            continue
        exports_by_run.setdefault(str(candidate["run_stem"]), candidate)

    return [exports_by_run[key] for key in sorted(exports_by_run)]


def _build_experiment_comparison_table(wandb: Any, exports: Sequence[Dict[str, Any]]):
    table = wandb.Table(columns=_COMPARISON_COLUMNS)
    for export in exports:
        metric_scores = _metric_score_map(export)
        table.add_data(
            export.get("run_stem"),
            export.get("agent_color"),
            export.get("selected_cube"),
            export.get("wrong_color"),
            export.get("correct"),
            export.get("completed"),
            export.get("steps"),
            export.get("total_score"),
            export.get("mean_score"),
            metric_scores.get("M1"),
            metric_scores.get("M2"),
            metric_scores.get("M3"),
            metric_scores.get("M4"),
            export.get("judge_model"),
            export.get("prompt_variant"),
            export.get("include_inner_reasoning"),
        )
    return table


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def _format_judge_output_markdown(export: Dict[str, Any]) -> str:
    lines = [
        "# Judge Output",
        "",
        "## Run",
        f"- Evaluated model: `{export['evaluated_model']}`",
        f"- Experiment type: `{export['experiment_type']}`",
        f"- Run stem: `{export['run_stem']}`",
        "",
        "## Judge Config",
        f"- Judge model: `{export['judge_model']}`",
        f"- Prompt variant: `{export['prompt_variant']}`",
        f"- Include inner reasoning: `{export['include_inner_reasoning']}`",
        "",
        "## Aggregate Scores",
        f"- Total score: `{export['total_score']}`",
        f"- Mean score: `{export['mean_score']}`",
        "",
        "## Metric Outputs",
    ]

    for result in export["results"]:
        lines.extend(
            [
                "",
                f"### {result['metric_code']} — {result['metric_name']}",
                f"- Score: `{result['score']}/5`",
                f"- Justification: {result['justification']}",
            ]
        )

    return "\n".join(lines) + "\n"


def _build_judge_output_artifact(wandb: Any, export: Dict[str, Any], artifact_dir: Path):
    artifact_name = (
        f"judge-output-{_slug(export['evaluated_model'])}"
        f"-{_slug(export['experiment_type'])}"
        f"-{_slug(export['run_stem'])}"
    )
    artifact = wandb.Artifact(artifact_name, type="judge-output")
    content = _format_judge_output_markdown(export)
    artifact_path = artifact_dir / "judge_output.md"
    artifact_path.write_text(content)
    artifact.add_file(str(artifact_path), name="judge_output.md")

    return artifact


def log_judge_results(
    results: Sequence["JudgeResult"],
    run_data: Dict[str, Any],
    *,
    judge_model: str,
    prompt_variant: str,
    include_inner_reasoning: bool,
    run_path: str | Path,
    project: str = "self-aware",
    run_name: str | None = None,
    judge_export: Dict[str, Any] | None = None,
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested, but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    export = _coerce_export(
        results,
        run_data,
        judge_model=judge_model,
        prompt_variant=prompt_variant,
        include_inner_reasoning=include_inner_reasoning,
        run_path=run_path,
        judge_export=judge_export,
    )
    context = _derive_run_context(run_path, run_name=run_name)
    metric_codes = list(export["metric_codes"])
    metric_count = len(metric_codes)
    total_score = export["total_score"]
    mean_score = export["mean_score"]

    # Prefer experiment_id / task_id / scene_id written by the simulator when available;
    # fall back to the path-derived experiment_type for legacy run JSONs.
    experiment_id = run_data.get("experiment_id") or export.get("experiment_type") or context.experiment_type
    task_id = run_data.get("task_id")
    scene_id = run_data.get("scene_id")

    config = {
        "judge_model": export["judge_model"],
        "prompt_variant": export["prompt_variant"],
        "include_inner_reasoning": export["include_inner_reasoning"],
        "env": export["env"],
        "nav_client": export["nav_client"],
        "evaluated_model": export["evaluated_model"],
        "experiment_type": experiment_id,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "scene_id": scene_id,
        "run_path": export["run_path"],
        "metric_codes": metric_codes,
    }

    tags = [
        "llm-as-judge",
        f"env:{context.env}",
        f"nav_client:{context.nav_client}",
        f"experiment:{experiment_id}",
    ]
    if task_id:
        tags.append(f"task:{task_id}")
    if scene_id:
        tags.append(f"scene:{scene_id}")

    rows = _export_result_rows(export)
    experiment_exports = _collect_experiment_exports(export, run_path)

    with wandb.init(
        project=project,
        name=context.name,
        group=context.group,
        job_type="llm-as-judge",
        config=config,
        tags=tags,
    ) as run:
        run.summary["agent_color"] = export.get("agent_color")
        run.summary["selected_cube"] = export.get("selected_cube")
        run.summary["wrong_color"] = export.get("wrong_color")
        run.summary["completed"] = export.get("completed")
        run.summary["steps"] = export.get("steps")
        run.summary["correct"] = export.get("correct")
        run.summary["metric_count"] = metric_count
        run.summary["total_score"] = total_score
        run.summary["mean_score"] = mean_score
        run.summary["comparison_row_count"] = len(experiment_exports)

        table = wandb.Table(columns=_JUDGE_SCORE_COLUMNS)
        for row in rows:
            table.add_data(
                row["metric_code"],
                row["metric_name"],
                row["score"],
                row["score_norm"],
                row["justification"],
            )

        comparison_table = _build_experiment_comparison_table(wandb, experiment_exports)
        log_payload: Dict[str, Any] = {
            "judge/total_score": total_score,
            "judge/mean_score": mean_score,
            "judge_scores": table,
            "judge_scores_bar": wandb.plot.bar(
                table,
                "metric_code",
                "score",
                title="Judge scores",
            ),
            "judge_experiment_comparison": comparison_table,
        }

        for row in rows:
            code = row["metric_code"]
            log_payload[f"judge/score/{code}"] = row["score"]
            log_payload[f"judge/score_norm/{code}"] = row["score_norm"]

        radar_image = _build_radar_image(export)
        run.summary["radar_logged"] = radar_image is not None
        if radar_image is not None:
            log_payload["judge_scores_radar"] = wandb.Image(radar_image)

        run.log(log_payload)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = _build_judge_output_artifact(wandb, export, Path(tmp_dir))
            run.log_artifact(artifact)
