"""
Judge reliability evaluation for LLM-as-a-judge metrics.

Usage
-----
    python -m experiments.evaluation.judge_reliability outputs/mujoco/e1/ollama/gemma3_4b/runs \
        --model qwen3.5:4b --model llama3.2

    python -m experiments.evaluation.judge_reliability outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json \
        outputs/mujoco/e1/ollama/gemma3_4b/runs/run02.json \
        --model qwen3.5:4b \
        --prompt-variant default strict \
        --include-inner-reasoning \
        --save outputs/mujoco/e1/ollama/gemma3_4b/runs/judge_reliability.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.evaluation.judge_metrics import METRIC_BY_CODE, Metric, normalize_metric_selection
from experiments.evaluation.judge_prompts import get_prompt_variants
from experiments.evaluation.judge import _DEFAULT_OLLAMA_HOST, build_transcript, judge_transcript

_RUN_JSON_RE = re.compile(r"^run\d+\.json$")


@dataclass(frozen=True)
class JudgeCondition:
    model: str
    prompt_variant: str

    @property
    def label(self) -> str:
        return f"{self.model} [{self.prompt_variant}]"

    def to_dict(self) -> Dict[str, str]:
        return {
            "model": self.model,
            "prompt_variant": self.prompt_variant,
            "label": self.label,
        }


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _collect_run_json_paths(paths: Sequence[str | Path]) -> List[Path]:
    collected: List[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if not _RUN_JSON_RE.match(path.name):
                raise ValueError(
                    f"Expected a run JSON named like run01.json, got: {path.name}"
                )
            collected.append(path.resolve())
            continue

        for candidate in sorted(path.rglob("*.json")):
            if _RUN_JSON_RE.match(candidate.name):
                collected.append(candidate.resolve())

    deduped: List[Path] = []
    seen_paths: set[Path] = set()
    for path in collected:
        if path not in seen_paths:
            seen_paths.add(path)
            deduped.append(path)

    if not deduped:
        raise ValueError("No run JSON files found. Expected files named like run01.json.")

    return deduped


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)

    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = math.sqrt(float(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered)))
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def _spearman_correlation(a: Sequence[int], b: Sequence[int]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size == 0 or arr_a.size != arr_b.size:
        return float("nan")
    return _pearson_corr(_average_ranks(arr_a), _average_ranks(arr_b))


def _cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    arr_a = np.asarray(a, dtype=int)
    arr_b = np.asarray(b, dtype=int)
    if arr_a.size == 0 or arr_a.size != arr_b.size:
        return float("nan")

    num_labels = 6  # judge scores are integers 0..5
    confusion = np.zeros((num_labels, num_labels), dtype=float)
    for score_a, score_b in zip(arr_a, arr_b):
        confusion[score_a, score_b] += 1.0

    total = float(arr_a.size)
    po = float(np.trace(confusion) / total)
    p_a = confusion.sum(axis=1) / total
    p_b = confusion.sum(axis=0) / total
    pe = float(np.dot(p_a, p_b))
    denom = 1.0 - pe
    if denom == 0.0:
        return 1.0 if np.array_equal(arr_a, arr_b) else float("nan")
    return float((po - pe) / denom)


def _mean_absolute_difference(a: Sequence[int], b: Sequence[int]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size == 0 or arr_a.size != arr_b.size:
        return float("nan")
    return float(np.mean(np.abs(arr_a - arr_b)))


def _round_or_none(value: float, digits: int = 6) -> float | None:
    if math.isnan(value):
        return None
    return round(value, digits)


def _pairwise_agreement(
    score_vectors: Dict[JudgeCondition, List[int]],
    conditions: Sequence[JudgeCondition],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for cond_a, cond_b in itertools.combinations(conditions, 2):
        scores_a = score_vectors[cond_a]
        scores_b = score_vectors[cond_b]
        pairs.append(
            {
                "judge_a": cond_a.to_dict(),
                "judge_b": cond_b.to_dict(),
                "num_items": len(scores_a),
                "mean_absolute_difference": _round_or_none(
                    _mean_absolute_difference(scores_a, scores_b)
                ),
                "cohen_kappa": _round_or_none(_cohen_kappa(scores_a, scores_b)),
                "spearman_correlation": _round_or_none(
                    _spearman_correlation(scores_a, scores_b)
                ),
            }
        )
    return pairs


def evaluate_judge_reliability(
    paths: Sequence[str | Path],
    models: Sequence[str],
    prompt_variants: Sequence[str],
    metrics: Sequence[Tuple[str, Metric]] | Sequence[Metric] | None = None,
    host: str | None = None,
    temperature: float = 0.0,
    print_reasoning: bool = False,
    include_inner_reasoning: bool = False,
) -> Dict[str, Any]:
    host = host or os.getenv("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST)
    run_paths = _collect_run_json_paths(paths)
    selected_metrics = normalize_metric_selection(metrics)
    selected_models = _dedupe_preserve_order(models)
    selected_variants = _dedupe_preserve_order(prompt_variants)

    conditions = [
        JudgeCondition(model=model, prompt_variant=variant)
        for model in selected_models
        for variant in selected_variants
    ]
    if len(conditions) < 2:
        raise ValueError(
            "Judge reliability requires at least two judge conditions. "
            "Use multiple --model values, multiple --prompt-variant values, or both."
        )

    score_vectors_by_metric: Dict[str, Dict[JudgeCondition, List[int]]] = {
        metric.code: {condition: [] for condition in conditions}
        for metric in selected_metrics
    }
    overall_vectors: Dict[JudgeCondition, List[int]] = {condition: [] for condition in conditions}
    raw_results: List[Dict[str, Any]] = []

    print(
        f"Scoring {len(run_paths)} run files across {len(conditions)} judge conditions "
        f"and {len(selected_metrics)} metrics."
    )

    for run_path in run_paths:
        run_data = json.loads(run_path.read_text())
        task_id = run_data.get("task_id")
        if task_id is not None:
            from experiments.tasks import resolve_task

            if not resolve_task(
                task_id=task_id,
                experiment_id=run_data.get("experiment_id"),
            ).capabilities.supports_judge:
                raise NotImplementedError(
                    f"judge_reliability does not support task_id={task_id!r} "
                    f"in {run_path}. Implement a task-specific judge to evaluate this run."
                )
        transcript = build_transcript(
            run_data,
            include_inner_reasoning=include_inner_reasoning,
        )

        for condition in conditions:
            results = judge_transcript(
                transcript=transcript,
                host=host,
                model=condition.model,
                metrics=selected_metrics,
                temperature=temperature,
                print_reasoning=print_reasoning,
                prompt_variant=condition.prompt_variant,
            )

            raw_results.append(
                {
                    "run_path": str(run_path),
                    "run_id": run_data.get("run_id"),
                    "agent_color": run_data.get("agent_color"),
                    "wrong_color": run_data.get("wrong_color"),
                    "judge_condition": condition.to_dict(),
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
            )

            for result in results:
                score_vectors_by_metric[result.metric_code][condition].append(result.score)
                overall_vectors[condition].append(result.score)

    per_metric: Dict[str, Any] = {}
    metric_name_lookup = {metric.code: metric.name for metric in selected_metrics}
    for metric_code, score_vectors in score_vectors_by_metric.items():
        per_metric[metric_code] = {
            "metric_name": metric_name_lookup[metric_code],
            "pairs": _pairwise_agreement(score_vectors, conditions),
        }

    payload = {
        "inputs": [str(Path(path)) for path in paths],
        "resolved_run_files": [str(path) for path in run_paths],
        "models": selected_models,
        "prompt_variants": selected_variants,
        "metrics": [metric.code for metric in selected_metrics],
        "include_inner_reasoning": include_inner_reasoning,
        "num_runs": len(run_paths),
        "num_conditions": len(conditions),
        "raw_results": raw_results,
        "agreement": {
            "per_metric": per_metric,
            "overall": {
                "pairs": _pairwise_agreement(overall_vectors, conditions),
            },
        },
    }
    return payload


def print_reliability_summary(report: Dict[str, Any]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(
        "  Judge Reliability"
        f"  ({report['num_runs']} runs, {report['num_conditions']} conditions)"
    )
    print(sep)
    print(
        "  Transcript includes inner_reasoning: "
        f"{report.get('include_inner_reasoning', False)}"
    )

    print("  Conditions:")
    for model in report["models"]:
        for prompt_variant in report["prompt_variants"]:
            print(f"  - {model} [{prompt_variant}]")

    for metric_code, metric_report in report["agreement"]["per_metric"].items():
        print(f"\n  {metric_code} — {metric_report['metric_name']}")
        for pair in metric_report["pairs"]:
            print(
                "    "
                f"{pair['judge_a']['label']} vs {pair['judge_b']['label']} | "
                f"MAD={pair['mean_absolute_difference']} | "
                f"kappa={pair['cohen_kappa']} | "
                f"spearman={pair['spearman_correlation']}"
            )

    print("\n  Overall")
    for pair in report["agreement"]["overall"]["pairs"]:
        print(
            "    "
            f"{pair['judge_a']['label']} vs {pair['judge_b']['label']} | "
            f"MAD={pair['mean_absolute_difference']} | "
            f"kappa={pair['cohen_kappa']} | "
            f"spearman={pair['spearman_correlation']}"
        )
    print(f"{sep}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run judge reliability analysis over saved run JSON files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run JSON files or directories containing run*.json files.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Judge model to evaluate. Repeat for multiple models.",
    )
    parser.add_argument(
        "--prompt-variant",
        nargs="+",
        choices=sorted(get_prompt_variants().keys()),
        default=["default", "strict"],
        help="Judge prompt variants to compare (default: default strict)",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=list(METRIC_BY_CODE.keys()),
        metavar="M",
        help="Run only specific metrics: M1 M2 M3 M4 (default: all)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST),
        help="Ollama host (default: OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save the full reliability report as JSON to this path.",
    )
    parser.add_argument(
        "--print_reasoning",
        action="store_true",
        help="Stream judge thinking tokens to stdout when supported by the model.",
    )
    parser.add_argument(
        "--include-inner-reasoning",
        action="store_true",
        help="Include step_records[*].inner_reasoning in the judged transcript.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_metrics = (
        [METRIC_BY_CODE[metric_code] for metric_code in args.metric]
        if args.metric
        else list(METRIC_BY_CODE.values())
    )
    report = evaluate_judge_reliability(
        paths=args.paths,
        models=args.model,
        prompt_variants=args.prompt_variant,
        metrics=selected_metrics,
        host=args.host,
        temperature=args.temperature,
        print_reasoning=args.print_reasoning,
        include_inner_reasoning=args.include_inner_reasoning,
    )
    print_reliability_summary(report)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2))
        print(f"Reliability report saved to {save_path}")


if __name__ == "__main__":
    main()
