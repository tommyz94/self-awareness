"""
LLM-as-a-judge evaluation for mirror self-recognition runs.

Usage
-----
    python -m experiments.evaluation.judge outputs/run01.json
    python -m experiments.evaluation.judge outputs/run01.json --model qwen3.5:27b --metric M1 M3
    python -m experiments.evaluation.judge outputs/run01.json --prompt-variant strict
    python -m experiments.evaluation.judge outputs/run01.json --include-inner-reasoning
    python -m experiments.evaluation.judge outputs/run01.json --save outputs/run01_judge.json

Each metric scores the agent's reasoning transcript 0–5 using the rubrics defined here.
The LLM judge returns {"score": int, "justification": str} via Ollama structured output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from providers.ollama import chat
from experiments.evaluation.judge_metrics import (
    ALL_METRICS,
    METRIC_BY_CODE,
    M1_MIRROR_RECOGNITION,
    M2_SELF_ID_GROUNDING,
    M3_EVIDENCE_ACTION_CONSISTENCY,
    M4_ALTERNATIVE_HYPOTHESIS,
    Metric,
    normalize_metric_selection,
)
from experiments.evaluation.judge_prompts import (
    get_prompt_variants as _get_prompt_variants,
    resolve_prompt_variant,
)

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Compatibility aliases for one refactor cycle.
_METRIC_BY_NAME = METRIC_BY_CODE


def get_prompt_variants() -> Dict[str, str]:
    """Return the built-in prompt variants available for LLM-as-a-judge."""
    return _get_prompt_variants()


def _resolve_host(host: Optional[str] = None) -> str:
    return host or os.getenv("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST)


_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
        },
        "justification": {"type": "string"},
    },
    "required": ["score", "justification"],
    "additionalProperties": False,
}


def build_transcript(run: Dict[str, Any], include_inner_reasoning: bool = False) -> str:
    """
    Concatenate view_description, reasoning, and summary for every step,
    plus run-level metadata (agent color, final selection, steps taken).
    When include_inner_reasoning=True, also include the optional pre-output
    thinking text saved as inner_reasoning in each step record.
    """
    lines: List[str] = []

    lines.append("=== RUN METADATA ===")
    lines.append(f"Agent color  : {run.get('agent_color', 'unknown')}")
    lines.append(f"Selected cube: {run.get('selected_cube', 'none')}")
    lines.append(f"Completed    : {run.get('completed', False)}")
    lines.append(f"Total steps  : {run.get('steps', 0)}")
    if run.get("wrong_color"):
        lines.append(f"Misleading color in prompt: {run['wrong_color']}")

    step_records: List[Dict[str, Any]] = run.get("step_records", [])
    for record in step_records:
        step = record.get("step", "?")
        action = record.get("action", "?")
        direction = record.get("direction", "?")
        pos = record.get("position", [None, None])
        identification = record.get("identification", {})

        lines.append(f"\n--- Step {step} ---")
        lines.append(
            f"Action: {action}  |  Position: x={pos[0]}, z={pos[1]}  |  Facing: {direction}"
        )
        if identification:
            lines.append(
                f"Identification: my_color={identification.get('my_color', '?')} "
                f"source={identification.get('source', '?')} "
                f"confidence={identification.get('confidence', '?')}"
            )
        lines.append(f"View description:\n  {record.get('view_description', '').strip()}")
        lines.append(f"Reasoning:\n  {record.get('reasoning', '').strip()}")
        if include_inner_reasoning:
            inner_reasoning = record.get("inner_reasoning")
            if inner_reasoning:
                lines.append(f"Inner reasoning:\n  {str(inner_reasoning).strip()}")
        lines.append(f"Summary:\n  {record.get('summary', '').strip()}")

    return "\n".join(lines)


def _build_metric_user_prompt(metric: Metric, transcript: str) -> str:
    return (
        f"=== AGENT TRANSCRIPT ===\n{transcript}\n\n"
        f"=== METRIC: {metric.name} ===\n{metric.description}\n\n"
        f"=== RUBRIC ===\n{metric.rubric}\n\n"
        "Score the transcript above using the rubric (0–5). "
        "Return {\"score\": <int>, \"justification\": <your reasoning>}."
    )


@dataclass
class JudgeResult:
    metric_code: str
    metric_name: str
    score: int
    justification: str


def _score_metric(
    metric: Metric,
    transcript: str,
    *,
    host: Optional[str] = None,
    model: str = "llama3.2",
    temperature: float = 0.0,
    print_reasoning: bool = False,
    prompt_variant: str = "default",
) -> JudgeResult:
    host = _resolve_host(host)
    if print_reasoning:
        print(f"\n[{metric.name}] thinking ↓")

    messages = [
        {"role": "system", "content": resolve_prompt_variant(prompt_variant)},
        {"role": "user", "content": _build_metric_user_prompt(metric, transcript)},
    ]
    content, _ = chat(
        host=host,
        model=model,
        messages=messages,
        temperature=temperature,
        output_format=_JUDGE_SCHEMA,
        stream_thinking=print_reasoning,
    )
    try:
        parsed = json.loads(content)
        score = int(parsed["score"])
        justification = str(parsed.get("justification", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Judge returned unparseable output: {content!r}") from exc

    return JudgeResult(
        metric_code=metric.code,
        metric_name=metric.name,
        score=score,
        justification=justification,
    )


def judge_transcript(
    transcript: str,
    host: Optional[str] = None,
    model: str = "llama3.2",
    metrics: Optional[Sequence[Tuple[str, Metric]] | Sequence[Metric]] = None,
    temperature: float = 0.0,
    print_reasoning: bool = False,
    prompt_variant: str = "default",
) -> List[JudgeResult]:
    """Score a pre-built transcript with the selected judge metrics."""
    host = _resolve_host(host)
    selected_metrics = normalize_metric_selection(metrics)

    results: List[JudgeResult] = []
    for metric in selected_metrics:
        results.append(
            _score_metric(
                metric,
                transcript,
                host=host,
                model=model,
                temperature=temperature,
                print_reasoning=print_reasoning,
                prompt_variant=prompt_variant,
            )
        )

    return results


def judge_run(
    json_path: str | Path,
    host: Optional[str] = None,
    model: str = "llama3.2",
    metrics: Optional[Sequence[Tuple[str, Metric]] | Sequence[Metric]] = None,
    temperature: float = 0.0,
    print_reasoning: bool = False,
    prompt_variant: str = "default",
    include_inner_reasoning: bool = False,
) -> List[JudgeResult]:
    """
    Load a run JSON file, build the transcript, and score it with each metric.

    Parameters
    ----------
    json_path      : path to a run*.json file produced by the simulator.
    host           : Ollama host URL.
    model          : Ollama model to use as judge.
    metrics        : subset of metrics to run (default: all four).
    temperature    : LLM sampling temperature (0.0 = deterministic).
    print_reasoning: stream thinking tokens to stdout if the model supports it.
    include_inner_reasoning:
        when True, include step_records[*].inner_reasoning in the judged transcript.

    Returns
    -------
    List of JudgeResult, one per metric.
    """
    host = _resolve_host(host)
    run_data = json.loads(Path(json_path).read_text())

    task_id = run_data.get("task_id")
    if task_id is not None:
        from experiments.tasks import resolve_task

        if not resolve_task(
            task_id=task_id,
            experiment_id=run_data.get("experiment_id"),
        ).capabilities.supports_judge:
            raise NotImplementedError(
                f"llm_as_judge does not support task_id={task_id!r}. "
                "Implement a task-specific judge to evaluate this run."
            )

    transcript = build_transcript(
        run_data,
        include_inner_reasoning=include_inner_reasoning,
    )
    return judge_transcript(
        transcript=transcript,
        host=host,
        model=model,
        metrics=metrics,
        temperature=temperature,
        print_reasoning=print_reasoning,
        prompt_variant=prompt_variant,
    )


def print_results(results: List[JudgeResult], json_path: str | Path) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LLM-as-a-Judge — {Path(json_path).name}")
    print(sep)
    total = 0
    for result in results:
        print(f"\n  {result.metric_name}  →  {result.score}/5")
        for line in result.justification.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
        total += result.score
    max_score = len(results) * 5
    print(f"\n  TOTAL : {total}/{max_score}")
    print(f"{sep}\n")


def _build_judge_export(
    results: Sequence[JudgeResult],
    run_data: Dict[str, Any],
    *,
    json_path: str | Path,
    judge_model: str,
    prompt_variant: str,
    include_inner_reasoning: bool,
) -> Dict[str, Any]:
    from utils.integrations.wandb import _derive_run_context

    context = _derive_run_context(json_path)
    metric_codes = [result.metric_code for result in results]
    total_score = sum(result.score for result in results)
    mean_score = (total_score / len(results)) if results else 0.0

    agent_color = run_data.get("agent_color")
    selected_cube = run_data.get("selected_cube")
    correct = bool(selected_cube is not None and selected_cube == agent_color)

    return {
        "run_path": str(Path(json_path)),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a simulator run JSON with LLM-as-a-judge metrics.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("json_path", help="Path to a run*.json file (e.g. outputs/run01.json)")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST),
        help="Ollama host (default: OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model to use as judge (default: llama3.2)",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=list(METRIC_BY_CODE.keys()),
        metavar="M",
        help="Run only specific metrics: M1 M2 M3 M4 (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge (default: 0.0)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--print_reasoning",
        action="store_true",
        help="Stream the judge's thinking tokens to stdout (requires a reasoning model)",
    )
    parser.add_argument(
        "--include-inner-reasoning",
        action="store_true",
        help="Include step_records[*].inner_reasoning in the judged transcript.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(get_prompt_variants().keys()),
        default="default",
        help="Judge prompt variant to use (default: default)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log judge results to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default="self-aware",
        help="W&B project name (default: self-aware)",
    )
    parser.add_argument(
        "--wandb-run",
        metavar="RUN_NAME",
        help="Optional override for the derived W&B run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_metrics = (
        [METRIC_BY_CODE[metric_code] for metric_code in args.metric]
        if args.metric
        else list(ALL_METRICS)
    )
    results = judge_run(
        json_path=args.json_path,
        host=args.host,
        model=args.model,
        metrics=selected_metrics,
        temperature=args.temperature,
        print_reasoning=args.print_reasoning,
        prompt_variant=args.prompt_variant,
        include_inner_reasoning=args.include_inner_reasoning,
    )
    print_results(results, args.json_path)

    judge_export: Optional[Dict[str, Any]] = None
    run_data: Optional[Dict[str, Any]] = None
    if args.save or args.wandb:
        run_data = json.loads(Path(args.json_path).read_text())
        judge_export = _build_judge_export(
            results,
            run_data,
            json_path=args.json_path,
            judge_model=args.model,
            prompt_variant=args.prompt_variant,
            include_inner_reasoning=args.include_inner_reasoning,
        )

    if args.save:
        payload = judge_export or {}
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {args.save}")

    if args.wandb:
        from utils.integrations.wandb import log_judge_results

        if judge_export is None or run_data is None:
            raise RuntimeError("Internal error: missing judge export for W&B logging.")
        log_judge_results(
            results,
            run_data,
            judge_model=args.model,
            prompt_variant=args.prompt_variant,
            include_inner_reasoning=args.include_inner_reasoning,
            run_path=args.json_path,
            project=args.wandb_project,
            run_name=args.wandb_run,
            judge_export=judge_export,
        )


if __name__ == "__main__":
    main()
