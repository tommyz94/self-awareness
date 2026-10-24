#!/usr/bin/env python3
"""
Run the built-in experiment sweep sequentially on Vertex AI Gemini models.

This launcher is intentionally conservative:
- no parallelism
- optional per-model smoke checks
- aggressive Google retry settings for transient 429/504 failures
- optional cooldown between jobs

By default it runs experiments e1/e2/e3/e4/e5 for both gemini-2.5-flash
and gemini-2.5-pro, forcing num_runs=7 and max_steps=100 for every job.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_EXPERIMENTS = [ "e4", "e5"]
DEFAULT_MODELS = ["gemini-2.5-pro"]
DEFAULT_MODEL_OUTPUT_DIRS = {
    "gemini-2.5-pro": "outputs/paper/gemini25pro",
}
DEFAULT_LAUNCHER_SUMMARY_DIR = "outputs/launcher_runs"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_python_bin(root: Path) -> str:
    explicit = os.getenv("PYTHON_BIN", "").strip()
    if explicit:
        return explicit
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _comma_list(raw: str) -> List[str]:
    values: List[str] = []
    for part in raw.split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def _parse_set_overrides(values: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --set override '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --set override '{raw}'. Key cannot be empty.")
        overrides[key] = value
    return overrides


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _build_job_command(
    *,
    python_bin: str,
    project: str,
    location: str,
    model: str,
    experiment: str,
    output_dir: str,
    num_runs: int,
    max_steps: int,
    max_retries: int,
    initial_retry_delay_sec: float,
    max_retry_delay_sec: float,
    max_output_tokens: Optional[int],
    extra_overrides: Dict[str, str],
) -> List[str]:
    command: List[str] = [
        python_bin,
        "main.py",
        f"experiment={experiment}",
        "client=google",
        f"nav={model}",
        f"num_runs={num_runs}",
        f"max_steps={max_steps}",
        "mujoco.headless=true",
        f"outputs.dir={output_dir}",
        "google.use_vertexai=true",
        f"google.project={project}",
        f"google.location={location}",
        f"google.max_retries={max_retries}",
        f"google.initial_retry_delay_sec={initial_retry_delay_sec}",
        f"google.max_retry_delay_sec={max_retry_delay_sec}",
    ]
    if max_output_tokens is not None:
        command.append(f"google.max_output_tokens={max_output_tokens}")
    for key, value in extra_overrides.items():
        command.append(f"{key}={value}")
    return command


def _resolve_output_dir(
    *,
    requested_output_dir: str,
    model: str,
) -> str:
    requested = requested_output_dir.strip()
    if requested and requested.lower() != "auto":
        return requested
    return DEFAULT_MODEL_OUTPUT_DIRS.get(model, "outputs")


def _resolve_launcher_summary_root(requested_output_dir: str) -> str:
    requested = requested_output_dir.strip()
    if requested and requested.lower() != "auto":
        return requested
    return DEFAULT_LAUNCHER_SUMMARY_DIR


def _run_command(
    command: List[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> int:
    printable = shlex.join(command)
    print(printable, flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd))
    return int(completed.returncode)


def _run_smoke_check(
    *,
    python_bin: str,
    cwd: Path,
    project: str,
    location: str,
    model: str,
    dry_run: bool,
) -> int:
    smoke_cmd = [
        python_bin,
        "scripts/dev/google_vertex_smoke.py",
        "--project",
        project,
        "--location",
        location,
        "--model",
        model,
        "--no-image",
    ]
    print(f"\n[smoke] model={model}", flush=True)
    return _run_command(smoke_cmd, cwd=cwd, dry_run=dry_run)


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Run the built-in experiment sweep sequentially on Vertex AI Gemini.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("PROJECT_ID", "gemini-self-perspetion"),
        help="Google Cloud project id.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("LOCATION", "global"),
        help="Vertex AI location, e.g. global or us-central1.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Gemini model ids to run sequentially.",
    )
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment ids to run sequentially.",
    )
    parser.add_argument(
        "--output-dir",
        default="auto",
        help=(
            "Shared outputs.dir root for all launched jobs. Use 'auto' to reuse "
            "known per-model roots when available."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=7,
        help="num_runs override for every job.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="max_steps override for every job.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=30,
        help="google.max_retries override for every job.",
    )
    parser.add_argument(
        "--initial-retry-delay-sec",
        type=float,
        default=10.0,
        help="google.initial_retry_delay_sec override for every job.",
    )
    parser.add_argument(
        "--max-retry-delay-sec",
        type=float,
        default=180.0,
        help="google.max_retry_delay_sec override for every job.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help=(
            "Optional google.max_output_tokens override for every job. "
            "Pass 0 to leave it unset."
        ),
    )
    parser.add_argument(
        "--sleep-between-jobs-sec",
        type=float,
        default=30.0,
        help="Cooldown between sequential jobs.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the lightweight google_vertex_smoke.py check for each model.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on the first failed smoke or experiment job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command without executing it.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra Hydra override applied to every job, e.g. --set seed=123.",
    )
    args = parser.parse_args()

    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative.")
    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be positive.")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive.")
    if args.initial_retry_delay_sec <= 0:
        raise SystemExit("--initial-retry-delay-sec must be positive.")
    if args.max_retry_delay_sec <= 0:
        raise SystemExit("--max-retry-delay-sec must be positive.")
    if args.sleep_between_jobs_sec < 0:
        raise SystemExit("--sleep-between-jobs-sec must be non-negative.")

    models = _comma_list(args.models)
    experiments = _comma_list(args.experiments)
    if not models:
        raise SystemExit("No models selected.")
    if not experiments:
        raise SystemExit("No experiments selected.")

    python_bin = _default_python_bin(root)
    max_output_tokens = None if int(args.max_output_tokens) == 0 else int(args.max_output_tokens)
    extra_overrides = _parse_set_overrides(args.set)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    launcher_summary_root = _resolve_launcher_summary_root(args.output_dir)
    launcher_dir = root / launcher_summary_root / session_id
    if not args.dry_run:
        launcher_dir.mkdir(parents=True, exist_ok=True)

    print("Sequential Vertex AI sweep", flush=True)
    print(f"  project={args.project}", flush=True)
    print(f"  location={args.location}", flush=True)
    print(f"  models={', '.join(models)}", flush=True)
    print(f"  experiments={', '.join(experiments)}", flush=True)
    print(f"  python_bin={python_bin}", flush=True)
    if args.output_dir.strip().lower() == "auto":
        print(
            f"  output_dir=auto ({json.dumps(DEFAULT_MODEL_OUTPUT_DIRS, sort_keys=True)})",
            flush=True,
        )
        print(f"  launcher_summary_root={launcher_summary_root}", flush=True)
    else:
        print(f"  output_dir={args.output_dir}", flush=True)
        print(f"  launcher_summary_root={launcher_summary_root}", flush=True)
    print(f"  num_runs={args.num_runs}", flush=True)
    print(f"  max_steps={args.max_steps}", flush=True)
    print(f"  google.max_retries={args.max_retries}", flush=True)
    print(
        "  google.retry_backoff="
        f"{args.initial_retry_delay_sec}s..{args.max_retry_delay_sec}s",
        flush=True,
    )
    print(
        f"  google.max_output_tokens={'unset' if max_output_tokens is None else max_output_tokens}",
        flush=True,
    )
    print(f"  sleep_between_jobs_sec={args.sleep_between_jobs_sec}", flush=True)
    if extra_overrides:
        print(f"  shared_overrides={json.dumps(extra_overrides, sort_keys=True)}", flush=True)

    results: List[Dict[str, Any]] = []

    for model in models:
        if not args.skip_smoke:
            smoke_exit = _run_smoke_check(
                python_bin=python_bin,
                cwd=root,
                project=args.project,
                location=args.location,
                model=model,
                dry_run=args.dry_run,
            )
            smoke_record = {
                "kind": "smoke",
                "model": model,
                "experiment": None,
                "exit_code": smoke_exit,
            }
            results.append(smoke_record)
            if smoke_exit != 0 and args.stop_on_error:
                if not args.dry_run:
                    (launcher_dir / "summary.json").write_text(
                        json.dumps(results, indent=2),
                        encoding="utf-8",
                    )
                return smoke_exit

        for experiment in experiments:
            print(f"\n[job] model={model} experiment={experiment}", flush=True)
            resolved_output_dir = _resolve_output_dir(
                requested_output_dir=args.output_dir,
                model=model,
            )
            command = _build_job_command(
                python_bin=python_bin,
                project=args.project,
                location=args.location,
                model=model,
                experiment=experiment,
                output_dir=resolved_output_dir,
                num_runs=args.num_runs,
                max_steps=args.max_steps,
                max_retries=args.max_retries,
                initial_retry_delay_sec=args.initial_retry_delay_sec,
                max_retry_delay_sec=args.max_retry_delay_sec,
                max_output_tokens=max_output_tokens,
                extra_overrides=extra_overrides,
            )
            started_at = time.time()
            exit_code = _run_command(command, cwd=root, dry_run=args.dry_run)
            duration_sec = time.time() - started_at
            results.append(
                {
                    "kind": "experiment",
                    "model": model,
                    "experiment": experiment,
                    "exit_code": exit_code,
                    "duration_sec": round(duration_sec, 2),
                }
            )

            if not args.dry_run:
                (launcher_dir / "summary.json").write_text(
                    json.dumps(results, indent=2),
                    encoding="utf-8",
                )

            if exit_code != 0 and args.stop_on_error:
                return exit_code

            if args.sleep_between_jobs_sec > 0 and not args.dry_run:
                time.sleep(args.sleep_between_jobs_sec)

    failures = [item for item in results if item["exit_code"] != 0]
    print("\nSweep summary", flush=True)
    print(f"  total_records={len(results)}", flush=True)
    print(f"  failures={len(failures)}", flush=True)
    if not args.dry_run:
        print(f"  summary_file={launcher_dir / 'summary.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
