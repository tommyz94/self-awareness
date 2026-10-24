#!/usr/bin/env python3
"""
Launch one macOS Terminal window per experiment for a predefined API profile.

Profiles:
- gpt51: direct OpenAI API using gpt-5.1
- sonnet46: OpenRouter using anthropic/claude-sonnet-4.6 with Anthropic-only routing

By default this script previews the generated jobs. Pass --launch to actually
open Terminal windows in parallel.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

DEFAULT_EXPERIMENTS = ["e1", "e2", "e3", "e4", "e5"]
RESERVED_OVERRIDE_KEYS = {
    "client",
    "experiment",
    "nav",
    "num_runs",
    "max_steps",
    "outputs.dir",
    "mujoco.headless",
}

PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    "gpt51": {
        "client": "openai",
        "model": "gpt-5.1",
        "env": ["OPENAI_API_KEY"],
        "output_dir": "outputs/paper/openai_gpt51",
        "hydra_args": {},
        "log_dir": "logs/api_parallel_profile/gpt51",
        "description": "Direct OpenAI API with gpt-5.1.",
    },
    "sonnet46": {
        "client": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "env": ["OPENROUTER_API_KEY"],
        "output_dir": "outputs/paper/claude_sonnet46",
        "hydra_args": {
            "openrouter.require_parameters": True,
            "openrouter.allow_fallbacks": False,
            "openrouter.provider_only": "[anthropic]",
            "openrouter.min_request_interval_sec": 1.0,
        },
        "log_dir": "logs/api_parallel_profile/sonnet46",
        "description": "OpenRouter route for Claude Sonnet 4.6 with Anthropic-only routing.",
    },
}

PROFILE_ALIASES = {
    "gpt51": "gpt51",
    "gpt-5.1": "gpt51",
    "openai-gpt51": "gpt51",
    "openai/gpt-5.1": "gpt51",
    "sonnet46": "sonnet46",
    "claude-sonnet-4.6": "sonnet46",
    "anthropic/claude-sonnet-4.6": "sonnet46",
    "openrouter-sonnet46": "sonnet46",
}


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "job"


def _comma_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_profile(raw: str) -> str:
    key = raw.strip().lower()
    profile = PROFILE_ALIASES.get(key)
    if profile is None:
        valid = ", ".join(sorted(PROFILE_ALIASES))
        raise SystemExit(f"Unknown --profile '{raw}'. Supported values: {valid}")
    return profile


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _parse_set_overrides(values: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --set override '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --set override '{raw}'. Key cannot be empty.")
        if key in RESERVED_OVERRIDE_KEYS:
            raise SystemExit(f"--set cannot override reserved key '{key}'.")
        overrides[key] = value
    return overrides


def _require_env(names: List[str]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for name in names:
        value = os.getenv(name)
        if not value:
            raise SystemExit(f"Required environment variable '{name}' is not set.")
        resolved[name] = value
    return resolved


def _build_python_command(
    *,
    python_bin: str,
    experiment: str,
    profile: Mapping[str, Any],
    num_runs: int,
    max_steps: int,
    output_dir: str,
    extra_overrides: Mapping[str, str],
) -> List[str]:
    command: List[str] = [
        python_bin,
        "main.py",
        f"experiment={experiment}",
        f"client={profile['client']}",
        f"nav={profile['model']}",
        f"num_runs={num_runs}",
        f"max_steps={max_steps}",
        "mujoco.headless=true",
        f"outputs.dir={output_dir}",
    ]
    for key, value in profile.get("hydra_args", {}).items():
        command.append(f"{key}={_hydra_value(value)}")
    for key, value in extra_overrides.items():
        command.append(f"{key}={value}")
    return command


def _build_shell_command(
    *,
    repo_root: Path,
    python_command: List[str],
    job_name: str,
    log_file: Path,
    copied_env: Mapping[str, str],
    keep_open: bool,
) -> str:
    exports = [f"export {key}={shlex.quote(value)}" for key, value in copied_env.items()]
    exports.append("export PYTHONUNBUFFERED=1")
    shell_parts = [
        "set -uo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(log_file.parent))}",
    ]
    shell_parts.extend(exports)
    shell_parts.append(f"echo {shlex.quote(f'[profile] Starting {job_name}')}")
    shell_parts.append(f"{shlex.join(python_command)} 2>&1 | tee {shlex.quote(str(log_file))}")
    shell_parts.append("status=${PIPESTATUS[0]}")
    shell_parts.append(
        f"echo {shlex.quote(f'[profile] Finished {job_name} with exit status')} \"$status\""
    )
    if keep_open:
        shell_parts.append("read -r -p 'Press Enter to close...' _")
    shell_parts.append("exit \"$status\"")
    return "bash -lc " + shlex.quote("; ".join(shell_parts))


def _quote_applescript(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _prepare_jobs(
    *,
    repo_root: Path,
    profile_name: str,
    experiments: List[str],
    python_bin: str,
    num_runs: int,
    max_steps: int,
    output_dir: Optional[str],
    batch_id: str,
    extra_overrides: Mapping[str, str],
    keep_open: bool,
) -> List[Dict[str, Any]]:
    profile = PROFILE_SPECS[profile_name]
    copied_env = _require_env(list(profile["env"]))
    resolved_output_dir = output_dir or str(profile["output_dir"])
    batch_dir = repo_root / str(profile["log_dir"]) / batch_id

    jobs: List[Dict[str, Any]] = []
    for experiment in experiments:
        job_name = f"{experiment}__{_slug(profile_name)}"
        log_file = batch_dir / f"{job_name}.log"
        python_command = _build_python_command(
            python_bin=python_bin,
            experiment=experiment,
            profile=profile,
            num_runs=num_runs,
            max_steps=max_steps,
            output_dir=resolved_output_dir,
            extra_overrides=extra_overrides,
        )
        shell_command = _build_shell_command(
            repo_root=repo_root,
            python_command=python_command,
            job_name=job_name,
            log_file=log_file,
            copied_env=copied_env,
            keep_open=keep_open,
        )
        jobs.append(
            {
                "name": job_name,
                "experiment": experiment,
                "profile": profile_name,
                "description": profile["description"],
                "output_dir": resolved_output_dir,
                "log_file": log_file,
                "python_command": python_command,
                "shell_command": shell_command,
            }
        )
    return jobs


def _print_jobs(jobs: List[Dict[str, Any]], batch_id: str) -> None:
    print(f"Batch id: {batch_id}")
    print(f"Prepared {len(jobs)} job(s).")
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}] {job['name']}")
        print(f"    experiment: {job['experiment']}")
        print(f"    profile: {job['profile']}")
        print(f"    output_dir: {job['output_dir']}")
        print(f"    log: {job['log_file']}")
        print(f"    command: {shlex.join(job['python_command'])}")


def _launch_in_terminal(
    *,
    jobs: List[Dict[str, Any]],
    terminal_app: str,
    pause_seconds: float,
) -> None:
    if sys.platform != "darwin":
        raise SystemExit("Terminal launching is currently supported only on macOS.")
    if not jobs:
        return

    subprocess.run(
        ["osascript", "-e", f'tell application "{terminal_app}" to activate'],
        check=True,
    )
    for job in jobs:
        script_literal = _quote_applescript(job["shell_command"])
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "{terminal_app}" to do script {script_literal}',
            ],
            check=True,
        )
        if pause_seconds > 0:
            time.sleep(pause_seconds)


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Preview or launch one Terminal window per experiment for a chosen API profile."
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile to launch: gpt51 or sonnet46.",
    )
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment ids. Default: e1,e2,e3,e4,e5.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=7,
        help="num_runs override for every launched job.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="max_steps override for every launched job.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional outputs.dir root override. Defaults to the profile-specific root.",
    )
    parser.add_argument(
        "--batch-id",
        help="Optional batch id used in log paths. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Open one macOS Terminal window per job. Without this flag the script only prints the jobs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override applied to every job, e.g. --set seed=123.",
    )
    parser.add_argument(
        "--terminal-app",
        default="Terminal",
        help="macOS terminal app name used by osascript. Default: Terminal.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between Terminal launches.",
    )
    parser.add_argument(
        "--close-on-exit",
        action="store_true",
        help="Do not keep each Terminal tab open after the command exits.",
    )
    args = parser.parse_args()

    profile_name = _normalize_profile(args.profile)
    experiments = _comma_list(args.experiments)
    if not experiments:
        raise SystemExit("No experiments selected.")
    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be positive.")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive.")
    if args.pause_seconds < 0:
        raise SystemExit("--pause-seconds must be non-negative.")

    python_bin = _default_python_bin(root)
    batch_id = args.batch_id or time.strftime("%Y%m%d_%H%M%S")
    extra_overrides = _parse_set_overrides(args.set)
    jobs = _prepare_jobs(
        repo_root=root,
        profile_name=profile_name,
        experiments=experiments,
        python_bin=python_bin,
        num_runs=args.num_runs,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        batch_id=batch_id,
        extra_overrides=extra_overrides,
        keep_open=not args.close_on_exit,
    )

    print(f"Profile: {profile_name}")
    print(f"Description: {PROFILE_SPECS[profile_name]['description']}")
    _print_jobs(jobs, batch_id=batch_id)
    if not args.launch:
        print("Dry run only. Re-run with --launch to open Terminal windows.")
        return

    _launch_in_terminal(
        jobs=jobs,
        terminal_app=args.terminal_app,
        pause_seconds=args.pause_seconds,
    )
    print("Launch complete.")


if __name__ == "__main__":
    main()
