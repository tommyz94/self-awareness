#!/usr/bin/env python3
"""
Launch an OpenRouter experiment matrix in parallel across Terminal windows.

The matrix is defined in a JSON file so models, experiments, and shared Hydra
arguments can be managed in one place. By default the script prints a dry-run
preview; pass ``--launch`` to open one macOS Terminal window per job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_RESERVED_JOB_KEYS = {"client", "experiment", "nav"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "job"


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config file is not valid JSON: {path}\n{exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config root must be a JSON object: {path}")
    return data


def _comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def _select_values(values: Iterable[str], selected: Optional[List[str]], label: str) -> List[str]:
    all_values = list(values)
    if selected is None:
        return all_values

    missing = [value for value in selected if value not in all_values]
    if missing:
        raise SystemExit(
            f"Unknown {label}: {', '.join(missing)}. Available: {', '.join(all_values)}"
        )
    return selected


def _mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"Config field '{field_name}' must be a JSON object.")
    return dict(value)


def _parse_set_overrides(values: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --set override '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --set override '{raw}'. Key cannot be empty.")
        if key in _RESERVED_JOB_KEYS:
            raise SystemExit(
                f"--set cannot override '{key}'. Use the matrix config for {key}."
            )
        overrides[key] = value
    return overrides


def _preset_locked_keys(repo_root: Path, experiment: str) -> set[str]:
    preset_path = repo_root / "conf" / "experiment" / f"{experiment}.yaml"
    if not preset_path.exists():
        raise SystemExit(f"Experiment preset not found: {preset_path}")

    locked_keys: set[str] = set()
    for raw_line in preset_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line[:1].isspace():
            continue
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        if key:
            locked_keys.add(key)
    return locked_keys


def _validate_unlocked_args(
    *,
    args: Mapping[str, Any],
    locked_keys: set[str],
    experiment: str,
) -> None:
    blocked = sorted(key for key in args if key in locked_keys)
    if blocked:
        raise SystemExit(
            "Refusing to override experiment-owned preset keys for "
            f"'{experiment}': {', '.join(blocked)}. "
            "Remove them from shared_args, per_model_args, per_experiment_args, "
            "pair_args, or --set."
        )


def _job_args(
    config: Mapping[str, Any],
    model: str,
    experiment: str,
    cli_overrides: Mapping[str, str],
    locked_keys: set[str],
) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    args.update(_mapping(config.get("shared_args"), "shared_args"))
    args.update(_mapping(_mapping(config.get("per_model_args"), "per_model_args").get(model), f"per_model_args.{model}"))
    args.update(
        _mapping(
            _mapping(config.get("per_experiment_args"), "per_experiment_args").get(experiment),
            f"per_experiment_args.{experiment}",
        )
    )

    pair_args = _mapping(config.get("pair_args"), "pair_args")
    model_pair_args = _mapping(pair_args.get(model), f"pair_args.{model}")
    args.update(_mapping(model_pair_args.get(experiment), f"pair_args.{model}.{experiment}"))
    args.update(cli_overrides)
    _validate_unlocked_args(args=args, locked_keys=locked_keys, experiment=experiment)

    args["client"] = str(config.get("client", "openrouter"))
    args["experiment"] = experiment
    args["nav"] = model
    return args


def _build_python_command(
    repo_root: Path,
    config: Mapping[str, Any],
    model: str,
    experiment: str,
    cli_overrides: Mapping[str, str],
    locked_keys: set[str],
) -> List[str]:
    python_bin = str(config.get("python_bin", "./.venv/bin/python"))
    main_script = str(config.get("main_script", "main.py"))
    raw_args = _job_args(config, model, experiment, cli_overrides, locked_keys)
    extra_cli_args = config.get("extra_cli_args", [])
    if not isinstance(extra_cli_args, list):
        raise SystemExit("Config field 'extra_cli_args' must be a JSON array.")

    command: List[str] = [python_bin, main_script]
    for key, value in raw_args.items():
        command.append(f"{key}={_hydra_value(value)}")
    command.extend(str(item) for item in extra_cli_args)
    return command


def _require_copied_env(config: Mapping[str, Any]) -> Dict[str, str]:
    copy_env = config.get("copy_env", [])
    if not isinstance(copy_env, list):
        raise SystemExit("Config field 'copy_env' must be a JSON array.")

    resolved: Dict[str, str] = {}
    for name in copy_env:
        if not isinstance(name, str) or not name:
            raise SystemExit("Config field 'copy_env' must contain non-empty strings.")
        value = os.environ.get(name)
        if value is None:
            raise SystemExit(
                f"Environment variable '{name}' is required by copy_env but is not set."
            )
        resolved[name] = value
    return resolved


def _build_shell_command(
    *,
    repo_root: Path,
    python_command: List[str],
    job_name: str,
    log_file: Path,
    copied_env: Mapping[str, str],
    config: Mapping[str, Any],
) -> str:
    launcher = _mapping(config.get("launcher"), "launcher")
    keep_open = bool(launcher.get("keep_open", True))

    env_vars = _mapping(config.get("env_vars"), "env_vars")
    exports: List[str] = []
    for key, value in {**env_vars, **copied_env}.items():
        exports.append(f"export {key}={shlex.quote(str(value))}")

    shell_parts = [
        "set -uo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(log_file.parent))}",
    ]
    shell_parts.extend(exports)
    shell_parts.append(f"echo {shlex.quote(f'[matrix] Starting {job_name}')}")
    shell_parts.append(
        f"{shlex.join(python_command)} 2>&1 | tee {shlex.quote(str(log_file))}"
    )
    shell_parts.append("status=${PIPESTATUS[0]}")
    shell_parts.append(
        f"echo {shlex.quote(f'[matrix] Finished {job_name} with exit status')} \"$status\""
    )
    if keep_open:
        shell_parts.append("read -r -p 'Press Enter to close...' _")
    shell_parts.append("exit \"$status\"")
    return "bash -lc " + shlex.quote("; ".join(shell_parts))


def _quote_applescript(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _terminal_batch_dir(config: Mapping[str, Any], batch_id: str) -> Path:
    launcher = _mapping(config.get("launcher"), "launcher")
    base = str(launcher.get("log_dir", "logs/openrouter_matrix"))
    return Path(base) / batch_id


def _prepare_jobs(
    config_path: Path,
    *,
    model_filter: Optional[List[str]],
    experiment_filter: Optional[List[str]],
    batch_id: str,
    cli_overrides: Mapping[str, str],
) -> List[Dict[str, Any]]:
    config = _load_json(config_path)

    models = config.get("models")
    experiments = config.get("experiments")
    if not isinstance(models, list) or not models:
        raise SystemExit("Config field 'models' must be a non-empty JSON array.")
    if not isinstance(experiments, list) or not experiments:
        raise SystemExit("Config field 'experiments' must be a non-empty JSON array.")

    selected_models = _select_values([str(model) for model in models], model_filter, "models")
    selected_experiments = _select_values(
        [str(experiment) for experiment in experiments],
        experiment_filter,
        "experiments",
    )

    copied_env = _require_copied_env(config)
    repo_root = Path(__file__).resolve().parents[2]
    batch_dir = repo_root / _terminal_batch_dir(config, batch_id)

    jobs: List[Dict[str, Any]] = []
    for experiment in selected_experiments:
        locked_keys = _preset_locked_keys(repo_root, experiment)
        for model in selected_models:
            job_name = f"{_slug(experiment)}__{_slug(model)}"
            log_file = batch_dir / f"{job_name}.log"
            python_command = _build_python_command(
                repo_root,
                config,
                model,
                experiment,
                cli_overrides,
                locked_keys,
            )
            shell_command = _build_shell_command(
                repo_root=repo_root,
                python_command=python_command,
                job_name=job_name,
                log_file=log_file,
                copied_env=copied_env,
                config=config,
            )
            jobs.append(
                {
                    "name": job_name,
                    "model": model,
                    "experiment": experiment,
                    "python_command": python_command,
                    "shell_command": shell_command,
                    "log_file": log_file,
                    "config": config,
                }
            )

    return jobs


def _print_jobs(jobs: List[Dict[str, Any]], batch_id: str) -> None:
    print(f"Batch id: {batch_id}")
    print(f"Prepared {len(jobs)} job(s).")
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}] {job['name']}")
        print(f"    model: {job['model']}")
        print(f"    experiment: {job['experiment']}")
        print(f"    log: {job['log_file']}")
        print(f"    command: {shlex.join(job['python_command'])}")


def _launch_in_terminal(jobs: List[Dict[str, Any]]) -> None:
    if sys.platform != "darwin":
        raise SystemExit("Terminal launching is currently supported only on macOS.")

    if not jobs:
        return

    config = jobs[0]["config"]
    launcher = _mapping(config.get("launcher"), "launcher")
    terminal_app = str(launcher.get("terminal_app", "Terminal"))
    pause_seconds = float(launcher.get("pause_seconds_between_launches", 0.2))

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
    parser = argparse.ArgumentParser(
        description="Preview or launch an OpenRouter model x experiment matrix."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON matrix config file.",
    )
    parser.add_argument(
        "--models",
        help="Optional comma-separated subset of models from the config.",
    )
    parser.add_argument(
        "--experiments",
        help="Optional comma-separated subset of experiments from the config.",
    )
    parser.add_argument(
        "--batch-id",
        help="Optional batch id used in log paths. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Open one macOS Terminal window per job. Without this flag the script only prints the matrix.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Hydra argument for every generated job. May be passed multiple times.",
    )
    args = parser.parse_args()

    batch_id = args.batch_id or time.strftime("%Y%m%d_%H%M%S")
    cli_overrides = _parse_set_overrides(args.set)
    jobs = _prepare_jobs(
        Path(args.config),
        model_filter=_comma_list(args.models),
        experiment_filter=_comma_list(args.experiments),
        batch_id=batch_id,
        cli_overrides=cli_overrides,
    )

    _print_jobs(jobs, batch_id=batch_id)
    if not args.launch:
        print("Dry run only. Re-run with --launch to open Terminal windows.")
        return

    _launch_in_terminal(jobs)
    print("Launch complete.")


if __name__ == "__main__":
    main()
