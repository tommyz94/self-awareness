from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_run_json(
    path: Path,
    *,
    experiment_id: str,
    run_id: int,
    completed: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "completed": completed,
                "run_setup": {"seed": seed},
            }
        )
    )


def test_metric_readiness_report_marks_three_seed_blocks_as_ready(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    session_dates = ["20260501_100000", "20260502_100000", "20260503_100000"]
    seeds = [101, 202, 303]

    for experiment_id in ["e1", "e2"]:
        for session_id, seed in zip(session_dates, seeds):
            session_root = (
                root
                / "mujoco"
                / experiment_id
                / "openai"
                / "gpt-5.1"
                / "level_2"
                / session_id
            )
            for run_id in range(1, 8):
                _write_run_json(
                    session_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
                    experiment_id=experiment_id,
                    run_id=run_id,
                    completed=True,
                    seed=seed,
                )

    script = ROOT / "scripts" / "dev" / "metric_readiness_report.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--experiments",
            "e1,e2",
            "--runs-per-seed",
            "7",
            "--min-seeds",
            "3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "openai | gpt-5.1 | ready_experiments=2/2" in result.stdout
    assert "e1:21runs/3seeds/3x>=7 READY" in result.stdout
    assert "e2:21runs/3seeds/3x>=7 READY" in result.stdout
    assert "Fully ready models:" in result.stdout
    assert "openai | gpt-5.1" in result.stdout


def test_metric_readiness_report_marks_partial_seed_blocks_as_not_ready(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for session_id, seed, run_count in [
        ("20260501_100000", 101, 7),
        ("20260502_100000", 202, 7),
        ("20260503_100000", 303, 3),
    ]:
        session_root = (
            root
            / "mujoco"
            / "e1"
            / "google"
            / "gemini-2.5-pro"
            / "level_2"
            / session_id
        )
        for run_id in range(1, run_count + 1):
            _write_run_json(
                session_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
                experiment_id="e1",
                run_id=run_id,
                completed=True,
                seed=seed,
            )

    script = ROOT / "scripts" / "dev" / "metric_readiness_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--experiments", "e1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "google | gemini-2.5-pro | ready_experiments=0/1" in result.stdout
    assert "e1:17runs/3seeds/2x>=7" in result.stdout
    assert "READY" not in result.stdout


def test_metric_readiness_report_can_group_provider_agnostic_families(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    for session_id, seed in [
        ("20260501_100000", 101),
        ("20260502_100000", 202),
    ]:
        session_root = (
            root
            / "mujoco"
            / "e1"
            / "openrouter"
            / "google_gemma-4-31b-it_nitro"
            / "level_2"
            / session_id
        )
        for run_id in range(1, 8):
            _write_run_json(
                session_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
                experiment_id="e1",
                run_id=run_id,
                completed=True,
                seed=seed,
            )

    session_root = (
        root
        / "mujoco"
        / "e1"
        / "ollama"
        / "gemma4_31b"
        / "level_2"
        / "20260503_100000"
    )
    for run_id in range(1, 8):
        _write_run_json(
            session_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
            experiment_id="e1",
            run_id=run_id,
            completed=True,
            seed=303,
        )

    script = ROOT / "scripts" / "dev" / "metric_readiness_report.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--experiments",
            "e1",
            "--group-by",
            "family",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "gemma-4-31b | providers=ollama,openrouter | exact_variants=2 | ready_experiments=1/1" in result.stdout
    assert "e1:21runs/3seeds/3x>=7 READY" in result.stdout
    assert "gemma-4-31b | providers=ollama,openrouter" in result.stdout
