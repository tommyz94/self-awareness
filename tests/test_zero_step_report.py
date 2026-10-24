from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_run_json(path: Path, *, experiment_id: str, run_id: int, steps: int, completed: bool, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "completed": completed,
                "steps": steps,
                "run_setup": {"seed": seed},
            }
        )
    )


def test_zero_step_report_counts_and_catalogs_zero_step_runs(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    session_root = (
        root
        / "mujoco"
        / "e3"
        / "openrouter"
        / "qwen_qwen3.6-plus"
        / "level_2"
        / "20260425_113049"
    )
    _write_run_json(
        session_root / "run_07" / "run07.json",
        experiment_id="e3",
        run_id=7,
        steps=0,
        completed=False,
        seed=111,
    )
    _write_run_json(
        session_root / "run_08" / "run08.json",
        experiment_id="e3",
        run_id=8,
        steps=12,
        completed=False,
        seed=112,
    )

    script = ROOT / "scripts" / "dev" / "zero_step_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scanned runs: 2" in result.stdout
    assert "Zero-step failed runs: 1" in result.stdout
    assert "By date:" in result.stdout
    assert "20260425: 1/2 zero-step failed" in result.stdout
    assert "e3 | openrouter | qwen_qwen3.6-plus: 1/2 zero-step failed" in result.stdout
    assert "session=20260425_113049: 1/2 zero-step failed | runs=7" in result.stdout
    assert "date=20260425 | session=20260425_113049 | run=7 | seed=111" in result.stdout


def test_zero_step_report_latest_date_filters_to_most_recent_session(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    older = (
        root
        / "mujoco"
        / "e2"
        / "openrouter"
        / "google_gemma-4-31b-it_nitro"
        / "level_2"
        / "20260424_101010"
    )
    newer = (
        root
        / "mujoco"
        / "e3"
        / "openrouter"
        / "qwen_qwen3.6-plus"
        / "level_2"
        / "20260425_113049"
    )
    _write_run_json(
        older / "run_01" / "run01.json",
        experiment_id="e2",
        run_id=1,
        steps=0,
        completed=False,
        seed=101,
    )
    _write_run_json(
        newer / "run_07" / "run07.json",
        experiment_id="e3",
        run_id=7,
        steps=0,
        completed=False,
        seed=111,
    )

    script = ROOT / "scripts" / "dev" / "zero_step_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--latest-date"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scanned runs: 1" in result.stdout
    assert "20260425: 1/1 zero-step failed" in result.stdout
    assert "20260424" not in result.stdout
