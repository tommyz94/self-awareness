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


def test_run_inventory_report_counts_models_and_seeds(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    openai_session = (
        root
        / "mujoco_openai_gpt51"
        / "mujoco"
        / "e1"
        / "openai"
        / "gpt-5.1"
        / "level_2"
        / "20260503_120000"
    )
    _write_run_json(
        openai_session / "run_01" / "run01.json",
        experiment_id="e1",
        run_id=1,
        completed=True,
        seed=111,
    )
    _write_run_json(
        openai_session / "run_02" / "run02.json",
        experiment_id="e1",
        run_id=2,
        completed=False,
        seed=111,
    )

    sonnet_session = (
        root
        / "mujoco_openrouter_claude_sonnet46_anthropic_only"
        / "mujoco"
        / "e3"
        / "openrouter"
        / "anthropic_claude-sonnet-4.6"
        / "level_2"
        / "20260504_090000"
    )
    _write_run_json(
        sonnet_session / "run_01" / "run01.json",
        experiment_id="e3",
        run_id=1,
        completed=True,
        seed=222,
    )

    script = ROOT / "scripts" / "dev" / "run_inventory_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scanned runs: 3" in result.stdout
    assert "Model buckets: 2" in result.stdout
    assert "openai | gpt-5.1: runs=2 | completed=1 | incomplete=1 | sessions=1 | seeds=1" in result.stdout
    assert "experiments: e1=2" in result.stdout
    assert "111: runs=2 | sessions=1 | experiments=e1=2" in result.stdout
    assert (
        "openrouter | anthropic_claude-sonnet-4.6: runs=1 | completed=1 | incomplete=0 | sessions=1 | seeds=1"
        in result.stdout
    )
    assert "222: runs=1 | sessions=1 | experiments=e3=1" in result.stdout


def test_run_inventory_report_latest_date_filter(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    older = (
        root
        / "mujoco"
        / "e2"
        / "google"
        / "gemini-2.5-flash"
        / "level_2"
        / "20260501_080000"
    )
    newer = (
        root
        / "mujoco"
        / "e2"
        / "google"
        / "gemini-2.5-pro"
        / "level_2"
        / "20260503_100000"
    )
    _write_run_json(
        older / "run_01" / "run01.json",
        experiment_id="e2",
        run_id=1,
        completed=True,
        seed=10,
    )
    _write_run_json(
        newer / "run_01" / "run01.json",
        experiment_id="e2",
        run_id=1,
        completed=True,
        seed=20,
    )

    script = ROOT / "scripts" / "dev" / "run_inventory_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--latest-date"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scanned runs: 1" in result.stdout
    assert "google | gemini-2.5-pro: runs=1" in result.stdout
    assert "gemini-2.5-flash" not in result.stdout
