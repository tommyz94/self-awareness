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


def _write_summary(path: Path, *, experiment_id: str, seed: int, num_runs: int) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "num_runs": num_runs,
            }
        )
    )


def test_build_metric_manifest_prefers_minimal_trimming_and_summary_seed(tmp_path: Path) -> None:
    root = tmp_path / "final_outputs"

    # A 20-run seed that should be dropped when cleaner 7-run seeds exist.
    long_seed_root = (
        root
        / "mujoco"
        / "e1"
        / "openrouter"
        / "google_gemma-4-26b-a4b-it_nitro"
        / "level_2"
        / "20260501_100000"
    )
    for run_id in range(1, 21):
        _write_run_json(
            long_seed_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
            experiment_id="e1",
            run_id=run_id,
            completed=True,
            seed=101,
        )

    for session_id, seed in [
        ("20260502_100000", 202),
        ("20260503_100000", 303),
    ]:
        session_root = (
            root
            / "mujoco"
            / "e1"
            / "ollama"
            / "gemma4_26b"
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

    mixed_session_root = (
        root
        / "mujoco"
        / "e1"
        / "openrouter"
        / "google_gemma-4-26b-a4b-it_nitro"
        / "level_2"
        / "20260504_100000"
    )
    mixed_seeds = [999, 555, 555, 555, 555, 555, 555]
    for run_id, raw_seed in enumerate(mixed_seeds, start=1):
        _write_run_json(
            mixed_session_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json",
            experiment_id="e1",
            run_id=run_id,
            completed=True,
            seed=raw_seed,
        )
    _write_summary(
        mixed_session_root / "experiment_summary.json",
        experiment_id="e1",
        seed=555,
        num_runs=7,
    )

    script = ROOT / "scripts" / "dev" / "build_metric_manifest.py"
    json_out = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--experiments",
            "e1",
            "--seed-selection",
            "minimal-trimming",
            "--json-out",
            str(json_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "gemma-4-26b | providers=ollama,openrouter" in result.stdout
    assert "selected=21 runs (3 seeds x 7)" in result.stdout
    assert "dropped valid seeds: 101=20" in result.stdout

    report = json.loads(json_out.read_text())
    model_row = report["models"][0]
    experiment_row = model_row["experiments"][0]

    kept_seeds = [seed_row["seed"] for seed_row in experiment_row["selected_seeds"]]
    assert kept_seeds == ["555", "303", "202"]

    dropped_valid_seeds = experiment_row["dropped_valid_seeds"]
    assert dropped_valid_seeds == [{"seed": "101", "available_runs": 20}]

    manifest_rows = report["manifest_rows"]
    kept_555 = [row for row in manifest_rows if row["seed"] == "555"]
    assert len(kept_555) == 7
    assert [row["run_id"] for row in kept_555] == [1, 2, 3, 4, 5, 6, 7]


def test_build_metric_manifest_uses_seed_order_and_first_runs(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    # Old seed with one empty run among the first seven.
    old_seed_root = (
        root
        / "mujoco"
        / "e3"
        / "openrouter"
        / "google_gemma-4-31b-it_nitro"
        / "level_2"
        / "20260501_100000"
    )
    for run_id in range(1, 11):
        payload = {
            "experiment_id": "e3",
            "run_id": run_id,
            "completed": True,
            "run_setup": {"seed": 111},
            "step_records": [] if run_id == 4 else [{"action": "w"}],
        }
        path = old_seed_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    # More recent seed, all informative.
    recent_seed_root = (
        root
        / "mujoco"
        / "e3"
        / "openrouter"
        / "google_gemma-4-31b-it_nitro"
        / "level_2"
        / "20260502_100000"
    )
    for run_id in range(1, 11):
        payload = {
            "experiment_id": "e3",
            "run_id": run_id,
            "completed": True,
            "run_setup": {"seed": 222},
            "step_records": [{"action": "w"}],
        }
        path = recent_seed_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    # Another older but fully informative seed.
    other_seed_root = (
        root
        / "mujoco"
        / "e3"
        / "ollama"
        / "gemma4_31b"
        / "level_2"
        / "20260430_100000"
    )
    for run_id in range(1, 8):
        payload = {
            "experiment_id": "e3",
            "run_id": run_id,
            "completed": True,
            "run_setup": {"seed": 333},
            "step_records": [{"action": "w"}],
        }
        path = other_seed_root / f"run_{run_id:02d}" / f"run{run_id:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    json_out = tmp_path / "manifest.json"
    script = ROOT / "scripts" / "dev" / "build_metric_manifest.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--experiments",
            "e3",
            "--json-out",
            str(json_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(json_out.read_text())
    experiment_row = report["models"][0]["experiments"][0]
    kept_seeds = [seed_row["seed"] for seed_row in experiment_row["selected_seeds"]]
    assert kept_seeds == ["111", "222", "333"]
    assert experiment_row["selected_seeds"][0]["usable_kept_runs"] == 6
    assert experiment_row["selected_seeds"][1]["usable_kept_runs"] == 7
    assert experiment_row["selected_seeds"][2]["usable_kept_runs"] == 7
    assert experiment_row["selected_seeds"][0]["kept_run_ids"] == [1, 2, 3, 4, 5, 6, 7]
