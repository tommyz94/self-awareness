from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_e4_run_json(
    path: Path,
    *,
    run_id: int,
    variant: str,
    expected_answer: str,
    actual_answer: str,
    success: bool,
    completed: bool,
    steps: int,
    used_lateral_action_before_done: bool,
    lateral_action_count_before_done: int,
    forward_action_count_before_done: int,
    step_records: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "e4",
                "task_id": "front_identity_task",
                "run_id": run_id,
                "completed": completed,
                "steps": steps,
                "task_output": {
                    "success": success,
                    "front_identity_variant": variant,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "front_identity": {
                        "is_self": actual_answer,
                        "confidence": "high" if success else "low",
                        "evidence_source": "test",
                        "reason": "test",
                    },
                    "used_lateral_action_before_done": used_lateral_action_before_done,
                    "lateral_action_count_before_done": lateral_action_count_before_done,
                    "forward_action_count_before_done": forward_action_count_before_done,
                },
                "step_records": step_records,
            }
        )
    )


def test_e4_metrics_report_summarizes_core_metrics(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    session_root = (
        root
        / "mujoco"
        / "e4"
        / "ollama"
        / "gemma4_26b"
        / "level_2"
        / "20260429_000901"
    )

    _write_e4_run_json(
        session_root / "run_01" / "run01.json",
        run_id=1,
        variant="self",
        expected_answer="yes",
        actual_answer="yes",
        success=True,
        completed=True,
        steps=3,
        used_lateral_action_before_done=True,
        lateral_action_count_before_done=1,
        forward_action_count_before_done=1,
        step_records=[
            {
                "step": 1,
                "action": "w",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
            {
                "step": 2,
                "action": "a",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": -2.0}},
            },
            {
                "step": 3,
                "action": "done",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "yes"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    _write_e4_run_json(
        session_root / "run_02" / "run02.json",
        run_id=2,
        variant="other",
        expected_answer="no",
        actual_answer="no",
        success=True,
        completed=True,
        steps=2,
        used_lateral_action_before_done=False,
        lateral_action_count_before_done=0,
        forward_action_count_before_done=1,
        step_records=[
            {
                "step": 1,
                "action": "w",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
            {
                "step": 2,
                "action": "done",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "no"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    script = ROOT / "scripts" / "dev" / "e4_metrics_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scanned runs: 2" in result.stdout
    assert "Model: gemma4_26b" in result.stdout
    assert "Accuracy: 1.00 (self=1.00, other=1.00, balanced=1.00)" in result.stdout
    assert "Done rate: 1.00 | Mean steps to done: 2.50" in result.stdout
    assert "Effective lateral probe rate: 0.50 | Mean lateral distance: 1.00" in result.stdout
    assert "Accuracy with lateral: 1.00 | without lateral: 1.00" in result.stdout
    assert "Premature commit rate: 0.50" in result.stdout
    assert "Self-evidence seen rate: self=1.00 | other=0.00" in result.stdout


def test_e4_metrics_report_latest_per_model_filters_old_sessions(tmp_path: Path) -> None:
    root = tmp_path / "outputs"

    older = (
        root
        / "mujoco"
        / "e4"
        / "ollama"
        / "gemma4_26b"
        / "level_2"
        / "20260428_212240"
    )
    newer = (
        root
        / "mujoco"
        / "e4"
        / "ollama"
        / "gemma4_26b"
        / "level_2"
        / "20260429_000901"
    )

    _write_e4_run_json(
        older / "run_01" / "run01.json",
        run_id=1,
        variant="self",
        expected_answer="yes",
        actual_answer="no",
        success=False,
        completed=True,
        steps=2,
        used_lateral_action_before_done=False,
        lateral_action_count_before_done=0,
        forward_action_count_before_done=1,
        step_records=[
            {
                "step": 1,
                "action": "w",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
            {
                "step": 2,
                "action": "done",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "no"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    _write_e4_run_json(
        newer / "run_01" / "run01.json",
        run_id=1,
        variant="self",
        expected_answer="yes",
        actual_answer="yes",
        success=True,
        completed=True,
        steps=2,
        used_lateral_action_before_done=False,
        lateral_action_count_before_done=0,
        forward_action_count_before_done=1,
        step_records=[
            {
                "step": 1,
                "action": "w",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
            {
                "step": 2,
                "action": "done",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "yes"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    script = ROOT / "scripts" / "dev" / "e4_metrics_report.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--latest-per-model", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Latest sessions per model: 1" in result.stdout
    assert "Sessions: 20260429_000901" in result.stdout
    assert "20260428_212240" not in result.stdout
    assert "Accuracy: 1.00 (self=1.00, other=N/A, balanced=N/A)" in result.stdout


def test_e4_metrics_report_writes_excel_style_table(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    older = (
        root
        / "mujoco"
        / "e4"
        / "ollama"
        / "gemma4_26b"
        / "level_2"
        / "20260428_212240"
    )
    newer = (
        root
        / "mujoco"
        / "e4"
        / "ollama"
        / "gemma4_26b"
        / "level_2"
        / "20260429_000901"
    )

    _write_e4_run_json(
        older / "run_01" / "run01.json",
        run_id=1,
        variant="self",
        expected_answer="yes",
        actual_answer="yes",
        success=True,
        completed=True,
        steps=4,
        used_lateral_action_before_done=True,
        lateral_action_count_before_done=1,
        forward_action_count_before_done=1,
        step_records=[
            {
                "step": 1,
                "action": "a",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": -2.0}},
            },
            {
                "step": 2,
                "action": "done",
                "ego_reflection_visible": True,
                "task_payload": {"front_identity": {"is_self": "yes"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    _write_e4_run_json(
        newer / "run_01" / "run01.json",
        run_id=1,
        variant="self",
        expected_answer="yes",
        actual_answer="no",
        success=False,
        completed=True,
        steps=8,
        used_lateral_action_before_done=False,
        lateral_action_count_before_done=0,
        forward_action_count_before_done=3,
        step_records=[
            {
                "step": 1,
                "action": "w",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "uncertain"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
            {
                "step": 2,
                "action": "done",
                "ego_reflection_visible": False,
                "task_payload": {"front_identity": {"is_self": "no"}},
                "step_metadata": {"movement_delta": {"lateral_delta": 0.0}},
            },
        ],
    )

    script = ROOT / "scripts" / "dev" / "e4_metrics_report.py"
    table_path = tmp_path / "reports" / "e4_metrics.tsv"
    subprocess.run(
        [sys.executable, str(script), str(root), "--table-out", str(table_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = table_path.read_text().splitlines()
    assert lines[0].startswith("experiment_id\tmodel_name\tFIA_mean\tFIA_std\tFIA_se")
    assert lines[1].startswith("e4\tgemma4_26b\t0.5\t0.7071067812\t0.5")
    assert "\tTTD_mean\t" in lines[0]
    assert "\tELPR_mean\t" in lines[0]
    assert "\tUPR_mean\t" in lines[0]
