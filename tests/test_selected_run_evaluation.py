"""Selection integrity and historical metric semantics, using synthetic runs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/evaluate_selected_runs.py"
spec = importlib.util.spec_from_file_location("selected_run_evaluation", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def step(action="w", color="unknown", mirror=False, selected=None):
    return {"action": action, "identification": {"my_color": color},
            "mirror_visible": mirror, "selected_cube": selected}


def write_run(root, experiment, number, steps):
    path = root / experiment / "provider" / "model" / f"run_{number}" / f"run{number:02}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"agent_color": "red", "experiment_id": experiment,
                                "run_setup": {"seed": 123}, "step_records": steps}))
    return path


def manifest_for(tmp_path, paths):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_rows": [
        {"path": str(path.relative_to(tmp_path)),
         "experiment_id": next(p.name for p in path.parents if p.name in runner.EXPERIMENTS)}
        for path in paths
    ]}))
    return manifest


def test_cli_selects_only_manifest_and_explicit_additional_cohort(tmp_path):
    pytest.importorskip("pandas", reason="Install requirements.txt for calculator tests")
    primary = write_run(tmp_path / "raw", "e1", 1,
                        [step(), step("done", "blue", selected="red")])
    # This neighboring episode must not enter the selection.
    write_run(tmp_path / "raw", "e1", 2, [step("done", selected="blue")])
    additional = write_run(tmp_path / "cosmos", "e7", 1,
                           [step(mirror=True), step(color="red")])
    empty = write_run(tmp_path / "cosmos", "e7", 2, [])
    manifest = manifest_for(tmp_path, [primary])
    original = {p: p.read_bytes() for p in [primary, additional, empty, manifest, runner.CALCULATOR]}
    output = tmp_path / "new-report"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--manifest", str(manifest),
        "--data-root", str(tmp_path), "--additional-root", str(tmp_path / "cosmos"),
        "--out-dir", str(output),
    ], cwd=tmp_path, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["selected_runs"] == 3
    assert summary["cohort_counts"] == {"primary": 1, "additional": 2}
    assert summary["zero_step_runs"] == 1 and summary["input_integrity_verified"]
    assert all(p.read_bytes() == content for p, content in original.items())
    with (output / "metrics.csv").open() as stream:
        rows = {r["experiment_id"]: r for r in csv.DictReader(stream)}
    cube = rows["e1"]
    assert [float(cube[f"{m}_mean"]) for m in runner.CORE_METRICS] == [1, 2, 0, 0, 0, 1]
    attribution = rows["e7"]
    assert float(attribution["TSA_mean"]) == 0
    assert float(attribution["CAAL_mean"]) == 0.5
    assert attribution["MTATO_mean"] == "" and attribution["TSA-COMPLETE_mean"] == ""
    assert float(attribution["MTATO_se"]) == 0  # Original calculator output is preserved.
    assert summary["calculator_sha256"] == hashlib.sha256(original[runner.CALCULATOR]).hexdigest()
    selected = json.loads((output / "selected_runs.json").read_text())
    assert {r["path"] for r in selected} == {str(p) for p in [primary, additional, empty]}


def test_mtato_denominator_and_standard_error_remain_historical(tmp_path):
    pytest.importorskip("pandas", reason="Install requirements.txt for calculator tests")
    good = write_run(tmp_path / "raw", "e1", 1,
                     [step(color="red", mirror=True), step("done", "red", selected="red")])
    bad = write_run(tmp_path / "raw", "e1", 2, [step("done", selected="blue")])
    manifest = manifest_for(tmp_path, [good, bad])
    output = tmp_path / "report"
    runner.evaluate(manifest, data_root=tmp_path, out_dir=output)
    with (output / "metrics.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["TSA_mean"]) == 0.5 and float(row["TSA_se"]) == 0.5
    assert float(row["MCR_mean"]) == 0.5
    assert float(row["MTATO_mean"]) == 1 and float(row["MTATO_se"]) == 0


def test_dry_run_needs_no_pandas_and_writes_nothing(tmp_path, monkeypatch):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    monkeypatch.setitem(sys.modules, "pandas", None)
    output = tmp_path / "unused-output"
    result = runner.evaluate(manifest, data_root=tmp_path, out_dir=output, dry_run=True)
    assert result["selected_runs"] == 1 and result["dry_run"]
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "metrics.csv"
    sentinel.write_text("original results")
    with pytest.raises(ValueError, match="already exists"):
        runner.evaluate(manifest, data_root=tmp_path, out_dir=output)
    assert sentinel.read_text() == "original results"


def test_output_cannot_be_created_inside_raw_directories(tmp_path):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    with pytest.raises(ValueError, match="outside the original"):
        runner.evaluate(manifest, data_root=tmp_path, out_dir=run.parent / "report")


def test_duplicate_paths_across_cohorts_are_rejected(tmp_path):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    with pytest.raises(ValueError, match="Duplicate selected run"):
        runner.evaluate(manifest, data_root=tmp_path, additional_root=tmp_path / "raw", dry_run=True)


@pytest.mark.parametrize("invalid", ["invalid JSON", json.dumps({"agent_color": "red", "step_records": None})])
def test_bad_runs_fail_instead_of_silently_disappearing(tmp_path, invalid):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    run.write_text(invalid)
    with pytest.raises(ValueError):
        runner.evaluate(manifest, data_root=tmp_path, out_dir=tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_optional_manifest_hash_is_verified(tmp_path):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    data = json.loads(manifest.read_text())
    data["manifest_rows"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.evaluate(manifest, data_root=tmp_path, dry_run=True)


def test_missing_run_or_empty_additional_root_fails(tmp_path):
    run = write_run(tmp_path / "raw", "e1", 1, [])
    manifest = manifest_for(tmp_path, [run])
    additional = tmp_path / "empty"
    additional.mkdir()
    with pytest.raises(ValueError, match="No runNN.json"):
        runner.evaluate(manifest, data_root=tmp_path, additional_root=additional, dry_run=True)
    run.unlink()
    with pytest.raises(FileNotFoundError):
        runner.evaluate(manifest, data_root=tmp_path, dry_run=True)


@pytest.mark.parametrize("document", [[], {"manifest_rows": [None]},
                                      {"manifest_rows": [{"path": 123, "experiment_id": "e1"}]}])
def test_manifest_shape_errors_are_explicit(tmp_path, document):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Manifest|manifest row"):
        runner.evaluate(manifest, data_root=tmp_path, dry_run=True)


def test_calculator_cannot_silently_omit_selected_models(tmp_path):
    pytest.importorskip("pandas", reason="Install requirements.txt for calculator tests")
    good = write_run(tmp_path / "raw", "e1", 1, [])
    skipped = write_run(tmp_path / "other", "e1", 1, [])
    skipped_model = skipped.parents[1].with_name("moonshotai_kimi-k2.6_nitro")
    skipped.parents[1].rename(skipped_model)
    skipped = skipped_model / skipped.parent.name / skipped.name
    manifest = manifest_for(tmp_path, [good, skipped])
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="every selected run"):
        runner.evaluate(manifest, data_root=tmp_path, out_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("paper_id,expected_data_id", [("e4", "e5"), ("e5", "e7")])
def test_new_paper_runs_use_correct_task_scoring_without_rewriting_raw(tmp_path, paper_id, expected_data_id):
    pytest.importorskip("pandas")
    run = write_run(tmp_path / "raw", paper_id, 1,
                    [step(), step("done", "red", selected="red" if paper_id == "e4" else None)])
    data = json.loads(run.read_text())
    data["run_context"] = {"experiment_id_scheme": "paper-v1", "paper_experiment_id": paper_id}
    data["paper_experiment_id"] = paper_id
    run.write_text(json.dumps(data))
    original = run.read_bytes()
    # A manifest identifies the source directory, not the calculator's old ID.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_rows": [{"path": str(run), "experiment_id": paper_id}]}))
    output = tmp_path / "report"
    runner.evaluate(manifest, data_root=tmp_path, out_dir=output)
    assert run.read_bytes() == original
    with (output / "metrics.csv").open() as stream:
        source_row = next(csv.DictReader(stream))
    with (output / "paper_metrics.csv").open() as stream:
        public_row = next(csv.DictReader(stream))
    assert source_row["experiment_id"] == expected_data_id
    assert public_row["experiment_id"] == paper_id
    assert public_row["experiment_id_scheme"] == "paper-v1"
    assert float(public_row["TSA_mean"]) == 1
    assert {k: v for k, v in source_row.items() if k != "experiment_id"} == {
        k: v for k, v in public_row.items() if k not in {"experiment_id", "calculator_experiment_id", "experiment_id_scheme"}}


def test_unversioned_e4_cannot_be_silently_scored_as_paper_e4(tmp_path):
    run = write_run(tmp_path / "raw", "e4", 1, [])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_rows": [{"path": str(run), "experiment_id": "e4"}]}))
    with pytest.raises(ValueError, match="paper-v1"):
        runner.evaluate(manifest, data_root=tmp_path, dry_run=True)


def test_old_and_new_e5_logs_are_not_confused_when_evaluated_together(tmp_path):
    pytest.importorskip("pandas")
    old = write_run(tmp_path / "original", "e5", 1, [step("done", "red", selected="blue")])
    new = write_run(tmp_path / "current", "e5", 1, [step("done", "red")])
    data = json.loads(new.read_text())
    data["run_context"] = {"experiment_id_scheme": "paper-v1"}
    new.write_text(json.dumps(data))
    manifest = manifest_for(tmp_path, [old, new])
    output = tmp_path / "report"
    runner.evaluate(manifest, data_root=tmp_path, out_dir=output)
    with (output / "paper_metrics.csv").open() as stream:
        rows = {r["experiment_id"]: r for r in csv.DictReader(stream)}
    assert float(rows["e4"]["TSA_mean"]) == 0  # Old e5 is cube selection.
    assert float(rows["e5"]["TSA_mean"]) == 1  # New E5 is self-attribution.
