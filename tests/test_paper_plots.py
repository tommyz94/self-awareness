"""Plotting must not silently drop models or fabricate missing statistics."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="Install requirements-analysis.txt for plot checks.")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_plots", ROOT / "notebooks/plot_paper_results.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


@pytest.fixture
def aggregates(tmp_path):
    rows = []
    for model in plots.MODEL_NAMES:
        for exp in plots.EXPERIMENTS:
            row = {"model_name": model, "experiment_id": exp,
                   "total_runs": 7 if model == "cosmos-reason2-8b" else 21}
            row.update({f"{metric}_mean": 50 if metric == "TTD" else 0.5 for metric in plots.METRICS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    path = tmp_path / "aggregates.csv"
    frame.to_csv(path, index=False)
    return frame, path


def test_all_nine_models_are_loaded_without_modifying_input(aggregates):
    frame, path = aggregates
    original = path.read_bytes()
    result = plots.load_results([path])
    assert len(result) == 45
    assert set(result.model_key) == set(plots.MODEL_NAMES)
    np.testing.assert_array_equal(result.TSA_mean, frame.TSA_mean)
    assert path.read_bytes() == original


def test_missing_cosmos_cannot_silently_produce_eight_model_figures(aggregates):
    frame, path = aggregates
    frame[frame.model_name != "cosmos-reason2-8b"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing paper model/condition"):
        plots.load_results([path])


def test_duplicate_provider_alias_cannot_double_count_a_model(aggregates):
    frame, path = aggregates
    duplicate = frame.iloc[[0]].copy()
    duplicate["model_name"] = "anthropic/claude-sonnet-4.6"
    pd.concat([frame, duplicate]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate model/condition"):
        plots.load_results([path])


def test_precise_aggregate_overrides_rounded_overlay(aggregates, tmp_path):
    frame, path = aggregates
    overlay = tmp_path / "rounded.json"
    rounded = frame[frame.model_name == "cosmos-reason2-8b"].copy()
    rounded["TSA_mean"] = 0.14
    overlay.write_text(json.dumps({"source": "test reported values", "rows": rounded.to_dict("records")}))
    result = plots.load_results([path], overlay)
    assert len(result) == 45
    assert (result[result.model_key == "cosmos-reason2-8b"].TSA_mean == 0.5).all()
    assert set(result.source_kind) == {"aggregate_csv"}
    assert len(result.attrs["provenance"]) == 1


def test_undefined_metric_and_missing_auxiliary_data_are_not_zero(aggregates):
    frame, path = aggregates
    mask = (frame.model_name == "cosmos-reason2-8b") & (frame.experiment_id == "e3")
    frame.loc[mask, "MTATO_mean"] = np.nan
    frame.to_csv(path, index=False)
    result = plots.load_results([path])
    row = result.loc[mask].iloc[0]
    assert pd.isna(row.MTATO_mean)
    assert "ident_confusion_counts_json" not in result.columns
    assert "TSA-COMPLETE_mean" not in result.columns


def test_original_plot_transforms_and_existing_model_styles_are_preserved():
    original = json.loads((ROOT / "notebooks/archive/results_analysis_legacy_source.ipynb").read_text())
    current = json.loads((ROOT / "notebooks/results_analysis.ipynb").read_text())
    for index in plots.PLOTS.values():
        old = ast.parse("".join(original["cells"][index]["source"]))
        new = ast.parse("".join(current["cells"][index]["source"]))
        old_functions = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
        for name, body in old_functions.items():
            if name != "e5_outcome_decomposition":  # Only Cosmos gets a missing-count guard.
                assert new_functions[name] == body
        def styles(tree):
            values = {}
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                name = getattr(node.targets[0], "id", "")
                if name in {"COLORS", "MARKERS", "MODEL_LABELS", "PAPER_MODEL_NAMES", "PAPER_MODEL_ORDER"}:
                    values[name] = ast.literal_eval(node.value)
                elif name == "MODEL_STYLES":
                    values[name] = ast.literal_eval(node.value.args[0])
            return values
        updated = styles(new)
        for name, value in styles(old).items():
            if isinstance(value, dict):
                assert {k: updated[name][k] for k in value} == value
            else:
                assert updated[name][:len(value)] == value
    for index in set(range(len(original["cells"]))) - {1, *plots.PLOTS.values()}:
        assert current["cells"][index]["source"] == original["cells"][index]["source"]


def test_other_cosmos_sizes_are_not_relabelled_as_the_eight_billion_model():
    assert plots.canonical_model("Cosmos-Reason2-2B") == "Cosmos-Reason2-2B"


def test_provider_ids_are_canonicalized_for_notebook_row_selection(aggregates):
    frame, path = aggregates
    frame.loc[frame.model_name == "cosmos-reason2-8b", "model_name"] = "mradermacher/Cosmos-Reason2-8B-GGUF:Q4_K_M"
    frame.to_csv(path, index=False)
    result = plots.read_results(path)
    assert set(result.model_name) == set(plots.MODEL_NAMES)


def test_source_notebooks_have_no_embedded_results_or_machine_paths():
    for name in ("results_analysis.ipynb", "archive/results_analysis_legacy_source.ipynb"):
        notebook = json.loads((ROOT / "notebooks" / name).read_text())
        for cell in notebook["cells"]:
            assert not cell.get("outputs")
            assert cell.get("execution_count") is None
            assert "/Users/" not in "".join(cell["source"])


def test_all_five_original_plot_cells_render_without_real_data(aggregates, tmp_path):
    pytest.importorskip("seaborn", reason="Install requirements-analysis.txt to render figures.")
    frame, path = aggregates
    frame["TSA-COMPLETE_mean"] = 0.75
    frame["ident_confusion_counts_json"] = frame.total_runs.map(
        lambda n: json.dumps({"red": {"red": int(n)}}))
    frame.loc[frame.model_name == "cosmos-reason2-8b",
              ["TSA-COMPLETE_mean", "ident_confusion_counts_json"]] = np.nan
    frame.to_csv(path, index=False)
    outputs = plots.render_all(plots.load_results([path]), tmp_path / "figures")
    assert len(outputs) == 15
    assert all(p.is_file() and p.stat().st_size > 0 for p in outputs)
    manifest = json.loads((tmp_path / "figures/figure_manifest.json").read_text())
    assert len(manifest["rows"]) == 45
