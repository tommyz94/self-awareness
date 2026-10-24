"""Plotting must not silently drop models or fabricate missing statistics."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

pd = pytest.importorskip("pandas", reason="Install requirements.txt for plot checks.")
ROOT = Path(__file__).resolve().parents[1]
from notebooks import plot_paper_results as plots


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


def test_canonical_paper_aggregates_preserve_figure_values_and_condition_labels(aggregates, tmp_path):
    frame, path = aggregates
    expected = plots.load_results([path])
    canonical = frame.copy()
    canonical["experiment_id"] = canonical["experiment_id"].map(
        {"e1": "e1", "e2": "e2", "e3": "e3", "e5": "e4", "e7": "e5"})
    canonical["experiment_id_scheme"] = "paper-v1"
    target = tmp_path / "paper_metrics.csv"
    canonical.to_csv(target, index=False)
    actual = plots.load_results([target])
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_flags=False)


def test_subset_does_not_require_or_invent_cosmos(aggregates):
    frame, path = aggregates
    frame = frame[frame.model_name != "cosmos-reason2-8b"]
    frame.to_csv(path, index=False)
    result = plots.load_results([path])
    assert len(result) == 40
    assert "cosmos-reason2-8b" not in plots.model_styles(result)


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
    assert result.ident_confusion_counts_json.isna().all()
    assert result["TSA-COMPLETE_mean"].isna().all()


def test_existing_paper_colors_and_markers_are_preserved():
    original = json.loads((ROOT / "notebooks/archive/results_analysis_legacy_source.ipynb").read_text())
    tree = ast.parse("".join(original["cells"][9]["source"]))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "MODEL_STYLES")
    for model, style in ast.literal_eval(node.value.args[0]):
        assert {key: plots.MODEL_STYLES[model][key] for key in style} == style


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
    pytest.importorskip("seaborn", reason="Install requirements.txt to render figures.")
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


@pytest.mark.parametrize("condition", ["e1", "e7"])
def test_single_unfamiliar_model_renders_without_registration(aggregates, condition):
    frame, path = aggregates
    frame = frame[frame.experiment_id == condition].iloc[:1].copy()
    frame["model_name"] = "new-provider/new-vision-model"
    frame.to_csv(path, index=False)
    data = plots.load_results([path])
    for name, (plot, _, _) in plots.PLOTS.items():
        with plots.plt.rc_context():
            fig = plot(data)
            if name == "e5_identification_outcomes" and condition != "e7":
                assert fig is None
                continue
            assert fig is not None
            fig.canvas.draw()
            labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
            labels += [t.get_text() for legend in fig.legends for t in legend.get_texts()]
            if name != "e5_identification_outcomes":
                assert "new-provider/new-vision-model" in labels
            if name in {"e5_identification_outcomes", "Tsa_Complete-Tsa"}:
                assert "N/A" in [t.get_text() for ax in fig.axes for t in ax.texts]
            plots.plt.close(fig)


def test_unknown_style_is_stable_when_input_order_or_companions_change(aggregates):
    frame, _ = aggregates
    own = frame.iloc[:1].assign(model_name="lab/model")
    first = plots.model_styles(own)["lab/model"]
    more = pd.concat([frame, own, own.assign(model_name="another/model")])
    assert plots.model_styles(more.sample(frac=1, random_state=123))["lab/model"] == first


def test_partial_conditions_and_more_than_nine_models_are_visible(aggregates):
    frame, path = aggregates
    # These new models have no E1 baseline; radar ordering must still include them.
    extra = frame.iloc[:1].assign(model_name="new-A", experiment_id="e3")
    extra2 = extra.assign(model_name="new-B", experiment_id="e7")
    pd.concat([frame, extra, extra2]).to_csv(path, index=False)
    data = plots.load_results([path])
    original = data.copy(deep=True)
    for plot, _, _ in plots.PLOTS.values():
        with plots.plt.rc_context():
            fig = plot(data)
            fig.canvas.draw()
            if plot == plots.plot_radar:
                assert len(fig.axes[0].lines) == 11
                assert "new-A" in [t.get_text() for t in fig.legends[0].texts]
            if plot == plots.plot_heatmap:
                assert len(fig.axes[0].get_yticklabels()) == 11
            plots.plt.close(fig)
    pd.testing.assert_frame_equal(data, original)


def test_missing_values_are_not_converted_to_zero_or_midpoint(aggregates):
    frame, path = aggregates
    frame = frame[frame.experiment_id == "e1"].iloc[:2].copy()
    frame.loc[frame.index[0], ["TTD_mean", "MTATO_mean"]] = np.nan
    frame.to_csv(path, index=False)
    data = plots.load_results([path])
    with plots.plt.rc_context():
        fig = plots.plot_radar(data)
        line = next(line for line in fig.axes[0].lines
                    if line.get_label() == plots.MODEL_NAMES[frame.iloc[0].model_name])
        assert np.isnan(line.get_ydata()[1])  # TTD remains undefined even with constant valid TTD.
        assert np.isnan(line.get_ydata()[3])
        plots.plt.close(fig)
        fig = plots.plot_heatmap(data)
        assert sum(t.get_text() == "N/A" for t in fig.axes[0].texts) == 2
        plots.plt.close(fig)


@pytest.mark.parametrize("value", ["bad-condition", None])
def test_invalid_conditions_are_rejected_instead_of_silently_dropped(aggregates, value):
    frame, path = aggregates
    frame.loc[0, "experiment_id"] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unsupported condition"):
        plots.load_results([path])


def test_notebook_uses_same_plot_functions_and_executes_with_new_model(aggregates):
    frame, path = aggregates
    frame.iloc[:1].assign(model_name="notebook-model").to_csv(path, index=False)
    nb = json.loads((ROOT / "notebooks/results_analysis.ipynb").read_text())
    env = {"display": lambda fig: fig.canvas.draw()}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace('ROOT / "reports/evaluation/paper_metrics.csv"', repr(str(path)))
        exec(compile(source, "results_analysis.ipynb", "exec"), env)
    assert env["df"].model_name.tolist() == ["notebook-model"]
