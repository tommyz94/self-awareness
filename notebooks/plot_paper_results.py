"""Paper figures from existing aggregate results; never evaluates raw episodes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "final_outputs/reports/result_trimmed.csv"
DEFAULT_REPORTED = ROOT / "notebooks/plot_sources/cosmos_reported_metrics.json"
DEFAULT_OUT = ROOT / "notebooks/paper_assets/all_models"
EXPERIMENTS = ("e1", "e2", "e3", "e5", "e7")
MODEL_NAMES = {
    "anthropic_claude-sonnet-4.6": "Claude Sonnet 4.6",
    "gpt-5.1": "GPT-5.1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen_qwen3.6-plus": "Qwen 3.6 Plus",
    "gemma4_31b": "Gemma 4 31B",
    "gemma4_26b": "Gemma 4 26B",
    "ministral-3_14b": "Ministral 3 14B",
    "cosmos-reason2-8b": "Cosmos-Reason2 8B",
}
METRICS = ("TSA", "TTD", "MCR", "MTATO", "CAAL", "CONFABULATION")


def canonical_model(value: str) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"cosmos-reason2-8b", "mradermacher/cosmos-reason2-8b-gguf:q4_k_m",
                 "mradermacher_cosmos-reason2-8b-gguf_q4_k_m"}:
        return "cosmos-reason2-8b"
    aliases = {
        "anthropic/claude-sonnet-4.6": "anthropic_claude-sonnet-4.6",
        "qwen/qwen3.6-plus": "qwen_qwen3.6-plus",
        "qwen3_6_plus": "qwen_qwen3.6-plus",
        "google/gemma-4-26b-a4b-it:nitro": "gemma4_26b",
        "google/gemma-4-31b-it:nitro": "gemma4_31b",
        "mistralai/ministral-14b-2512:nitro": "ministral-3_14b",
    }
    return aliases.get(lower, text)


def load_results(csv_paths, reported_path=None):
    frames = []
    provenance = []
    for raw_path in csv_paths:
        path = Path(raw_path).resolve()
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        if missing := {"model_name", "experiment_id", "total_runs"} - set(frame.columns):
            raise ValueError(f"Missing identity/count columns in {path.name}: {sorted(missing)}")
        frame["model_key"] = frame["model_name"].map(canonical_model)
        frame["source_kind"] = "aggregate_csv"
        frames.append(frame)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = pd.concat(frames, ignore_index=True)
    if reported_path is not None:
        path = Path(reported_path).resolve()
        reported = json.loads(path.read_text())
        extra = pd.DataFrame(reported["rows"])
        extra["model_key"] = extra["model_name"].map(canonical_model)
        extra["source_kind"] = "reported_rounded"
        # A supplied aggregate takes precedence over the explicitly rounded overlay.
        present = set(zip(data["model_key"], data["experiment_id"]))
        extra = extra[[key not in present for key in zip(extra["model_key"], extra["experiment_id"])]]
        if len(extra):
            data = pd.concat([data, extra], ignore_index=True)
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "description": reported["source"]})
    unknown = set(data["model_key"]) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unregistered models: {sorted(unknown)}. Register them before plotting.")
    data = data[data["experiment_id"].isin(EXPERIMENTS)].copy()
    if data.duplicated(["model_key", "experiment_id"]).any():
        raise ValueError("Duplicate model/condition rows: combine results explicitly before plotting.")
    for column in data.columns:
        if column.endswith(("_mean", "_se")) or column == "total_runs":
            data[column] = pd.to_numeric(data[column], errors="raise")
    if data["total_runs"].isna().any() or not (data["total_runs"] > 0).all():
        raise ValueError("Every model/condition must have a positive episode count.")
    required = {f"{metric}_mean" for metric in METRICS}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing core metric columns: {sorted(missing)}")
    expected = {(model, exp) for model in MODEL_NAMES for exp in EXPERIMENTS}
    found = set(zip(data["model_key"], data["experiment_id"]))
    if missing := expected - found:
        raise ValueError(f"Missing paper model/condition rows: {sorted(missing)}")
    for metric in METRICS:
        values = data[f"{metric}_mean"].dropna()
        maximum = 100 if metric == "TTD" else 1
        if not values.between(0, maximum).all():
            raise ValueError(f"Out-of-range {metric} values; input results were not modified.")
    # Notebook cells use canonical IDs for labels and per-model row selection.
    data["model_name"] = data["model_key"]
    data.attrs["provenance"] = provenance
    return data


def read_results(csv_path, reported_path=None):
    """Read the unchanged aggregates and append only the reported Cosmos rows."""
    return load_results([csv_path], reported_path).drop(columns=["model_key", "source_kind"])


# Zero-based positions of the five existing final figure cells.
PLOTS = {
    "radar_profiles_final": 8,
    "scatter_tsa_caal_ttd": 9,
    "Heatmap_final": 10,
    "e5_identification_outcomes": 11,
    "Tsa_Complete-Tsa": 12,
}


class _RouteExports(ast.NodeTransformer):
    """Keep plotting code intact; let the batch runner route saved files."""
    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if call.func.attr in {"savefig", "show"}:
                return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def _save_options(tree, extension):
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig"]
    matches = [call for call in calls if call.args and f".{extension}" in ast.unparse(call.args[0])]
    options = {}
    for kw in (matches or calls)[0].keywords:
        if kw.arg == "format":
            continue
        options[kw.arg] = ast.literal_eval(kw.value)
    return options


def render_all(data, output_dir=DEFAULT_OUT):
    """Execute the original final plotting cells with only Cosmos added."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((ROOT / "notebooks/results_analysis.ipynb").read_text())
    plt.switch_backend("Agg")
    paths = []
    frame = data.drop(columns=["model_key", "source_kind"]).copy()
    for name, index in PLOTS.items():
        source = "".join(notebook["cells"][index]["source"])
        tree = ast.parse(source)
        pdf_options = _save_options(tree, "pdf")
        png_options = _save_options(tree, "png")
        env = {"__name__": "__paper_plot__", "CSV_PATH": DEFAULT_CSV,
               "OUTPUT_DIR": output_dir, "df": frame.copy(),
               "read_results": lambda *_args, **_kwargs: frame.copy()}
        routed = ast.fix_missing_locations(_RouteExports().visit(tree))
        exec(compile(routed, f"results_analysis.ipynb:cell-{index + 1}", "exec"), env)
        fig = env["fig"]
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, **(png_options if extension == "png" else pdf_options))
            paths.append(path)
        plt.close(fig)
    manifest = {
        "models": MODEL_NAMES,
        "condition_ids": dict(zip(EXPERIMENTS, range(1, 6))),
        "inputs": data.attrs.get("provenance", []),
        "rows": data[["model_key", "experiment_id", "total_runs", "source_kind"]].to_dict("records"),
        "plotting_policy": "Original notebook figure cells: original transforms, percentile marker sizes, palettes, fonts, layouts and references. Cosmos appended without changing the existing model styles.",
        "metric_policy": "Existing aggregate values only. No evaluator, raw-episode processing or changes to metric conventions.",
        "cosmos_limitations": (
            "Rounded core values from PDF Table 1; auxiliary statistics unavailable (N/A)."
            if (data["source_kind"] == "reported_rounded").any()
            else "Aggregate CSV values; undefined auxiliary statistics remain N/A."
        ),
        "files": [p.name for p in paths],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, default=[DEFAULT_CSV])
    parser.add_argument("--reported-cosmos", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = load_results(args.csv, args.reported_cosmos)
    paths = render_all(data, args.output_dir)
    print(f"Rendered {len(PLOTS)} original-style figures for nine models ({len(paths)} exports).")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
