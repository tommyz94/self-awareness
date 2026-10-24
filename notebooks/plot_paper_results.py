"""Benchmark figures from aggregate CSVs; no raw-episode metric calculation."""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/figures"
# Official paper conditions, matching the evaluator CSV.
EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")
EXPERIMENT_LABELS = dict(zip(EXPERIMENTS, ("E1", "E2", "E3", "E4", "E5")))

# Preserve the paper palette and its completion-gap label/order variants.
MODEL_STYLES = {
    "anthropic_claude-sonnet-4.6": {
        "label": 'Claude Sonnet 4.6', "color": '#0072B2', "marker": 'o',
        "gap_label": 'Claude Sonnet 4.6', "gap_order": 7,
    },
    "qwen_qwen3.6-plus": {
        "label": 'Qwen 3.6+', "color": '#D55E00', "marker": 's',
        "gap_label": 'Qwen 3.6 Plus', "gap_order": 5,
    },
    "gemma4_26b": {
        "label": 'Gemma 4 26B', "color": '#009E73', "marker": '^',
        "gap_label": 'Gemma4 26B', "gap_order": 1,
    },
    "gemma4_31b": {
        "label": 'Gemma 4 31B', "color": '#CC79A7', "marker": 'D',
        "gap_label": 'Gemma4 31B', "gap_order": 2,
    },
    "gpt-5.1": {
        "label": 'GPT-5.1', "color": '#E69F00', "marker": 'P',
        "gap_label": 'GPT 5.1', "gap_order": 6,
    },
    "gemini-2.5-flash": {
        "label": 'Gemini 2.5 Flash', "color": '#56B4E9', "marker": 'X',
        "gap_label": 'Gemini 2.5 Flash', "gap_order": 3,
    },
    "gemini-2.5-pro": {
        "label": 'Gemini 2.5 Pro', "color": '#6A3D9A', "marker": 'v',
        "gap_label": 'Gemini 2.5 Pro', "gap_order": 4,
    },
    "ministral-3_14b": {
        "label": 'Ministral-3 14B', "color": '#6B6B6B', "marker": 'h',
        "gap_label": 'Ministral 3 14B', "gap_order": 0, "radar_color": '#4D4D4D',
    },
    "cosmos-reason2-8b": {
        "label": 'Cosmos-Reason2 8B', "color": '#8F305A', "marker": '>',
        "gap_label": 'Cosmos-Reason2 8B', "gap_order": 8,
    },
}
MODEL_NAMES = {key: style["label"] for key, style in MODEL_STYLES.items()}
METRICS = ("TSA", "TTD", "MCR", "MTATO", "CAAL", "CONFABULATION")
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "h", ">")


def model_styles(df):
    """Known paper styles plus stable, automatic styles for any other model ID."""
    present = set(df["model_name"])
    styles = {key: dict(style) for key, style in MODEL_STYLES.items() if key in present}
    for key in sorted(present - styles.keys()):
        code = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        color = colorsys.hls_to_rgb((code % 360) / 360, 0.45, 0.65)
        styles[key] = {"label": key, "color": color, "marker": MARKERS[code % len(MARKERS)],
                       "gap_label": key, "gap_order": len(MODEL_STYLES)}
    return styles


def available_conditions(df):
    return [exp for exp in EXPERIMENTS if exp in set(df["experiment_id"])]


def annotate_heatmap(ax, values, threshold, fontsize):
    # Seaborn omits masked cells; locate annotations by position, not flat index.
    for text in ax.texts:
        x, y = text.get_position()
        value = values.iloc[int(y), int(x)]
        text.set_color("white" if value >= threshold else "#222222")
    for i, j in zip(*np.where(values.isna())):
        ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                fontsize=fontsize, color="#666666")


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


def load_results(csv_paths):
    """Read evaluator CSVs without changing their values; missing statistics stay NaN."""
    frames = []
    provenance = []
    for raw_path in csv_paths:
        path = Path(raw_path).resolve()
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        if missing := {"model_name", "experiment_id", "total_runs"} - set(frame.columns):
            raise ValueError(f"Missing identity/count columns in {path.name}: {sorted(missing)}")
        if "experiment_id_scheme" in frame:
            schemes = set(frame["experiment_id_scheme"].dropna())
            if schemes != {"paper-v1"} or frame["experiment_id_scheme"].isna().any():
                raise ValueError("Declare paper-v1 consistently for canonical E1-E5 aggregates")
        frame["model_key"] = frame["model_name"].map(canonical_model)
        frame["source_kind"] = "aggregate_csv"
        frames.append(frame)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("No aggregate rows to plot")
    if data["model_name"].isna().any() or data["model_key"].eq("").any():
        raise ValueError("Every row must have a nonempty model_name")
    if not data["experiment_id"].isin(EXPERIMENTS).all():
        raise ValueError("Unsupported condition IDs; use e1-e5 from the evaluator CSV")
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
    # Auxiliary statistics are optional; absence must not become a zero score.
    for column in ("TSA-COMPLETE_mean", "ident_confusion_counts_json"):
        if column not in data:
            data[column] = np.nan
    for metric in METRICS:
        values = data[f"{metric}_mean"].dropna()
        maximum = 100 if metric == "TTD" else 1
        if not values.between(0, maximum).all():
            raise ValueError(f"Out-of-range {metric} values; input results were not modified.")
    # Plotting functions use canonical model IDs for row selection.
    data["model_name"] = data["model_key"]
    data.attrs["provenance"] = provenance
    return data


# Keep styling and transformations shared instead of repeating them per figure.
plt_style = {"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 140,
             "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9}
SCORE_COLORS = ["#F7F7F7", "#DCEDE7", "#8EC5B6", "#3B8EA5", "#1D4E89"]


def reverse_scale(values):
    """Map smaller values to higher scores; preserve undefined observations."""
    low, high = values.min().min(), values.max().max()
    if pd.isna(low):
        return values * np.nan
    if np.isclose(low, high):
        return (values * 0 + 0.5).where(values.notna())
    return (1 - (values - low) / (high - low)).clip(0, 1)


def style_axis(ax, grid_axis="both"):
    ax.set_facecolor("#FAFAF8")
    ax.grid(True, axis=grid_axis, color="#D9D9D6", linewidth=0.65, alpha=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(length=3, width=0.75, colors="#333333", pad=3)


def color_model_labels(ax, models, styles):
    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_yticklabels([styles[m]["label"] for m in models], rotation=0)
    for text, model in zip(ax.get_yticklabels(), models):
        text.set_color(styles[model]["color"])
        text.set_fontweight("semibold")


def heatmap(ax, scores, annotations, *, colors=SCORE_COLORS, **kwargs):
    cmap = LinearSegmentedColormap.from_list("scores", colors)
    cmap.set_bad("white")
    sns.heatmap(scores, ax=ax, cmap=cmap, vmin=0, vmax=1,
                annot=annotations, fmt="", linewidths=0.45, linecolor="white",
                annot_kws={"fontsize": 8}, **kwargs)
    annotate_heatmap(ax, scores, threshold=0.62, fontsize=8)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)


def plot_radar(df):
    """Six metrics by condition; TTD and CR are reversed for visual comparison."""
    styles = model_styles(df)
    conditions = available_conditions(df)
    baseline = df[df.experiment_id == "e1"].sort_values("TSA_mean", ascending=False)
    models = baseline.model_name.tolist()
    models += [m for m in styles if m not in models]
    if "cosmos-reason2-8b" in models:
        models.remove("cosmos-reason2-8b")
        models.append("cosmos-reason2-8b")
    # Retain the paper radar's palette assigned by baseline rank.
    palette = [s for m, s in MODEL_STYLES.items() if m != "cosmos-reason2-8b"]
    ranked = iter(palette)
    radar_styles = {m: next(ranked) if m in MODEL_STYLES and m != "cosmos-reason2-8b"
                    else styles[m] for m in models}
    scores = df.copy()
    scores["TTD_mean"] = reverse_scale(scores["TTD_mean"])
    scores["CONFABULATION_mean"] = 1 - scores["CONFABULATION_mean"]
    columns = [f"{m}_mean" for m in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    with plt.rc_context({**plt_style, "font.family": "serif"}):
        fig, axes = plt.subplots(1, len(conditions), squeeze=False,
                                 subplot_kw={"projection": "polar"},
                                 figsize=(max(6, 4.3 * len(conditions)), 5.8))
        handles = []
        for ax, condition in zip(axes.flat, conditions):
            rows = scores[scores.experiment_id == condition].set_index("model_name")
            for i, model in enumerate(models):
                values = rows.reindex(models).loc[model, columns].to_numpy(dtype=float)
                style = radar_styles[model]
                color = style.get("radar_color", style["color"])
                line, = ax.plot(closed_angles, np.r_[values, values[0]],
                                color=color, marker=style["marker"],
                                linewidth=2.3 if i < 3 else 1.25,
                                markersize=7 if i < 3 else 5.8,
                                alpha=0.94 if i < 3 else 0.55,
                                markeredgecolor="white", label=styles[model]["label"])
                if ax is axes.flat[0]:
                    handles.append(line)
            if condition != "e5":
                ax.scatter([0], [1 / 3], color="#C00000", marker="X",
                           edgecolors="black", s=68, zorder=80)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set(ylim=(0, 1), title=EXPERIMENT_LABELS[condition])
            ax.set_xticks(angles, ["TSA", "TTD", "MCR", "MTATO", "CAAL", "CR"])
            ax.set_yticks([0.25, 0.5, 0.75, 1], ["", "0.5", "", "1.0"])
            ax.grid(alpha=0.28)
            if condition == "e2":
                for index in (2, 3):
                    ax.get_xticklabels()[index].set_color("#999999")
        if any(c != "e5" for c in conditions):
            handles.append(Line2D([], [], marker="X", color="#C00000",
                                  linestyle="None", label="Chance TSA"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
        if scores[columns].isna().any().any():
            fig.text(0.995, 0.99, "Gaps in profiles = N/A", ha="right", va="top",
                     fontsize=8, color="#666666")
        fig.subplots_adjust(left=0.02, right=0.995, top=0.90, bottom=0.18, wspace=0.18)
    return fig


def plot_scatter(df):
    """TSA against CAAL; marker areas encode TTD percentile groups."""
    styles, conditions = model_styles(df), available_conditions(df)
    ttd = df.TTD_mean.dropna()
    with plt.rc_context(plt_style):
        fig, axes = plt.subplots(1, len(conditions), squeeze=False, sharex=True,
                                 sharey=True, figsize=(max(6, 3.36 * len(conditions)), 4.25))
        for ax, condition in zip(axes.flat, conditions):
            style_axis(ax)
            ax.plot([0, 1], [0, 1], "--", color="#9B9B9B", linewidth=0.9)
            rows = df[df.experiment_id == condition].set_index("model_name").reindex(styles)
            missing = []
            for model, row in rows.iterrows():
                style = styles[model]
                if row[["TSA_mean", "CAAL_mean", "TTD_mean"]].isna().any():
                    missing.append(style["label"])
                    continue
                percentile = 100 * (ttd <= row.TTD_mean).mean()
                size = 58 if percentile < 20 else 118 if percentile < 60 else 205
                ax.scatter(row.CAAL_mean, row.TSA_mean, s=size, color=style["color"],
                           marker=style["marker"], edgecolors="white", alpha=0.86)
            if missing:
                ax.text(0.02, 0.98, "N/A: " + ", ".join(missing),
                        transform=ax.transAxes, va="top", fontsize=7, wrap=True)
            ax.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035),
                   title=EXPERIMENT_LABELS[condition], aspect="equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
        handles = [Line2D([], [], linestyle="None", marker=s["marker"],
                          color=s["color"], label=s["label"]) for s in styles.values()]
        for size, label in [(58, "Fastest 20%"), (118, "Middle 40%"), (205, "Slowest 40%")]:
            handles.append(Line2D([], [], linestyle="None", marker="o", color="#9A9A9A",
                                  alpha=0.42, markersize=np.sqrt(size), label=label))
        axes.flat[0].set_ylabel("TSA")
        fig.supxlabel("CAAL", y=0.12)
        fig.suptitle("TSA vs. CAAL Across Experiments", fontweight="bold")
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
        fig.subplots_adjust(left=0.055, right=0.992, top=0.84, bottom=0.23, wspace=0.18)
    return fig


def plot_heatmap(df):
    """Raw metric annotations; colors reverse CR and scale TTD within each condition."""
    styles, conditions = model_styles(df), available_conditions(df)
    models = list(styles)
    with plt.rc_context(plt_style):
        fig, axes = plt.subplots(1, len(conditions), squeeze=False,
                                 figsize=(max(6, 3.6 * len(conditions)), max(3, 6 * len(models) / 9)))
        for index, (ax, condition) in enumerate(zip(axes.flat, conditions)):
            rows = df[df.experiment_id == condition].set_index("model_name").reindex(models)
            raw = rows[[f"{m}_mean" for m in METRICS]].copy()
            raw.columns = ["TSA", "TTD", "MCR", "MTATO", "CAAL", "CR"]
            scores = raw.copy()
            scores["TTD"] = reverse_scale(raw.TTD)
            scores["CR"] = 1 - raw.CR
            annotations = raw.copy().astype(object)
            for column in raw:
                fmt = "{:.1f}" if column == "TTD" else "{:.2f}"
                annotations[column] = raw[column].map(lambda v: "N/A" if pd.isna(v) else fmt.format(v))
            heatmap(ax, scores, annotations.values, cbar=index == len(conditions) - 1)
            ax.set_title(EXPERIMENT_LABELS[condition], fontweight="semibold")
            if index == 0:
                color_model_labels(ax, models, styles)
            else:
                ax.set_yticklabels([])
        fig.suptitle("Model Performance Across Experiments and Metrics", fontweight="bold")
        fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.15, wspace=0.18)
    return fig


def plot_identification(df):
    """Final recorded identification in E5, including the last claim on timeouts."""
    if "e5" not in set(df.experiment_id):
        return None
    rows = df[df.experiment_id == "e5"].copy()
    styles = model_styles(rows)
    rows = rows.set_index("model_name").reindex(styles)
    fractions = []
    for _, row in rows.iterrows():
        counts = row.ident_confusion_counts_json
        counts = json.loads(counts) if isinstance(counts, str) else counts
        if not isinstance(counts, dict) or not counts:
            fractions.append([np.nan] * 3)
            continue
        correct = sum(pred.get(truth, 0) for truth, pred in counts.items())
        unknown = sum(pred.get("unknown_identification", 0) for pred in counts.values())
        fractions.append([correct / row.total_runs,
                          (row.total_runs - correct - unknown) / row.total_runs,
                          unknown / row.total_runs])
    values = np.array(fractions)
    with plt.rc_context(plt_style):
        fig, ax = plt.subplots(figsize=(10.8, max(3, 0.6 * len(styles))))
        style_axis(ax, "x")
        left = np.zeros(len(rows))
        ys = np.arange(len(rows))
        categories = [("Correct", "#5A9B58"), ("Wrong final identification", "#D47A2A"),
                      ("Unknown final identification", "#D2D2CF")]
        for index, (label, color) in enumerate(categories):
            widths = np.nan_to_num(values[:, index])
            ax.barh(ys, widths, left=left, color=color, height=0.66, edgecolor="white", label=label)
            for y, width in enumerate(widths):
                if width >= 0.075:
                    ax.text(left[y] + width / 2, y, f"{width:.0%}", ha="center", va="center",
                            color="#333333" if index == 2 else "white", fontsize=8.5)
            left += widths
        for y, model in enumerate(rows.index):
            style = styles[model]
            ax.scatter(-0.045, y, s=92, color=style["color"], marker=style["marker"], clip_on=False)
            if np.isnan(values[y, 0]):
                ax.text(0.5, y, "N/A", ha="center", va="center", color="#666666")
        ax.set_yticks(ys, [styles[m]["label"] for m in rows.index])
        ax.invert_yaxis()
        ax.set(xlim=(-0.08, 1.02), xlabel="Fraction of runs (%)",
               title="E5: Final Identification Outcome Decomposition")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
        fig.subplots_adjust(left=0.20, right=0.985, top=0.86, bottom=0.24)
    return fig


def plot_completion_gap(df):
    """Difference between success among done episodes and overall success."""
    styles, conditions = model_styles(df), available_conditions(df)
    models = sorted(styles, key=lambda m: styles[m]["gap_order"])
    values = df.assign(gap=df["TSA-COMPLETE_mean"] - df.TSA_mean).pivot(
        index="model_name", columns="experiment_id", values="gap").reindex(index=models, columns=conditions)
    values.columns = [EXPERIMENT_LABELS[c] for c in conditions]
    annotations = values.map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}")
    # Colors express relative gap, while annotations retain the unscaled difference.
    normalized = 1 - reverse_scale(values)
    with plt.rc_context(plt_style):
        fig, ax = plt.subplots(figsize=(8.4, max(3, 5.1 * len(models) / 9)))
        heatmap(ax, normalized, annotations.values,
                colors=["#F7F7F7", "#FBE5D6", "#F4A261", "#D55E00", "#8C2D04"])
        ax.set_yticklabels([styles[m]["gap_label"] for m in models], rotation=0)
        ax.set_title(r"Completion Gap: $\mathrm{TSA}_{\mathrm{complete}} - \mathrm{TSA}$", pad=14)
        fig.subplots_adjust(left=0.24, right=0.96, top=0.87, bottom=0.15)
    return fig


PLOTS = {
    "radar_profiles_final": (plot_radar, 500, 0.05),
    "scatter_tsa_caal_ttd": (plot_scatter, 300, 0.025),
    "Heatmap_final": (plot_heatmap, 300, 0.1),
    "e5_identification_outcomes": (plot_identification, 300, 0.1),
    "Tsa_Complete-Tsa": (plot_completion_gap, 500, 0.03),
}


def render_all(data, output_dir=DEFAULT_OUT):
    """Export available figures as PDF, SVG and PNG and record their input hashes."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, skipped = [], []
    for name, (plot, dpi, padding) in PLOTS.items():
        with plt.rc_context():
            fig = plot(data)
        if fig is None:
            skipped.append(name)
            continue
        try:
            for extension in ("pdf", "svg", "png"):
                path = output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=dpi if extension == "png" else None,
                            bbox_inches="tight", pad_inches=padding)
                paths.append(path)
        finally:
            plt.close(fig)
    manifest = {"inputs": data.attrs.get("provenance", []),
                "models": {m: s["label"] for m, s in model_styles(data).items()},
                "condition_ids": {c: EXPERIMENT_LABELS[c] for c in available_conditions(data)},
                "rows": data[["model_key", "experiment_id", "total_runs", "source_kind"]].to_dict("records"),
                "skipped_figures": skipped, "files": [p.name for p in paths]}
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True, help="Evaluator paper_metrics.csv files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    data = load_results(args.csv)
    paths = render_all(data, args.output_dir)
    print(f"Exported {len(paths)} files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
