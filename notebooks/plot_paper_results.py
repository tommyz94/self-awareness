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
DEFAULT_OUT = ROOT / "notebooks/paper_assets/all_models"
# Internal aggregate IDs; paper-v1 CSVs are translated on input.
EXPERIMENTS = ("e1", "e2", "e3", "e5", "e7")
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


def load_results(csv_paths, reported_path=None):
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
            # Adapt public E1-E5 IDs to the aggregate IDs used by the plots.
            mapping = {"e1": "e1", "e2": "e2", "e3": "e3", "e4": "e5", "e5": "e7"}
            if not frame["experiment_id"].isin(mapping).all():
                raise ValueError("Invalid paper-v1 experiment IDs")
            frame["experiment_id"] = frame["experiment_id"].map(mapping)
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
    if data.empty:
        raise ValueError("No aggregate rows to plot")
    if data["model_name"].isna().any() or data["model_key"].eq("").any():
        raise ValueError("Every row must have a nonempty model_name")
    if not data["experiment_id"].isin(EXPERIMENTS).all():
        raise ValueError("Unsupported condition IDs; use paper-v1 for E1-E5 CSVs")
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


def read_results(csv_path, reported_path=None):
    """Read validated aggregate values for notebook use."""
    return load_results([csv_path], reported_path).drop(columns=["model_key", "source_kind"])


def plot_radar(df):
    styles = model_styles(df)
    conditions = available_conditions(df)
    labels = {key: spec["label"] for key, spec in styles.items()}

    radar_metrics = {
        "TSA":   {"source": "TSA_mean",            "label": "TSA",   "transform": "identity"},
        "TTD":   {"source": "TTD_mean",            "label": "TTD",   "transform": "reverse_minmax"},
        "MCR":   {"source": "MCR_mean",            "label": "MCR",   "transform": "identity"},
        "MTATO": {"source": "MTATO_mean",          "label": "MTATO", "transform": "identity"},
        "CAAL":  {"source": "CAAL_mean",           "label": "CAAL",  "transform": "identity"},
        "CR":    {"source": "CONFABULATION_mean",  "label": "CR",    "transform": "reverse_identity"},
    }

    chance_tsa = {
        "e1": 1 / 3,
        "e2": 1 / 3,
        "e3": 1 / 3,
        "e5": 1 / 3,
    }

    n_top_models = 3

    plot_df = df.copy()
    plot_df.columns = plot_df.columns.astype(str).str.strip()

    plot_df["model_label"] = plot_df["model_name"].map(labels).fillna(plot_df["model_name"])

    for col in plot_df.columns:
        if col not in ["experiment_id", "model_name", "model_label"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df[plot_df["experiment_id"].isin(conditions)].copy()

    ttd_min = float(plot_df["TTD_mean"].min())
    ttd_max = float(plot_df["TTD_mean"].max())

    def reverse_minmax(series, min_val, max_val):
        if np.isclose(max_val, min_val):
            return pd.Series(0.5, index=series.index).where(series.notna())
        return 1 - (series - min_val) / (max_val - min_val)

    for metric_name, spec in radar_metrics.items():
        source = spec["source"]
        out_col = f"{metric_name}_plot"

        if spec["transform"] == "identity":
            plot_df[out_col] = plot_df[source].clip(0, 1)
        elif spec["transform"] == "reverse_identity":
            plot_df[out_col] = (1 - plot_df[source]).clip(0, 1)
        elif spec["transform"] == "reverse_minmax":
            plot_df[out_col] = reverse_minmax(plot_df[source], ttd_min, ttd_max).clip(0, 1)
        else:
            raise ValueError(f"Unknown transform for {metric_name}: {spec['transform']}")

    model_order = (
        plot_df[(plot_df["experiment_id"] == "e1") & (plot_df["model_name"] != "cosmos-reason2-8b")]
        .sort_values("TSA_mean", ascending=False)["model_label"]
        .tolist()
    )

    if len(model_order) == 0:
        model_order = (
            plot_df.groupby("model_label")["TSA_mean"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

    # Include models lacking E1, then keep the paper's Cosmos-last ordering.
    missing_models = [s["label"] for s in styles.values() if s["label"] not in model_order]
    model_order += missing_models
    if "Cosmos-Reason2 8B" in model_order:
        model_order.remove("Cosmos-Reason2 8B")
        model_order.append("Cosmos-Reason2 8B")
    paper_styles = list(MODEL_STYLES.values())
    by_label = {s["label"]: s for s in styles.values()}
    # The paper radar assigns the original palette by rank, unlike other figures.
    rank_styles = [s for s in paper_styles if s["label"] != "Cosmos-Reason2 8B"]
    colors, markers = [], []
    rank = 0
    for label in model_order:
        style = by_label[label]
        if label in MODEL_NAMES.values() and label != "Cosmos-Reason2 8B":
            style = rank_styles[rank]
            rank += 1
        colors.append(style.get("radar_color", style["color"]))
        markers.append(style["marker"])

    top_models = set(model_order[:n_top_models])

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 15,
        "figure.titlesize": 18,
        "legend.fontsize": 11,
        "font.family": "serif",
    })

    metric_keys = list(radar_metrics.keys())
    metric_labels = [radar_metrics[m]["label"] for m in metric_keys]
    plot_cols = [f"{m}_plot" for m in metric_keys]
    n_metrics = len(metric_keys)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    tsa_angle = angles[0]

    fig, axes = plt.subplots(
        1,
        len(conditions),
        subplot_kw={"projection": "polar"},
        figsize=(max(6, 4.3 * len(conditions)), 5.8),
        constrained_layout=False,
    )

    if len(conditions) == 1:
        axes = [axes]

    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.92,      # more room for the plots
        bottom=0.15,
        wspace=0.18,   # less wasted gap between panels
    )

    for ax, exp in zip(axes, conditions):
        sub_exp = (
            plot_df[plot_df["experiment_id"] == exp]
            .set_index("model_label")
            .reindex(model_order)
        )

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_facecolor("#FCFCFC")

        if exp != "e7":
            chance_tsa_value = chance_tsa[exp]

            ax.scatter(
                [tsa_angle],
                [chance_tsa_value],
                s=68,
                color="#C00000",
                marker="X",
                linewidths=1.2,
                edgecolor="black",
                zorder=80,
            )

        for idx, model in enumerate(model_order):
            values = sub_exp.loc[model, plot_cols].astype(float).values
            values_closed = np.concatenate([values, [values[0]]])

            is_top = model in top_models

            ax.plot(
                angles_closed,
                values_closed,
                color=colors[idx],
                linewidth=2.3 if is_top else 1.25,   # thinner non-top lines
                linestyle="-",
                marker=markers[idx],
                markersize=7.0 if is_top else 5.8,   # slightly smaller non-top markers
                markerfacecolor=colors[idx],
                markeredgecolor="white",
                markeredgewidth=0.85,
                alpha=0.94 if is_top else 0.55,      # much less spaghetti
                label=model,
                zorder=20 + idx if is_top else 10 + idx,
            )

        ax.set_title(EXPERIMENT_LABELS[exp], pad=12, fontweight="bold")

        ax.set_xticks(angles)
        ax.set_xticklabels(metric_labels, fontsize=10.8, fontweight="bold")
        ax.tick_params(axis="x", pad=7)   # bring labels closer, frees inner space

        if exp == "e2":
            xticklabels = ax.get_xticklabels()
            for key in ["MCR", "MTATO"]:
                idx = metric_keys.index(key)
                xticklabels[idx].set_color("#999999")

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=8.0, color="#666666")
        ax.set_rlabel_position(18)

        ax.grid(alpha=0.28, linewidth=0.9, color="#9A9A9A")
        ax.spines["polar"].set_alpha(0.68)
        ax.spines["polar"].set_linewidth(1.0)
        ax.spines["polar"].set_color("#888888")

    legend_handles = []
    legend_labels = []

    for idx, model in enumerate(model_order):
        is_top = model in top_models
        legend_handles.append(
            Line2D(
                [0], [0],
                color=colors[idx],
                linewidth=2.3 if is_top else 1.25,
                linestyle="-",
                marker=markers[idx],
                markersize=6.7 if is_top else 5.8,
                markerfacecolor=colors[idx],
                markeredgecolor="white",
                markeredgewidth=0.85,
                alpha=0.94 if is_top else 0.55,
                label=model,
            )
        )
        legend_labels.append(model)

    chance_handle = Line2D(
        [0], [0],
        marker="X",
        color="w",
        markerfacecolor="#C00000",
        markeredgecolor="black",
        markersize=8.0,
        linewidth=0,
        label="Chance TSA",
    )

    fig.legend(
        legend_handles + ([chance_handle] if any(e != "e7" for e in conditions) else []),
        legend_labels + (["Chance TSA"] if any(e != "e7" for e in conditions) else []),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=9,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.7,
        handletextpad=0.35,
        borderpad=0.05,
        labelspacing=0.55,
        fontsize=15.0,
    )
    if plot_df[plot_cols].isna().any().any():
        fig.text(0.995, 0.99, "Gaps in profiles = N/A", ha="right", va="top",
                 fontsize=8, color="#666666")
    return fig


def plot_scatter(df):
    styles = model_styles(df)
    model_ids = list(styles)
    conditions = available_conditions(df)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    df = df[df["experiment_id"].isin(conditions)].copy()
    df["paper_exp"] = df["experiment_id"].map(EXPERIMENT_LABELS)

    def style_axis(ax):
        ax.set_facecolor("#FAFAF8")

        ax.grid(
            True,
            color="#D9D9D6",
            linewidth=0.65,
            alpha=0.55,
        )
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)

        ax.tick_params(
            axis="both",
            length=3,
            width=0.75,
            colors="#333333",
            pad=3,
        )

    def model_legend_handles():
        return [
            Line2D(
                [0],
                [0],
                color="none",
                marker=spec["marker"],
                linestyle="None",
                markersize=6.8,
                markerfacecolor=spec["color"],
                markeredgecolor="white",
                markeredgewidth=0.85,
                label=spec["label"],
            )
            for spec in styles.values()
        ]

    ttd_valid = df["TTD_mean"].dropna()

    def ttd_percentile(ttd):
        if pd.isna(ttd) or len(ttd_valid) == 0:
            return np.nan

        return 100.0 * (ttd_valid <= ttd).mean()

    def ttd_size_bin(ttd):
        pct = ttd_percentile(ttd)

        if pd.isna(pct):
            return 70, "unknown"

        if pct < 20:
            return 58, "Fastest 20%"

        if pct < 60:
            return 118, "Middle 40%"

        return 205, "Slowest 40%"

    def size_legend_handles():
        return [
            plt.scatter([], [], s=58,  color="#9A9A9A", alpha=0.42, edgecolors="none", label="Fastest 20%"),
            plt.scatter([], [], s=118, color="#9A9A9A", alpha=0.42, edgecolors="none", label="Middle 40%"),
            plt.scatter([], [], s=205, color="#9A9A9A", alpha=0.42, edgecolors="none", label="Slowest 40%"),
        ]

    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(max(6, 3.36 * len(conditions)), 4.25),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, exp_id in zip(axes, conditions):
        sub = (
            df[df["experiment_id"] == exp_id]
            .set_index("model_name")
            .reindex(model_ids)
            .reset_index()
        )

        style_axis(ax)

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle=(0, (4, 3)),
            color="#9B9B9B",
            linewidth=0.9,
            alpha=0.65,
            zorder=1,
        )

        for _, row in sub.iterrows():
            model_id = row["model_name"]

            if model_id not in styles:
                continue

            spec = styles[model_id]
            if row[["CAAL_mean", "TSA_mean", "TTD_mean"]].isna().any():
                continue
            size, _ = ttd_size_bin(row["TTD_mean"])

            ax.scatter(
                row["CAAL_mean"],
                row["TSA_mean"],
                s=size,
                color=spec["color"],
                marker=spec["marker"],
                alpha=0.86,
                edgecolors="white",
                linewidths=0.95,
                zorder=3,
            )

        unavailable = sub.loc[sub[["CAAL_mean", "TSA_mean", "TTD_mean"]].isna().any(axis=1), "model_name"]
        if len(unavailable):
            labels = ", ".join(styles[m]["label"] for m in unavailable)
            ax.text(0.02, 0.98, "N/A: " + labels, transform=ax.transAxes,
                    va="top", fontsize=7, wrap=True)
        ax.set_title(
            EXPERIMENT_LABELS[exp_id],
            fontweight="semibold",
            color="#222222",
            pad=8,
        )

        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)

        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("TSA", labelpad=8)

    fig.supxlabel(
        "CAAL",
        y=0.105,
        fontsize=11,
    )

    axes[-1].text(
        0.58,
        0.68,
        "TSA = CAAL",
        rotation=38,
        color="#777777",
        fontsize=8.5,
        alpha=0.75,
        transform=axes[-1].transAxes,
    )

    fig.suptitle(
        "TSA vs. CAAL Across Experiments",
        y=0.965,
        fontsize=15,
        fontweight="semibold",
        color="#111111",
    )

    all_handles = model_legend_handles() + size_legend_handles()

    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=11,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.28,
        borderaxespad=0.0,
        labelspacing=0.35,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        top=0.84,
        bottom=0.20,
        wspace=0.18,
    )
    return fig


def plot_heatmap(df):
    styles = model_styles(df)
    model_ids = list(styles)
    conditions = available_conditions(df)

    METRICS = dict([
        ("TSA",   {"raw_col": "TSA_mean",           "fmt": "{:.2f}", "mode": "direct"}),
        ("TTD",   {"raw_col": "TTD_mean",           "fmt": "{:.1f}", "mode": "reverse_within_exp"}),
        ("MCR",   {"raw_col": "MCR_mean",           "fmt": "{:.2f}", "mode": "direct"}),
        ("MTATO", {"raw_col": "MTATO_mean",         "fmt": "{:.2f}", "mode": "direct"}),
        ("CAAL",  {"raw_col": "CAAL_mean",          "fmt": "{:.2f}", "mode": "direct"}),
        ("CR",    {"raw_col": "CONFABULATION_mean", "fmt": "{:.2f}", "mode": "reverse_unit"}),
    ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

    df = df[df["experiment_id"].isin(conditions)].copy()

    def normalize_series(series, mode):
        s = series.astype(float)

        if mode == "direct":
            return s.clip(0, 1)

        if mode == "reverse_unit":
            return (1 - s).clip(0, 1)

        if mode == "reverse_within_exp":
            valid = s.dropna()

            if len(valid) == 0:
                return pd.Series(np.nan, index=s.index)

            mn, mx = valid.min(), valid.max()

            if np.isclose(mx, mn):
                return pd.Series(0.5, index=s.index).where(s.notna())

            return (1 - (s - mn) / (mx - mn)).clip(0, 1)

        raise ValueError(f"Unknown mode: {mode}")

    exp_scores = {}
    exp_annots = {}

    for exp in conditions:
        exp_label = EXPERIMENT_LABELS[exp]

        sub = (
            df[df["experiment_id"] == exp]
            .set_index("model_name")
            .reindex(model_ids)
        )

        score_mat = pd.DataFrame(index=model_ids)
        annot_mat = pd.DataFrame(index=model_ids)

        for metric_name, spec in METRICS.items():
            raw = sub[spec["raw_col"]]
            score = normalize_series(raw, spec["mode"])
            fmt = spec.get("fmt", "{:.2f}")

            score_mat[metric_name] = score
            annot_mat[metric_name] = [
                "N/A" if pd.isna(v) else fmt.format(v)
                for v in raw
            ]

        exp_scores[exp_label] = score_mat
        exp_annots[exp_label] = annot_mat

    row_labels = [styles[m]["label"] for m in model_ids]

    cmap = LinearSegmentedColormap.from_list(
        "elegant_score",
        ["#F7F7F7", "#DCEDE7", "#8EC5B6", "#3B8EA5", "#1D4E89"],
    )
    cmap.set_bad("#FFFFFF")

    fig = plt.figure(figsize=(max(6, 3.6 * len(conditions)), max(3, 6 * len(model_ids) / 9)))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        1,
        len(conditions) + 1,
        width_ratios=[6] * len(conditions) + [0.45],
        wspace=0.18,
    )

    axes = [
        fig.add_subplot(gs[0, i])
        for i in range(len(conditions))
    ]

    cax = fig.add_subplot(gs[0, len(conditions)])

    for i, (ax_i, exp_label) in enumerate(zip(axes, exp_scores.keys())):
        show_y = i == 0

        sns.heatmap(
            exp_scores[exp_label],
            ax=ax_i,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=exp_annots[exp_label].values,
            fmt="",
            linewidths=0.45,
            linecolor="#FFFFFF",
            cbar=i == len(axes) - 1,
            cbar_ax=cax if i == len(axes) - 1 else None,
            annot_kws={
                "fontsize": 8,
                "fontweight": "regular",
            },
        )

        annotate_heatmap(ax_i, exp_scores[exp_label], threshold=0.62, fontsize=8)

        ax_i.set_title(
            exp_label,
            fontweight="semibold",
            color="#222222",
            pad=10,
        )

        ax_i.set_xlabel("")
        ax_i.set_ylabel("")
        ax_i.tick_params(axis="both", length=0)

        ax_i.set_xticklabels(
            ax_i.get_xticklabels(),
            rotation=0,
            ha="center",
        )

        if show_y:
            ax_i.set_yticklabels(row_labels, rotation=0)

            for tick, model_id in zip(ax_i.get_yticklabels(), model_ids):
                tick.set_color(styles[model_id]["color"])
                tick.set_fontweight("semibold")
        else:
            ax_i.set_yticklabels([])
            ax_i.set_ylabel("")

        for spine in ax_i.spines.values():
            spine.set_visible(False)

    cax.tick_params(labelsize=8, length=3, width=0.6)

    for spine in cax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Model Performance Across Experiments and Metrics",
        fontsize=16,
        fontweight="semibold",
        color="#111111",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.86,
        bottom=0.15,
    )
    return fig


def plot_identification(df):
    if "e7" not in set(df["experiment_id"]):
        return None
    df = df[df["experiment_id"] == "e7"].copy()
    styles = model_styles(df)
    model_ids = list(styles)
    conditions = available_conditions(df)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    df = df[df["experiment_id"].isin(conditions)].copy()
    df["paper_exp"] = df["experiment_id"].map(EXPERIMENT_LABELS)

    def style_axis(ax):
        ax.set_facecolor("#FAFAF8")

        ax.grid(
            True,
            axis="x",
            color="#D9D9D6",
            linewidth=0.65,
            alpha=0.55,
        )
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)

        ax.tick_params(
            axis="both",
            length=3,
            width=0.75,
            colors="#333333",
            pad=3,
        )

    def parse_json_field(value):
        if isinstance(value, dict):
            return value
        if pd.isna(value):
            return {}

        return json.loads(value)

    def e5_outcome_decomposition(row):
        counts = parse_json_field(row["ident_confusion_counts_json"])
        if not counts:
            return dict.fromkeys(("correct", "wrong", "unknown", "correct_n", "wrong_n", "unknown_n"), np.nan)

        total = int(row["total_runs"])
        unknown = 0
        correct = 0

        for true_label, pred_dict in counts.items():
            for pred_label, c in pred_dict.items():
                if pred_label == "unknown_identification":
                    unknown += c

            if true_label in pred_dict:
                correct += pred_dict[true_label]

        wrong = total - correct - unknown

        return {
            "correct": correct / total if total else np.nan,
            "wrong": wrong / total if total else np.nan,
            "unknown": unknown / total if total else np.nan,
            "correct_n": correct,
            "wrong_n": wrong,
            "unknown_n": unknown,
        }

    sub = (
        df[df["experiment_id"] == "e7"]
        .set_index("model_name")
        .reindex(model_ids)
        .reset_index()
    )

    rows = []
    for _, row in sub.iterrows():
        if pd.isna(row["model_name"]):
            continue

        dec = e5_outcome_decomposition(row)

        rows.append({
            "model_name": row["model_name"],
            **dec,
        })

    plot_df = pd.DataFrame(rows)

    plot_df["model_label"] = plot_df["model_name"].map(
        lambda m: styles[m]["label"]
    )

    ys = np.arange(len(plot_df))

    correct_color = "#5A9B58"
    wrong_color = "#D47A2A"
    unknown_color = "#D2D2CF"

    bar_specs = [
        ("correct", correct_color, "Correct"),
        ("wrong", wrong_color, "Wrong committed"),
        ("unknown", unknown_color, "Unknown / no committed identification"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, max(3, 0.6 * len(model_ids))))
    fig.patch.set_facecolor("white")

    style_axis(ax)

    left = np.zeros(len(plot_df))

    for key, color, label in bar_specs:
        vals = plot_df[key].fillna(0).values

        ax.barh(
            ys,
            vals,
            left=left,
            height=0.66,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.9,
            label=label,
            zorder=2,
        )

        for i, value in enumerate(vals):
            if value >= 0.075:
                ax.text(
                    left[i] + value / 2,
                    ys[i],
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if key in {"correct", "wrong"} else "#333333",
                    fontweight="semibold",
                    zorder=5,
                )

        left += vals

    for i, row in plot_df.iterrows():
        if pd.isna(row["correct"]):
            ax.text(0.5, ys[i], "N/A", ha="center", va="center", fontsize=8.5, color="#666666")

    for i, row in plot_df.iterrows():
        spec = styles[row["model_name"]]

        ax.scatter(
            -0.045,
            ys[i],
            s=92,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.9,
            clip_on=False,
            zorder=4,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(plot_df["model_label"])

    for tick, model_id in zip(ax.get_yticklabels(), plot_df["model_name"]):
        tick.set_color(styles[model_id]["color"])
        tick.set_fontweight("semibold")

    ax.set_xlim(-0.085, 1.02)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])

    ax.set_xlabel("Fraction of runs (%)", labelpad=8)

    ax.invert_yaxis()

    ax.set_title(
        "E5: Final Identification Outcome Decomposition",
        fontweight="semibold",
        color="#111111",
        pad=14,
    )

    for x in [0.25, 0.50, 0.75, 1.0]:
        ax.axvline(
            x,
            color="#D9D9D6",
            linewidth=0.65,
            alpha=0.45,
            zorder=0,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.5,
    )

    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        top=0.86,
        bottom=0.22,
    )
    return fig


def plot_completion_gap(df):
    styles = model_styles(df)
    model_ids = list(styles)
    conditions = available_conditions(df)
    gap_labels = {key: s["gap_label"] for key, s in styles.items()}
    gap_order = [s["gap_label"] for s in sorted(styles.values(), key=lambda s: s["gap_order"])]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    plot_df = df.copy()
    plot_df = plot_df[plot_df["experiment_id"].isin(conditions)].copy()
    plot_df["paper_exp"] = plot_df["experiment_id"].map(EXPERIMENT_LABELS)
    plot_df["paper_model"] = plot_df["model_name"].map(gap_labels)

    plot_df["delta_complete"] = (
        plot_df["TSA-COMPLETE_mean"] - plot_df["TSA_mean"]
    )

    delta_df = (
        plot_df
        .pivot(index="paper_model", columns="paper_exp", values="delta_complete")
        .reindex(
            index=gap_order,
            columns=[EXPERIMENT_LABELS[e] for e in conditions],
        )
    )

    annot_df = delta_df.map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    delta_min = delta_df.min().min()
    delta_max = delta_df.max().max()

    if np.isclose(delta_max, delta_min):
        delta_norm = delta_df.copy() * 0 + 0.5
    else:
        delta_norm = (delta_df - delta_min) / (delta_max - delta_min)

    delta_norm = delta_norm.clip(0, 1)

    cmap = LinearSegmentedColormap.from_list(
        "completion_gap",
        ["#F7F7F7", "#FBE5D6", "#F4A261", "#D55E00", "#8C2D04"],
    )
    cmap.set_bad("#FFFFFF")

    fig, ax = plt.subplots(figsize=(8.4, max(3, 5.1 * len(model_ids) / 9)))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        delta_norm,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot_df.values,
        fmt="",
        linewidths=0.55,
        linecolor="#FFFFFF",
        cbar=True,
        annot_kws={
            "fontsize": 8.8,
            "fontweight": "semibold",
        },
    )

    annotate_heatmap(ax, delta_norm, threshold=0.58, fontsize=8.8)

    ax.set_title(
        r"Completion Gap: $\mathrm{TSA}_{\mathrm{complete}} - \mathrm{TSA}$",
        pad=14,
        fontweight="semibold",
        color="#111111",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        delta_norm.columns,
        rotation=0,
        ha="center",
    )

    ax.set_yticklabels(
        delta_norm.index,
        rotation=0,
    )

    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8.5, length=3, width=0.6)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_size(9)

    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    fig.subplots_adjust(
        left=0.24,
        right=0.96,
        top=0.87,
        bottom=0.15,
    )
    return fig


# Export settings retain the paper figure dimensions and resolution.
PLOTS = {
    "radar_profiles_final": (plot_radar, 500, 0.05),
    "scatter_tsa_caal_ttd": (plot_scatter, 300, 0.025),
    "Heatmap_final": (plot_heatmap, 300, 0.1),
    "e5_identification_outcomes": (plot_identification, 300, 0.1),
    "Tsa_Complete-Tsa": (plot_completion_gap, 500, 0.03),
}


def render_all(data, output_dir=DEFAULT_OUT):
    """Export the figures applicable to the models and conditions in the CSV."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    skipped = []
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
    manifest = {
        "models": {key: style["label"] for key, style in model_styles(data).items()},
        "condition_ids": {exp: EXPERIMENT_LABELS[exp] for exp in available_conditions(data)},
        "inputs": data.attrs.get("provenance", []),
        "rows": data[["model_key", "experiment_id", "total_runs", "source_kind"]].to_dict("records"),
        "metric_policy": "Existing aggregate values only; missing statistics remain N/A.",
        "skipped_figures": skipped,
        "files": [p.name for p in paths],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True, help="One or more aggregate CSVs; use evaluator paper_metrics.csv outputs")
    parser.add_argument("--reported-cosmos", type=Path, help="Optional rounded PDF values, used only when a CSV row is absent")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    data = load_results(args.csv, args.reported_cosmos)
    paths = render_all(data, args.output_dir)
    print(f"Rendered {len(paths) // 3} figures for {data.model_key.nunique()} models ({len(paths)} exports).")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
