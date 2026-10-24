#!/usr/bin/env python3
"""Generate small-multiple plots for the SELF_CORRECTION metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_LABELS = {
    "e1": "Mirror",
    "e2": "No mirror trap",
    "e3": "Wrong prior",
    "e5": "Distractors",
    "e7": "Cluttered exploration",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SELF_CORRECTION correlations from an aggregated metrics CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="final_outputs/reports/result_trimmed.csv",
        help="Metrics CSV with SELF_CORRECTION columns. Default: final_outputs/reports/result_trimmed.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Default: <csv_dir>/self_correction_figures",
    )
    parser.add_argument(
        "--experiments",
        default="e1,e3,e5",
        help="Comma-separated experiment ids to facet. Default: e1,e3,e5",
    )
    parser.add_argument(
        "--min-applicable-runs",
        type=int,
        default=3,
        help="Only plot model-experiment rows with at least this many applicable runs. Default: 3",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI. Default: 220",
    )
    return parser.parse_args()


def _load_data(csv_path: Path, experiments: list[str], min_applicable_runs: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "experiment_id",
        "model_name",
        "TSA_mean",
        "CAAL_mean",
        "CONFABULATION_mean",
        "TTD_mean",
        "SELF_CORRECTION_mean",
        "SELF_CORRECTION_applicable_runs",
        "SELF_CORRECTION_applicable_rate",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[df["experiment_id"].isin(experiments)].copy()
    df["execution_gap"] = df["CAAL_mean"] - df["TSA_mean"]
    df["SELF_CORRECTION_applicable_runs"] = df["SELF_CORRECTION_applicable_runs"].fillna(0).astype(int)
    df = df[df["SELF_CORRECTION_applicable_runs"] >= min_applicable_runs].copy()
    return df


def _model_colors(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {model: cmap(i % 10) for i, model in enumerate(models)}


def _point_sizes(applicable_runs: pd.Series) -> pd.Series:
    return 50 + applicable_runs.astype(float) * 28.0


def _add_labels(ax: plt.Axes, panel_df: pd.DataFrame) -> None:
    for _, row in panel_df.iterrows():
        ax.annotate(
            row["model_name"],
            (row["_x"], row["_y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )


def _plot_small_multiples(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    experiments: list[str],
    model_colors: dict[str, tuple[float, float, float, float]],
    dpi: int,
    add_zero_x: bool = False,
    add_zero_y: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, exp in zip(axes, experiments):
        panel_df = df[df["experiment_id"] == exp].copy()
        ax.set_title(EXPERIMENT_LABELS.get(exp, exp))
        ax.grid(alpha=0.2, linewidth=0.7)

        if add_zero_x:
            ax.axvline(0.0, color="0.4", linewidth=1.0, alpha=0.8)
        if add_zero_y:
            ax.axhline(0.0, color="0.4", linewidth=1.0, alpha=0.8)

        if panel_df.empty:
            ax.text(0.5, 0.5, "No supported runs", ha="center", va="center", transform=ax.transAxes)
            continue

        panel_df["_x"] = panel_df[x_col]
        panel_df["_y"] = panel_df[y_col]
        sizes = _point_sizes(panel_df["SELF_CORRECTION_applicable_runs"])
        colors = [model_colors[m] for m in panel_df["model_name"]]

        ax.scatter(
            panel_df["_x"],
            panel_df["_y"],
            s=sizes,
            c=colors,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
        )
        _add_labels(ax, panel_df)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

    fig.suptitle(title, fontsize=16, y=1.03)
    fig.supxlabel(x_label, fontsize=12)
    fig.supylabel(y_label, fontsize=12)

    # Size legend
    size_values = sorted(set(df["SELF_CORRECTION_applicable_runs"].astype(int).tolist()))
    if size_values:
        legend_values = [size_values[0], size_values[len(size_values) // 2], size_values[-1]]
        legend_values = list(dict.fromkeys(legend_values))
        handles = [
            plt.scatter([], [], s=_point_sizes(pd.Series([v])).iloc[0], color="white", edgecolors="black")
            for v in legend_values
        ]
        fig.legend(
            handles,
            [f"{v} applicable runs" for v in legend_values],
            title="Support",
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.input_csv).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else csv_path.parent / "self_correction_figures"
    )
    experiments = [part.strip() for part in args.experiments.split(",") if part.strip()]

    df = _load_data(csv_path, experiments, args.min_applicable_runs)
    if df.empty:
        raise SystemExit("No model-experiment rows met the plotting filter.")

    models = sorted(df["model_name"].unique().tolist())
    colors = _model_colors(models)

    _plot_small_multiples(
        df=df,
        x_col="CONFABULATION_mean",
        y_col="SELF_CORRECTION_mean",
        title="Self-correction after premature commitment",
        x_label="Confabulation Rate",
        y_label="Self-Correction Rate",
        output_path=output_dir / "self_correction_vs_confabulation.png",
        experiments=experiments,
        model_colors=colors,
        dpi=args.dpi,
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )

    _plot_small_multiples(
        df=df,
        x_col="SELF_CORRECTION_mean",
        y_col="TSA_mean",
        title="Belief revision versus final task accuracy",
        x_label="Self-Correction Rate",
        y_label="Target Selection Accuracy",
        output_path=output_dir / "self_correction_vs_tsa.png",
        experiments=experiments,
        model_colors=colors,
        dpi=args.dpi,
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )

    _plot_small_multiples(
        df=df,
        x_col="SELF_CORRECTION_mean",
        y_col="execution_gap",
        title="Belief revision versus execution gap",
        x_label="Self-Correction Rate",
        y_label="Execution gap: CAAL - TSA",
        output_path=output_dir / "self_correction_vs_execution_gap.png",
        experiments=experiments,
        model_colors=colors,
        dpi=args.dpi,
        add_zero_y=True,
        xlim=(-0.02, 1.02),
    )

    summary_cols = [
        "experiment_id",
        "model_name",
        "SELF_CORRECTION_mean",
        "SELF_CORRECTION_applicable_runs",
        "SELF_CORRECTION_applicable_rate",
        "CONFABULATION_mean",
        "TSA_mean",
        "execution_gap",
        "TTD_mean",
    ]
    df[summary_cols].sort_values(["experiment_id", "model_name"]).to_csv(
        output_dir / "self_correction_plot_table.csv",
        index=False,
    )
    print(f"Wrote self-correction figures to {output_dir}")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
