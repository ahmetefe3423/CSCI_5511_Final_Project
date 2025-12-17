#!/usr/bin/env python3
"""
plot_utils.py

Utility functions for plotting boxplots from batch_results CSV
using seaborn.

Typical workflow:

1) Run batch experiments:
       python batch_run.py

2) Plot results:
   - From Python:
        from plot_utils import plot_boxplots_from_csv

        plot_boxplots_from_csv(
            csv_path="outputs_batch/batch_results.csv",
            group_by=["purpose", "tour_algo_name"],
            metrics=["simulation.total_distance", "tours.avg_runtime"],
            output_dir="outputs_batch/plots",
            show=False,
            x_axis_label="Experiment purpose | Tour algorithm",
        )

   - Or from the command line (using defaults at the bottom):
        python plot_utils.py

Defaults are configured via the DEFAULT_* constants at the bottom.
"""

from pathlib import Path
from typing import Sequence, Optional, Union, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns


PathLike = Union[str, Path]


def plot_boxplots_from_csv(
    csv_path: PathLike,
    group_by: Sequence[str],
    metrics: Sequence[str],
    output_dir: Optional[PathLike] = None,
    show: bool = True,
    figsize_per_group: float = 1.5,
    x_axis_label: Optional[str] = None,
    seaborn_style: str = "whitegrid",
    palette_name: str = "colorblind",
    title_template: Optional[str] = None,
    y_axis_labels: Optional[Dict[str, str]] = None,
    log_scale: bool = True,
) -> None:
    """
    Read a CSV and make seaborn boxplots for given metrics grouped by given columns.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file (e.g. 'outputs_batch/batch_results.csv').
    group_by : list[str]
        Column(s) to group by. Example: ['purpose', 'tour_algo_name'].
        When multiple columns are given, their values are concatenated
        into a single group label per row, and those labels are used on the x-axis.
    metrics : list[str]
        Numeric columns to plot as boxplots.
        Example: ['simulation.total_distance', 'tours.avg_runtime'].
    output_dir : str or Path or None, default None
        If provided, plots are saved as PDF files in that directory.
        If None, plots are not saved.
    show : bool, default True
        If True, show plots interactively via plt.show().
        If False, figures are created (and possibly saved) and then closed.
    figsize_per_group : float, default 1.5
        Horizontal size per group for the figure width. Width is
        max(6, figsize_per_group * number_of_groups).
    x_axis_label : str or None, default None
        Text to use as the x-axis label. If None, a label is constructed
        as " | ".join(group_by).
    seaborn_style : str, default "whitegrid"
        Style passed to seaborn.set_style, e.g. "whitegrid", "darkgrid".
    palette_name : str, default "colorblind"
        Name of seaborn color palette to use.
    title_template : str or None, default None
        If None: title is "{metric} by {group_by}".
        If not None: used as title_template.format(metric=metric).
    y_axis_labels : dict or None, default None
        Optional mapping from metric name to pretty y-axis labels.
    log_scale : bool, default True
        If True, use log scale on the y-axis (good for runtimes).
        If False, keep y-axis linear.
    """
    TITLE_FONTSIZE = 18
    AXIS_LABEL_FONTSIZE = 16
    LEGEND_FONTSIZE = 11
    MAX_LEGEND_COLS = 10
    LEGEND_LABEL_SPACING = 0.7

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    group_by = list(group_by)
    metrics = list(metrics)

    # Check grouping columns
    for col in group_by:
        if col not in df.columns:
            raise ValueError(
                f"group_by column '{col}' not found in CSV. "
                f"Available columns include: {list(df.columns)[:20]} ..."
            )

    # Check metric columns
    for m in metrics:
        if m not in df.columns:
            raise ValueError(
                f"metric column '{m}' not found in CSV. "
                f"Available columns include: {list(df.columns)[:20]} ..."
            )

    # Build combined group label if multiple group-by columns
    if len(group_by) == 1:
        group_label_col = group_by[0]
    else:
        group_label_col = "__group_label__"
        df[group_label_col] = df[group_by].astype(str).agg(" | ".join, axis=1)

    # Prepare output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(df[group_label_col].unique())
    n_groups = len(categories)
    x_label_text = x_axis_label if x_axis_label is not None else " | ".join(group_by)

    sns.set_style(seaborn_style)
    sns.set_context("paper", font_scale=1.2)

    # Fixed order of categories for stable colors
    palette = sns.color_palette(palette_name, n_colors=n_groups)
    palette_mapping = dict(zip(categories, palette))

    for metric in metrics:
        sub = df[[group_label_col, metric]].dropna()
        if sub.empty:
            print(f"[WARN] No data for metric '{metric}' after dropping NaNs. Skipping.")
            continue

        # ---- print median stats per group (and quartiles) ----
        stats = (
            sub.groupby(group_label_col)[metric]
            .agg(
                n="count",
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            .reindex(categories)  # keep same order as the plot
        )

        print(f"\n[STATS] {metric}")
        print(stats.to_string(float_format=lambda x: f"{x:.4g}"))

        width = max(6.0, figsize_per_group * max(1, n_groups))
        fig, ax = plt.subplots(figsize=(width, 6))

        # seaborn boxplot with stable categories and colors
        ax = sns.boxplot(
            data=sub,
            x=group_label_col,
            y=metric,
            hue=group_label_col,
            order=categories,
            palette=palette_mapping,
            dodge=False,
            ax=ax,
        )

        # Title
        if title_template is None:
            title_text = f"{metric} by {', '.join(group_by)}"
        else:
            title_text = title_template.format(metric=metric)

        ax.set_title(title_text, fontsize=TITLE_FONTSIZE, pad=28)
        ax.set_xlabel(x_label_text, fontsize=AXIS_LABEL_FONTSIZE)

        ax.set_ylabel(
            y_axis_labels.get(metric, metric) if y_axis_labels else metric,
            fontsize=AXIS_LABEL_FONTSIZE,
        )

        plt.xticks(rotation=45, ha="right")

        # Optional log scale on y-axis
        if log_scale:
            ax.set_yscale("log")

        # Legend: top-center, between title and axes, split into columns
        handles = [mpatches.Patch(color=palette_mapping[c], label=c) for c in categories]
        legend_ncol = min(len(handles), MAX_LEGEND_COLS)
        ax.legend(
            handles=handles,
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=legend_ncol,
            frameon=False,
            columnspacing=LEGEND_LABEL_SPACING,
            fontsize=LEGEND_FONTSIZE,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        if output_dir is not None:
            safe_metric = metric.replace(".", "_").replace(" ", "_")
            safe_groups = "_".join(group_by)
            fname = output_dir / f"box_{safe_metric}_by_{safe_groups}.pdf"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved boxplot for '{metric}' to {fname}")

        if show:
            plt.show()
        else:
            plt.close()


# ---------------------------------------------------------------------
# Default behavior when running this file directly
# ---------------------------------------------------------------------
# data_set can be 'pathfinding', 'tours', or 'target_sharing'.
data_set = "pathfinding"

data_set_str = data_set[0].upper() + data_set[1:]

DEFAULT_CSV = "outputs_batch/batch_results.csv"
DEFAULT_GROUP_BY = [f"{data_set}.algorithm"]
DEFAULT_METRICS = [
    f"{data_set}.avg_runtime",
    "simulation.total_distance",
]
DEFAULT_OUTPUT_DIR = "outputs_batch/plots"
DEFAULT_SHOW = False  # set to True for interactive windows
DEFAULT_X_AXIS_LABEL = f"{data_set} algorithm"
DEFAULT_TITLE = f"{data_set_str} Algorithm Comparison"


def _run_with_defaults() -> None:
    """
    Helper used when running this module as a script.
    Uses the DEFAULT_* constants defined above.
    """
    print(f"Reading CSV: {DEFAULT_CSV}")
    print(f"Grouping by: {DEFAULT_GROUP_BY}")
    print(f"Metrics: {DEFAULT_METRICS}")
    print(f"Output dir: {DEFAULT_OUTPUT_DIR!r}, show={DEFAULT_SHOW}")
    print(f"X-axis label: {DEFAULT_X_AXIS_LABEL!r}")

    plot_boxplots_from_csv(
        csv_path=DEFAULT_CSV,
        group_by=DEFAULT_GROUP_BY,
        metrics=DEFAULT_METRICS,
        output_dir=DEFAULT_OUTPUT_DIR,
        show=DEFAULT_SHOW,
        x_axis_label=DEFAULT_X_AXIS_LABEL,
        palette_name="colorblind",
        title_template=DEFAULT_TITLE,
        y_axis_labels={
            "simulation.total_distance": "Total distance",
            f"{data_set}.avg_runtime": "Average runtime (s)",
        },
        log_scale=True,
    )


if __name__ == "__main__":
    _run_with_defaults()
