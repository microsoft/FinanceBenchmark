"""
plot_scores.py

Grouped bar plots comparing eval scores across providers / runs.

Outputs (to --output-dir, default: results/):
  scores_overall.png         — top-line metrics averaged across all questions
                               (segment-specific metrics pooled into their base metric,
                               e.g. peers&competitivepositioning_depth → depth)
  scores_by_plugin.png       — same, faceted by plugin (erp_qa / finance_qa / ...)
  scores_by_segment.png      — per-segment breakdown for metrics that have segment
                               variants (one facet per base metric, x=segment)

Segment-specific metric detection: a tag name of the form `{segment}_{base}` is
classified as segment-specific when its suffix after the final `_` matches a base
metric (i.e., a metric name that also appears as a plain unsuffixed tag).

Usage:
  uv run scripts/analysis/plot_scores.py file1.json file2.json \\
      --labels "Claude (Sonnet-4.5)" "GPT-5.2" --output-dir results/

Options:
  --labels      One display label per file (default: filename stem)
  --output-dir  Directory for output PNGs (default: results/)
  --top-n-tags  Max number of tag metrics to show alongside overall (default: 7)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_eval_results(path: str) -> list[dict]:
    """Load results from eval JSON, handling both flat-array and envelope formats."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def _detect_base_metrics(all_tag_names: set[str]) -> set[str]:
    """Return the set of base metric names (appear as plain tags without a segment prefix)."""
    return {m for m in all_tag_names if "_" not in m}


def _classify_tag(tag: str, base_metrics: set[str]) -> tuple[str | None, str]:
    """Return (segment, base_metric).

    If the tag's suffix (after the last '_') matches a base metric, it is
    segment-specific and the prefix is the segment name.
    Otherwise the tag itself is the base metric (segment=None).
    """
    if "_" in tag:
        segment, suffix = tag.rsplit("_", 1)
        if suffix in base_metrics:
            return segment, suffix
    return None, tag


def build_dataframe(files: list[str], labels: list[str]) -> pd.DataFrame:
    """Build a long-form DataFrame with columns:
      provider, plugin, metric (raw tag name), base_metric, segment, score
    """
    # First pass: collect all tag names to detect base metrics
    all_tag_names: set[str] = set()
    raw_rows = []
    for path, label in zip(files, labels):
        for r in load_eval_results(path):
            if r.get("skipped"):
                continue
            plugin = r.get("plugin", "unknown")
            overall = r.get("overall_score")
            if overall is not None:
                raw_rows.append({"provider": label, "plugin": plugin,
                                  "metric": "overall", "score": overall})
            for tag, score in (r.get("tag_scores") or {}).items():
                if score is not None:
                    all_tag_names.add(tag)
                    raw_rows.append({"provider": label, "plugin": plugin,
                                      "metric": tag, "score": score})

    base_metrics = _detect_base_metrics(all_tag_names)

    rows = []
    for row in raw_rows:
        tag = row["metric"]
        if tag == "overall":
            segment, base = None, "overall"
        else:
            segment, base = _classify_tag(tag, base_metrics)
        rows.append({**row, "base_metric": base, "segment": segment})

    return pd.DataFrame(rows)


def select_metrics(df: pd.DataFrame, top_n: int = 8) -> list[str]:
    """overall first, then the most-common base metrics up to top_n total."""
    counts = (
        df[(df["base_metric"] != "overall")]
        .groupby("base_metric")["score"]
        .count()
        .sort_values(ascending=False)
    )
    return ["overall"] + list(counts.index[: top_n - 1])


def _metric_label(m: str) -> str:
    return m.replace("_", " ").replace("&", " & ")


def _barplot(ax, data: pd.DataFrame, labels: list[str], palette, order: list[str]) -> None:
    agg = data.groupby(["provider", "metric_label"])["score"].mean().reset_index()
    sns.barplot(
        data=agg,
        x="metric_label",
        y="score",
        hue="provider",
        order=order,
        hue_order=[l for l in labels if l in agg["provider"].unique()],
        palette={l: c for l, c in zip(labels, palette)},
        ax=ax,
    )
    ax.set_ylim(0, 1.09)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=25)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def plot_overall(df: pd.DataFrame, labels: list[str], output: str, top_n: int) -> None:
    metrics = select_metrics(df, top_n)
    # Use base_metric for pooling (e.g. all *_depth → depth)
    plot_df = df[df["base_metric"].isin(metrics)].copy()
    plot_df["metric_label"] = plot_df["base_metric"].map(_metric_label)
    order = [_metric_label(m) for m in metrics]

    palette = sns.color_palette("tab10", n_colors=len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * len(labels) * 0.55 + 2), 5))
    _barplot(ax, plot_df, labels, palette, order)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, ha="right")
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Eval Scores by Provider — all questions", fontsize=13, fontweight="bold")
    ax.legend(title="Provider", fontsize=9, title_fontsize=9, loc="lower right")
    sns.despine()
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_by_plugin(df: pd.DataFrame, labels: list[str], output: str, top_n: int) -> None:
    plugins = sorted(p for p in df["plugin"].unique() if p != "unknown")
    if not plugins:
        print("WARNING: no plugin field found in results — skipping per-plugin plot.", file=sys.stderr)
        return

    metrics = select_metrics(df, top_n)
    plot_df = df[df["base_metric"].isin(metrics) & df["plugin"].isin(plugins)].copy()
    plot_df["metric_label"] = plot_df["base_metric"].map(_metric_label)
    order = [_metric_label(m) for m in metrics]

    palette = sns.color_palette("tab10", n_colors=len(labels))
    n = len(plugins)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(6, len(metrics) * len(labels) * 0.55 + 2) * n, 5),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    for i, (ax, plugin) in enumerate(zip(axes, plugins)):
        plugin_df = plot_df[plot_df["plugin"] == plugin]
        _barplot(ax, plugin_df, labels, palette, order)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, ha="right")
        ax.set_title(plugin.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Score" if i == 0 else "", fontsize=11)
        leg = ax.get_legend()
        if i == 0 and leg:
            leg.set_title("Provider")
            leg._set_loc(3)
        elif leg:
            leg.remove()
        sns.despine(ax=ax)

    fig.suptitle("Eval Scores by Provider and Plugin", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_by_segment(df: pd.DataFrame, labels: list[str], output: str) -> None:
    """One facet per base metric that has segment variants; x=segment, hue=provider."""
    seg_df = df[df["segment"].notna()].copy()
    if seg_df.empty:
        return

    base_metrics_with_segments = sorted(seg_df["base_metric"].unique())
    n = len(base_metrics_with_segments)

    palette = sns.color_palette("tab10", n_colors=len(labels))
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(6, n * (len(labels) * 1.8 + 2)), 5),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    for i, (ax, base) in enumerate(zip(axes, base_metrics_with_segments)):
        base_df = seg_df[seg_df["base_metric"] == base].copy()
        segments = sorted(base_df["segment"].unique())
        seg_labels = [_metric_label(s) for s in segments]
        base_df["segment_label"] = base_df["segment"].map(_metric_label)

        agg = base_df.groupby(["provider", "segment_label"])["score"].mean().reset_index()
        sns.barplot(
            data=agg,
            x="segment_label",
            y="score",
            hue="provider",
            order=seg_labels,
            hue_order=[l for l in labels if l in agg["provider"].unique()],
            palette={l: c for l, c in zip(labels, palette)},
            ax=ax,
        )
        ax.set_ylim(0, 1.09)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(_metric_label(base).title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Score" if i == 0 else "", fontsize=11)
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        leg = ax.get_legend()
        if i == 0 and leg:
            leg.set_title("Provider")
        elif leg:
            leg.remove()
        sns.despine(ax=ax)

    fig.suptitle("Scores by Segment", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval score bar charts across providers.")
    parser.add_argument("files", nargs="+", help="Eval results JSON files to compare")
    parser.add_argument("--labels", nargs="+", default=None, help="Display label for each file")
    parser.add_argument("--output-dir", default="results", help="Directory for output PNGs (default: results/)")
    parser.add_argument(
        "--top-n-tags", type=int, default=7, metavar="N",
        help="Max tag metrics to show alongside 'overall' (default: 7)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"--labels must have {len(args.files)} entries, got {len(args.labels)}.")

    labels = args.labels or [Path(f).stem for f in args.files]
    output_dir = Path(args.output_dir)

    df = build_dataframe(args.files, labels)
    if df.empty:
        print("ERROR: No score data found in the provided files.", file=sys.stderr)
        sys.exit(1)

    plot_overall(df, labels, str(output_dir / "scores_overall.png"), args.top_n_tags + 1)
    plot_by_plugin(df, labels, str(output_dir / "scores_by_plugin.png"), args.top_n_tags + 1)
    plot_by_segment(df, labels, str(output_dir / "scores_by_segment.png"))


if __name__ == "__main__":
    main()
