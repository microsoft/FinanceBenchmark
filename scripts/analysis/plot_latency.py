"""
plot_latency.py

Plots the distribution of inference latency (seconds) per question across one or more providers.

Usage:
  uv run scripts/erp_qa/plot_latency.py <answers_file1> [answers_file2 ...]
  uv run scripts/erp_qa/plot_latency.py results/erp_qa/answers_claude-sonnet-4-5.json results/erp_qa/answers_gpt5p2.json --labels "Claude Code (Sonnet-4.5)" "OpenAI (GPT-5.2)"

Options:
  --labels     Display labels for each file (defaults to filename stem)
  --output     Output PNG path (default: results/erp_qa/latency_distribution.png)
  --stat       Statistic for y-axis: count | percent | density (default: percent)
  --by-plugin  Facet by plugin; one subplot per plugin, all providers overlaid within each
  --timeout    Exclude entries with inference_time_secs above this value (default: 60)

Providers without any recorded latency data are skipped with a warning.
Prints the output path on stdout so callers can embed it.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PLUGIN_DISPLAY_NAMES = {
    "erp_qa": "Entity Financial Obligations",
    "finance_qa": "Entity Financial Performance",
    "business_brief": "Finance Business Briefs",
}


def load_answers(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def _stats_box_text(label: str, latencies: list[float]) -> str:
    mean = sum(latencies) / len(latencies)
    sorted_l = sorted(latencies)
    median = sorted_l[len(sorted_l) // 2]
    p90 = sorted_l[int(len(sorted_l) * 0.9)]
    return f"{label}: mean={mean:.0f}s, median={median:.0f}s, p90={p90:.0f}s, n={len(latencies)}"


def main():
    parser = argparse.ArgumentParser(description="Plot latency distribution across providers.")
    parser.add_argument("files", nargs="+", help="Answers JSON files to plot")
    parser.add_argument("--labels", nargs="+", default=None, help="Display label for each file")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--stat", default="percent", choices=["count", "percent", "density"],
        help="Y-axis statistic (default: percent)",
    )
    parser.add_argument(
        "--by-plugin", action="store_true", default=False,
        help="Facet by plugin; one subplot per plugin with all providers overlaid",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Exclude entries with inference_time_secs above this value (default: 60)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"--labels must have {len(args.files)} entries, got {len(args.labels)}.")

    labels = args.labels or [Path(f).stem for f in args.files]
    output = args.output or (
        "results/images/latency_by_plugin.png" if args.by_plugin
        else "results/erp_qa/latency_distribution.png"
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    # series: list of (label, list of (latency_float, plugin_str))
    series: list[tuple[str, list[tuple[float, str]]]] = []
    for path, label in zip(args.files, labels):
        records = load_answers(path)
        pairs = [
            (r["inference_time_secs"], r.get("plugin", "unknown"))
            for r in records
            if r.get("inference_time_secs") is not None
        ]
        if not pairs:
            print(f"WARNING: {label} has no 'inference_time_secs' data — skipping.", file=sys.stderr)
            continue
        filtered = [(v, p) for v, p in pairs if v <= args.timeout]
        n_excluded = len(pairs) - len(filtered)
        if n_excluded:
            print(
                f"WARNING: {label}: excluded {n_excluded}/{len(pairs)} entries "
                f"with inference_time_secs > {args.timeout}s.",
                file=sys.stderr,
            )
        if not filtered:
            print(f"WARNING: {label} has no entries within timeout threshold — skipping.", file=sys.stderr)
            continue
        series.append((label, filtered))

    if not series:
        print("ERROR: No providers with latency data found.", file=sys.stderr)
        sys.exit(1)

    # ── Plot ───────────────────────────────────────────────────────────────────
    palette = sns.color_palette("tab10", n_colors=len(series))
    stat_label = {"count": "Questions", "percent": "% of Questions", "density": "Density"}[args.stat]

    if args.by_plugin:
        # Collect unique plugins across all providers (exclude "unknown" from facet list)
        plugins_found = sorted({
            plugin
            for _, pairs in series
            for _, plugin in pairs
            if plugin and plugin != "unknown"
        })

        if not plugins_found:
            print(
                "WARNING: --by-plugin requested but no plugin labels found in data — "
                "falling back to single-panel plot.",
                file=sys.stderr,
            )
            args.by_plugin = False
        else:
            fig, axes = plt.subplots(
                1, len(plugins_found),
                figsize=(9 * len(plugins_found), 5),
                squeeze=False,
            )

            for plugin, ax in zip(plugins_found, axes[0]):
                for (label, pairs), color in zip(series, palette):
                    plugin_latencies = [v for v, p in pairs if p == plugin]
                    if not plugin_latencies:
                        continue
                    sns.histplot(
                        plugin_latencies,
                        ax=ax,
                        label=label,
                        color=color,
                        stat=args.stat,
                        alpha=0.55,
                        edgecolor="white",
                        linewidth=0.6,
                        bins=20,
                    )

                # Stats box per subplot
                lines = []
                for label, pairs in series:
                    plugin_latencies = [v for v, p in pairs if p == plugin]
                    if plugin_latencies:
                        lines.append(_stats_box_text(label, plugin_latencies))
                if lines:
                    ax.text(
                        0.98, 0.97, "\n".join(lines),
                        transform=ax.transAxes,
                        fontsize=8.5,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7, edgecolor="lightgrey"),
                    )

                ax.set_title(PLUGIN_DISPLAY_NAMES.get(plugin, plugin), fontsize=12, fontweight="bold")
                ax.set_xlabel("Inference Time (seconds)", fontsize=11)
                ax.set_ylabel(stat_label, fontsize=11)
                sns.despine(ax=ax)

            # Single shared legend above the figure
            handles, labels_l = axes[0][0].get_legend_handles_labels()
            fig.legend(
                handles, labels_l,
                title="Provider",
                loc="upper center",
                ncol=len(series),
                bbox_to_anchor=(0.5, 1.02),
            )

    # Fallthrough: either --by-plugin was not set, or it was reset to False above
    if not args.by_plugin:
        fig, ax = plt.subplots(figsize=(9, 5))

        for (label, pairs), color in zip(series, palette):
            latencies = [v for v, _ in pairs]
            sns.histplot(
                latencies,
                ax=ax,
                label=label,
                color=color,
                stat=args.stat,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.6,
                bins=20,
            )

        ax.set_xlabel("Inference Time (seconds)", fontsize=12)
        ax.set_ylabel(stat_label, fontsize=12)
        ax.set_title("Distribution of Inference Latency per Question", fontsize=13, fontweight="bold")
        ax.legend(title="Provider", fontsize=10, title_fontsize=10)
        sns.despine()

        # Summary stats text box
        lines = []
        for label, pairs in series:
            latencies = [v for v, _ in pairs]
            lines.append(_stats_box_text(label, latencies))
        ax.text(
            0.98, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7, edgecolor="lightgrey"),
        )

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
