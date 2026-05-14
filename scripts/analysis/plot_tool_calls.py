"""
plot_tool_calls.py

Plots the distribution of tool calls per question across one or more providers.

Usage:
  uv run scripts/erp_qa/plot_tool_calls.py <answers_file1> [answers_file2 ...]
  uv run scripts/erp_qa/plot_tool_calls.py results/erp_qa/answers_claude-opus-4-6.json results/erp_qa/answers_gpt5p2.json --labels "Claude (Opus-4.6)" "OpenAI (GPT-5.2)"

Options:
  --labels     Display labels for each file (defaults to filename stem)
  --output     Output PNG path (default: results/erp_qa/tool_calls_distribution.png)
  --stat       Statistic for y-axis: count | percent | density (default: percent)
  --field      Which count field to plot: tool_call_count | successful_tool_calls (default: tool_call_count)
  --by-plugin  Facet by plugin; one subplot per plugin with all providers overlaid

Providers without any recorded tool call data (e.g. scraped answers) are skipped with a warning.
Prints the output path on stdout so callers can embed it.
"""

import argparse
import json
import statistics
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


def main():
    parser = argparse.ArgumentParser(description="Plot tool-call distribution across providers.")
    parser.add_argument("files", nargs="+", help="Answers JSON files to plot")
    parser.add_argument("--labels", nargs="+", default=None, help="Display label for each file")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--stat", default="percent", choices=["count", "percent", "density"],
        help="Y-axis statistic (default: percent)",
    )
    parser.add_argument(
        "--field", default="tool_call_count",
        choices=["tool_call_count", "successful_tool_calls"],
        help="Which count field to plot (default: tool_call_count)",
    )
    parser.add_argument(
        "--by-plugin", action="store_true", default=False,
        help="Facet by plugin; one subplot per plugin with all providers overlaid",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"--labels must have {len(args.files)} entries, got {len(args.labels)}.")

    labels = args.labels or [Path(f).stem for f in args.files]

    # ── Load data ──────────────────────────────────────────────────────────────
    # series stores (label, list_of_(count, plugin)) pairs
    series: list[tuple[str, list[tuple[int, str]]]] = []
    for path, label in zip(args.files, labels):
        records = load_answers(path)
        pairs = [
            (r[args.field], r.get("plugin", "unknown"))
            for r in records
            if r.get(args.field) is not None
        ]
        if not pairs:
            print(f"WARNING: {label} has no '{args.field}' data — skipping.", file=sys.stderr)
            continue
        series.append((label, pairs))

    if not series:
        print("ERROR: No providers with tool call data found.", file=sys.stderr)
        sys.exit(1)

    field_title = "Tool Calls" if args.field == "tool_call_count" else "Successful Tool Calls"
    stat_label = {"count": "Questions", "percent": "% of Questions", "density": "Density"}[args.stat]

    palette = sns.color_palette("tab10", n_colors=len(series))

    # ── By-plugin faceted path ─────────────────────────────────────────────────
    if args.by_plugin:
        plugins_found = sorted({
            plugin
            for _, pairs in series
            for _, plugin in pairs
            if plugin and plugin != "unknown"
        })

        if not plugins_found:
            print(
                "WARNING: No plugin data found in records; falling back to single-panel plot.",
                file=sys.stderr,
            )
            args.by_plugin = False  # fall through to single-panel below
        else:
            output = args.output or "results/images/tool_calls_by_plugin.png"

            fig, axes = plt.subplots(
                1, len(plugins_found),
                figsize=(9 * len(plugins_found), 5),
                squeeze=False,
            )

            for plugin, ax in zip(plugins_found, axes[0]):
                stats_lines = []
                for (label, pairs), color in zip(series, palette):
                    counts = [v for v, p in pairs if p == plugin]
                    if not counts:
                        continue
                    sns.histplot(
                        counts,
                        ax=ax,
                        label=label,
                        color=color,
                        discrete=True,
                        stat=args.stat,
                        alpha=0.55,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                    mean = sum(counts) / len(counts)
                    median = statistics.median(counts)
                    stats_lines.append(
                        f"{label}: mean={mean:.1f}, median={median}, n={len(counts)}"
                    )

                ax.set_title(PLUGIN_DISPLAY_NAMES.get(plugin, plugin), fontsize=12, fontweight="bold")
                ax.set_xlabel(f"Number of {field_title} per Question", fontsize=11)
                ax.set_ylabel(stat_label, fontsize=11)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                sns.despine(ax=ax)

                if stats_lines:
                    ax.text(
                        0.98, 0.97, "\n".join(stats_lines),
                        transform=ax.transAxes,
                        fontsize=8.5,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            alpha=0.7,
                            edgecolor="lightgrey",
                        ),
                    )

            # Shared legend above the subplots
            handles, labels_l = axes[0][0].get_legend_handles_labels()
            fig.legend(
                handles, labels_l,
                title="Provider",
                loc="upper center",
                ncol=len(series),
                bbox_to_anchor=(0.5, 1.02),
            )

            plt.tight_layout()
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(output)
            return

    # ── Single-panel path (original behaviour) ─────────────────────────────────
    output = args.output or "results/erp_qa/tool_calls_distribution.png"

    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, pairs), color in zip(series, palette):
        counts = [v for v, _ in pairs]
        sns.histplot(
            counts,
            ax=ax,
            label=label,
            color=color,
            discrete=True,
            stat=args.stat,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xlabel(f"Number of {field_title} per Question", fontsize=12)
    ax.set_ylabel(stat_label, fontsize=12)
    ax.set_title(f"Distribution of {field_title} per Question", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(title="Provider", fontsize=10, title_fontsize=10)
    sns.despine()

    # ── Summary stats in a text box ────────────────────────────────────────────
    lines = []
    for label, pairs in series:
        counts = [v for v, _ in pairs]
        mean = sum(counts) / len(counts)
        median = statistics.median(counts)
        lines.append(f"{label}: mean={mean:.1f}, median={median}, n={len(counts)}")
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
