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

Providers without any recorded latency data are skipped with a warning.
Prints the output path on stdout so callers can embed it.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def load_answers(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def main():
    parser = argparse.ArgumentParser(description="Plot latency distribution across providers.")
    parser.add_argument("files", nargs="+", help="Answers JSON files to plot")
    parser.add_argument("--labels", nargs="+", default=None, help="Display label for each file")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--stat", default="percent", choices=["count", "percent", "density"],
        help="Y-axis statistic (default: percent)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"--labels must have {len(args.files)} entries, got {len(args.labels)}.")

    labels = args.labels or [Path(f).stem for f in args.files]
    output = args.output or "results/erp_qa/latency_distribution.png"

    # ── Load data ──────────────────────────────────────────────────────────────
    series: list[tuple[str, list[float]]] = []
    for path, label in zip(args.files, labels):
        records = load_answers(path)
        latencies = [r["inference_time_secs"] for r in records if r.get("inference_time_secs") is not None]
        if not latencies:
            print(f"WARNING: {label} has no 'inference_time_secs' data — skipping.", file=sys.stderr)
            continue
        series.append((label, latencies))

    if not series:
        print("ERROR: No providers with latency data found.", file=sys.stderr)
        sys.exit(1)

    # ── Plot ───────────────────────────────────────────────────────────────────
    palette = sns.color_palette("tab10", n_colors=len(series))
    stat_label = {"count": "Questions", "percent": "% of Questions", "density": "Density"}[args.stat]

    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, latencies), color in zip(series, palette):
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

    # ── Summary stats in a text box ────────────────────────────────────────────
    lines = []
    for label, latencies in series:
        mean = sum(latencies) / len(latencies)
        sorted_l = sorted(latencies)
        median = sorted_l[len(sorted_l) // 2]
        p90 = sorted_l[int(len(sorted_l) * 0.9)]
        lines.append(f"{label}: mean={mean:.0f}s, median={median:.0f}s, p90={p90:.0f}s, n={len(latencies)}")
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
