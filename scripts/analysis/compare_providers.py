"""
compare_providers.py

Compute per-provider statistics and generate comparison plots for the
Finance Copilot benchmark.  Produces:
  - docs/images/scores_overall_weighted.png
  - docs/images/scores_by_plugin.png
  - docs/compare_stats.json

No LLM involvement — all computation is deterministic.

CLI
---
    uv run scripts/analysis/compare_providers.py \\
      [--inf-run-ids <sydney_slug> <claude_slug> <openai_slug>] \\
      [--providers sydney claude openai] \\
      [--diagnose-threshold 0.8] \\
      --output-dir docs/images/ \\
      --stats-output docs/compare_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

_REPO_ROOT = Path(__file__).parent.parent.parent

# Make project modules importable
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.query_runs import query_runs, list_runs

# ---------------------------------------------------------------------------
# Colour palette: Sydney, Claude, OpenAI
# ---------------------------------------------------------------------------
_PALETTE = {
    "sydney": "#4C72B0",
    "claude": "#DD8452",
    "openai": "#55A868",
}


# ---------------------------------------------------------------------------
# Metric detection & pooling
# ---------------------------------------------------------------------------

def _detect_base_metrics(score_cols: list[str]) -> set[str]:
    """Return the set of plain base metric names.

    A name is a plain base metric if:
    - It appears as a standalone column with no underscores (e.g. ``depth_score``), OR
    - Its suffix (after the last ``_``) appears as the suffix of 2+ distinct columns,
      indicating it is used as a segment-qualified metric (e.g. ``peers_depth``,
      ``vendors_depth`` → implied base metric ``depth``).

    This handles the case where a base metric like ``depth`` only ever appears in
    segment-qualified form in a particular run's data.
    """
    explicit: set[str] = set()
    suffix_counts: dict[str, int] = {}

    for col in score_cols:
        name = col[:-6]  # strip _score suffix
        if "_" not in name:
            explicit.add(name)
        else:
            suffix = name.rsplit("_", 1)[1]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    # Treat suffixes that appear in 2+ columns as implied base metrics
    implied = {suffix for suffix, count in suffix_counts.items() if count >= 2}
    return explicit | implied


def _pool_segment_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pool segment-specific score columns into their base metric by taking
    the per-question mean across all segment variants.

    A column ``{segment}_{base}_score`` is segment-specific when its suffix
    (after the final ``_``) matches one of the plain base metric names.

    Returns (df_with_pooled_columns, canonical_metric_list).
    """
    score_cols = [c for c in df.columns if c.endswith("_score") and c != "overall_score"]
    base_metrics = _detect_base_metrics(score_cols)

    # Map each score column to its canonical base metric
    col_to_base: dict[str, str] = {}
    for col in score_cols:
        name = col[:-6]
        if "_" in name:
            suffix = name.rsplit("_", 1)[1]
            if suffix in base_metrics:
                col_to_base[col] = suffix
                continue
        col_to_base[col] = name

    # Group columns by their canonical base metric and compute mean per question
    base_to_cols: dict[str, list[str]] = {}
    for col, base in col_to_base.items():
        base_to_cols.setdefault(base, []).append(col)

    out_df = df.copy()
    canonical_metrics: list[str] = []
    for base, cols in base_to_cols.items():
        pooled_col = f"{base}_score"
        if len(cols) == 1 and cols[0] == pooled_col:
            # Already the canonical column — nothing to do
            pass
        else:
            out_df[pooled_col] = df[cols].mean(axis=1)
        # Drop the segment-specific columns (keep only canonical)
        for col in cols:
            if col != pooled_col and col in out_df.columns:
                out_df.drop(columns=[col], inplace=True)
        canonical_metrics.append(base)

    return out_df, sorted(canonical_metrics)


# ---------------------------------------------------------------------------
# Per-plugin mean scores
# ---------------------------------------------------------------------------

def _compute_plugin_scores(
    df: pd.DataFrame, metrics: list[str]
) -> dict[str, dict[str, float]]:
    """Return {plugin: {metric: mean_score}} from the provider's DataFrame."""
    result: dict[str, dict[str, float]] = {}
    plugins = sorted(p for p in df["plugin"].dropna().unique())
    for plugin in plugins:
        pdata = df[df["plugin"] == plugin]
        scores: dict[str, float] = {}
        for m in metrics:
            col = f"{m}_score"
            if col in pdata.columns:
                vals = pdata[col].dropna()
                if len(vals) > 0:
                    scores[m] = float(vals.mean())
        if scores:
            result[plugin] = scores
    return result


def _compute_weighted_scores(
    plugin_scores: dict[str, dict[str, float]], metrics: list[str]
) -> dict[str, float]:
    """Mean of per-plugin means — same weight per plugin regardless of question count."""
    result: dict[str, float] = {}
    for m in metrics:
        vals = [plugin_scores[p][m] for p in plugin_scores if m in plugin_scores[p]]
        if vals:
            result[m] = float(np.mean(vals))
    return result


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def _wilcoxon_test(
    ref_df: pd.DataFrame, other_df: pd.DataFrame, metric: str
) -> Optional[dict]:
    """Run Wilcoxon signed-rank test on paired per-question scores.

    Returns ``{"statistic": ..., "pvalue": ..., "n_pairs": ...}`` or None
    if fewer than 5 pairs are available.
    """
    col = f"{metric}_score"
    if col not in ref_df.columns or col not in other_df.columns:
        return None

    merged = pd.merge(
        ref_df[["question", col]].rename(columns={col: "ref"}),
        other_df[["question", col]].rename(columns={col: "other"}),
        on="question",
        how="inner",
    ).dropna()

    n = len(merged)
    if n < 5:
        return None

    diffs = merged["ref"].values - merged["other"].values

    if np.all(diffs == 0):
        return {"statistic": 0.0, "pvalue": 1.0, "n_pairs": n}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")

    return {"statistic": float(stat), "pvalue": float(pval), "n_pairs": n}


def _compute_significance(
    provider_dfs: dict[str, pd.DataFrame],
    providers: list[str],
    reference: str,
    metrics: list[str],
) -> dict:
    """Compute Wilcoxon tests for each (non-ref provider, metric) pair.

    Returns structure:
    {
        "overall": {
            metric: {
                "{ref}_vs_{other}": {...}
            }
        },
        "by_plugin": {
            plugin: {
                metric: {
                    "{ref}_vs_{other}": {...}
                }
            }
        }
    }
    """
    result: dict = {"overall": {}, "by_plugin": {}}

    ref_df = provider_dfs.get(reference)
    if ref_df is None:
        return result

    non_ref = [p for p in providers if p != reference and p in provider_dfs]

    # Overall significance
    for m in metrics:
        entry: dict = {}
        for other in non_ref:
            key = f"{reference}_vs_{other}"
            entry[key] = _wilcoxon_test(ref_df, provider_dfs[other], m)
        result["overall"][m] = entry

    # Per-plugin significance
    all_plugins: set[str] = set()
    for df in provider_dfs.values():
        all_plugins.update(p for p in df["plugin"].dropna().unique())

    for plugin in sorted(all_plugins):
        plugin_entry: dict = {}
        ref_plugin = ref_df[ref_df["plugin"] == plugin]
        for m in metrics:
            m_entry: dict = {}
            for other in non_ref:
                other_plugin = provider_dfs[other][provider_dfs[other]["plugin"] == plugin]
                key = f"{reference}_vs_{other}"
                m_entry[key] = _wilcoxon_test(ref_plugin, other_plugin, m)
            plugin_entry[m] = m_entry
        result["by_plugin"][plugin] = plugin_entry

    return result


# ---------------------------------------------------------------------------
# Reliability stats
# ---------------------------------------------------------------------------

def _load_raw_json(rel_path: str) -> tuple[list[dict], dict]:
    """Load a results JSON file, returning (results_list, metadata)."""
    path = _REPO_ROOT / rel_path
    if not path.exists():
        return [], {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("results", []), data.get("metadata") or {}


def _compute_reliability(
    inference_file: Optional[str], eval_file: Optional[str]
) -> dict:
    """Count total/error/sorry entries from raw inference and eval files."""
    stats: dict = {
        "total": 0,
        "error_count": 0,
        "skipped_count": 0,
        "sorry_count": 0,
        "answered_count": 0,
    }

    if inference_file:
        inf_results, _ = _load_raw_json(inference_file)
        stats["total"] = len(inf_results)
        sorry_re = re.compile(r"\bsorry\b", re.IGNORECASE)
        for r in inf_results:
            if r.get("error") is not None:
                stats["error_count"] += 1
            answer = r.get("answer") or ""
            if answer and sorry_re.search(answer):
                stats["sorry_count"] += 1

    if eval_file:
        eval_results, _ = _load_raw_json(eval_file)
        for r in eval_results:
            if r.get("skipped"):
                stats["skipped_count"] += 1

    stats["answered_count"] = (
        stats["total"] - stats["error_count"] - stats["sorry_count"]
    )
    # answered_count should not go below zero
    stats["answered_count"] = max(0, stats["answered_count"])
    return stats


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    df: pd.DataFrame, metric: str, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """95% CI for the weighted mean score of *metric* via bootstrap resampling.

    Resamples questions, then computes mean-of-plugin-means each iteration.
    Returns (lower, upper) percentiles.
    """
    rng = np.random.default_rng(seed)
    col = f"{metric}_score"
    if col not in df.columns:
        return (0.0, 0.0)

    sub = df[["question", "plugin", col]].dropna()
    if sub.empty:
        return (0.0, 0.0)

    questions = sub["question"].values
    n = len(questions)
    boot_means: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = sub.iloc[idx]
        # Compute mean-of-plugin-means
        plugin_means = sample.groupby("plugin")[col].mean()
        boot_means.append(float(plugin_means.mean()))

    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_bracket(ax, x1: float, x2: float, y_top: float, pval, bar_h: float = 0.015) -> None:
    """Draw a significance bracket above bars. Skipped if pval is None or >= 0.05."""
    if pval is None or pval >= 0.05:
        return
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
    ax.plot([x1, x1, x2, x2], [y_top, y_top + bar_h, y_top + bar_h, y_top], lw=1, c="black")
    ax.text(
        (x1 + x2) / 2, y_top + bar_h * 1.1, stars,
        ha="center", va="bottom", fontsize=8
    )


def _provider_color(provider: str, providers: list[str]) -> str:
    if provider in _PALETTE:
        return _PALETTE[provider]
    # Fallback to seaborn default cycle
    palette = sns.color_palette("Set2", n_colors=len(providers))
    idx = providers.index(provider) if provider in providers else 0
    r, g, b = palette[idx % len(palette)]
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _plot_overall(
    provider_dfs: dict[str, pd.DataFrame],
    providers: list[str],
    metrics: list[str],
    significance: dict,
    output_path: str,
) -> None:
    """Plot A: weighted overall scores with bootstrap CI and significance brackets."""
    n_metrics = len(metrics)
    n_providers = len(providers)
    fig_w = max(8, n_metrics * n_providers * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bar_width = 0.8 / n_providers
    x = np.arange(n_metrics)

    # Precompute weighted scores and CIs
    weighted: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple]] = {}
    for p in providers:
        if p not in provider_dfs:
            continue
        df = provider_dfs[p]
        plugin_scores = _compute_plugin_scores(df, metrics)
        weighted[p] = _compute_weighted_scores(plugin_scores, metrics)
        cis[p] = {}
        for m in metrics:
            cis[p][m] = _bootstrap_ci(df, m)

    # Draw bars
    bar_positions: dict[str, list[float]] = {}
    for pi, p in enumerate(providers):
        if p not in weighted:
            continue
        offsets = x + (pi - n_providers / 2 + 0.5) * bar_width
        bar_positions[p] = list(offsets)
        heights = [weighted[p].get(m, 0.0) for m in metrics]
        yerr_lower = [weighted[p].get(m, 0.0) - cis[p][m][0] for m in metrics]
        yerr_upper = [cis[p][m][1] - weighted[p].get(m, 0.0) for m in metrics]
        color = _provider_color(p, providers)
        bars = ax.bar(
            offsets, heights,
            width=bar_width * 0.9,
            color=color,
            label=p.capitalize(),
            yerr=[yerr_lower, yerr_upper],
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(yerr_upper) + 0.01 if h > 0 else 0.01,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    # Significance brackets
    reference = providers[0] if providers else None
    non_ref = [p for p in providers[1:] if p in bar_positions and reference in bar_positions]

    sig_overall = significance.get("overall", {})
    for mi, m in enumerate(metrics):
        bar_tops = []
        for p in providers:
            if p in bar_positions and p in weighted:
                h = weighted[p].get(m, 0.0)
                ci_up = cis[p][m][1]
                bar_tops.append(ci_up)
        max_top = max(bar_tops) if bar_tops else 0.0
        margin = 0.04
        step = 0.06

        for ki, other in enumerate(non_ref):
            key = f"{reference}_vs_{other}"
            sig_entry = sig_overall.get(m, {}).get(key)
            pval = sig_entry["pvalue"] if sig_entry else None
            x1 = bar_positions[reference][mi]
            x2 = bar_positions[other][mi]
            y_top = max_top + margin + ki * step
            _draw_bracket(ax, x1, x2, y_top, pval)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").capitalize() for m in metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall weighted scores (avg of plugin averages)")
    ax.legend(title="Provider", fontsize=9, loc="lower right")
    sns.despine()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_by_plugin(
    provider_dfs: dict[str, pd.DataFrame],
    providers: list[str],
    metrics: list[str],
    significance: dict,
    output_path: str,
) -> None:
    """Plot B: per-plugin grouped bar charts, one row of subplots per plugin."""
    all_plugins: list[str] = sorted({
        p
        for df in provider_dfs.values()
        for p in df["plugin"].dropna().unique()
    })
    if not all_plugins:
        print("  WARNING: no plugin data found — skipping scores_by_plugin.png", file=sys.stderr)
        return

    n_plugins = len(all_plugins)
    n_metrics = len(metrics)
    n_providers = len(providers)
    bar_width = 0.8 / n_providers
    x = np.arange(n_metrics)

    fig, axes = plt.subplots(
        n_plugins, 1,
        figsize=(max(8, n_metrics * n_providers * 0.55 + 2), 4 * n_plugins),
        squeeze=False,
    )

    for ri, plugin in enumerate(all_plugins):
        ax = axes[ri][0]
        bar_positions: dict[str, list[float]] = {}

        for pi, p in enumerate(providers):
            if p not in provider_dfs:
                continue
            df = provider_dfs[p][provider_dfs[p]["plugin"] == plugin]
            if df.empty:
                continue
            offsets = x + (pi - n_providers / 2 + 0.5) * bar_width
            bar_positions[p] = list(offsets)
            heights = []
            for m in metrics:
                col = f"{m}_score"
                vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
                heights.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            color = _provider_color(p, providers)
            bars = ax.bar(
                offsets, heights,
                width=bar_width * 0.9,
                color=color,
                label=p.capitalize(),
            )
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.01,
                        f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        # Significance brackets
        reference = providers[0] if providers else None
        non_ref = [p for p in providers[1:] if p in bar_positions and reference in bar_positions]
        sig_plugin = significance.get("by_plugin", {}).get(plugin, {})

        for mi, m in enumerate(metrics):
            bar_tops = []
            for p in providers:
                if p in bar_positions and p in provider_dfs:
                    df = provider_dfs[p][provider_dfs[p]["plugin"] == plugin]
                    col = f"{m}_score"
                    if col in df.columns:
                        vals = df[col].dropna()
                        bar_tops.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            max_top = max(bar_tops) if bar_tops else 0.0
            margin = 0.04
            step = 0.06

            for ki, other in enumerate(non_ref):
                key = f"{reference}_vs_{other}"
                sig_entry = sig_plugin.get(m, {}).get(key)
                pval = sig_entry["pvalue"] if sig_entry else None
                x1 = bar_positions[reference][mi]
                x2 = bar_positions[other][mi]
                y_top = max_top + margin + ki * step
                _draw_bracket(ax, x1, x2, y_top, pval)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").capitalize() for m in metrics],
            rotation=20, ha="right",
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(plugin.replace("_", " ").title())
        sns.despine(ax=ax)

    # Shared legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title="Provider",
            loc="upper center",
            ncol=n_providers,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------

def _resolve_run(
    provider: str,
    inf_slug: Optional[str],
    eval_slug: Optional[str] = None,
) -> Optional[tuple[pd.DataFrame, dict]]:
    """Return (df, run_info_dict) for the given provider/slugs, or None if not found.

    run_info_dict contains: inf_run_id, eval_run_id, model, judge, eval_file, inference_file.
    """
    if inf_slug:
        df = query_runs(inf_run_id=inf_slug, eval_run_id=eval_slug)
        if df.empty:
            slug_desc = f"{inf_slug}_{eval_slug}" if eval_slug else inf_slug
            print(f"  WARNING: no run found for slug '{slug_desc}' (provider={provider})", file=sys.stderr)
            return None
        df = df[df["provider"] == provider] if "provider" in df.columns else df
    else:
        runs = list_runs(provider=provider)
        # list_runs returns rows with eval_file; prefer runs that have eval data
        runs_with_eval = runs[runs["eval_file"].notna()] if not runs.empty and "eval_file" in runs.columns else runs
        if runs_with_eval.empty:
            if runs.empty:
                print(f"  WARNING: no registered runs found for provider '{provider}'", file=sys.stderr)
                return None
            runs_with_eval = runs  # fall back to all runs

        latest = runs_with_eval.iloc[0]
        inf_run_id = latest.get("inf_run_id")
        if not inf_run_id:
            print(f"  WARNING: latest run for '{provider}' has no inf_run_id", file=sys.stderr)
            return None
        df = query_runs(inf_run_id=inf_run_id)
        if df.empty:
            print(f"  WARNING: query_runs returned empty for '{provider}' slug '{inf_run_id}'", file=sys.stderr)
            return None
        df = df[df["provider"] == provider] if "provider" in df.columns else df

    if df.empty:
        print(f"  WARNING: no rows for provider '{provider}' after filtering", file=sys.stderr)
        return None

    # Extract run info from the first row
    first = df.iloc[0]
    run_info = {
        "inf_run_id": str(first.get("inf_run_id", "")),
        "eval_run_id": str(first.get("eval_run_id", "")),
        "model": str(first.get("model", "")),
        "judge": str(first.get("judge", "")),
        "eval_file": str(first.get("eval_file", "")),
        "inference_file": str(first.get("inference_file", "")),
    }
    return df, run_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute provider comparison statistics and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--providers", nargs="+", default=["sydney", "claude", "openai"],
        metavar="PROVIDER",
        help="Providers to compare, in order. First is the reference. (default: sydney claude openai)",
    )
    parser.add_argument(
        "--inf-run-ids", nargs="+", default=None, dest="inf_run_ids",
        metavar="SLUG",
        help="Inference run-ID slugs, one per provider in the same order as --providers.",
    )
    parser.add_argument(
        "--eval-run-ids", nargs="+", default=None, dest="eval_run_ids",
        metavar="SLUG",
        help="Eval run-ID slugs, one per provider (optional; pins evaluation run when multiple evals exist for the same inference).",
    )
    parser.add_argument(
        "--diagnose-threshold", type=float, default=0.8, dest="diagnose_threshold",
        metavar="T",
        help="(plugin, metric) cells where reference score >= T are excluded from diagnose_cells. (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir", default="docs/images", dest="output_dir",
        metavar="DIR",
        help="Directory for output PNGs. (default: docs/images/)",
    )
    parser.add_argument(
        "--stats-output", default="docs/compare_stats.json", dest="stats_output",
        metavar="FILE",
        help="Path for output JSON stats. (default: docs/compare_stats.json)",
    )
    args = parser.parse_args()

    providers: list[str] = args.providers
    reference = providers[0]

    inf_slugs: dict[str, Optional[str]] = {}
    eval_slugs: dict[str, Optional[str]] = {}
    if args.inf_run_ids:
        if len(args.inf_run_ids) != len(providers):
            parser.error(
                f"--inf-run-ids must have {len(providers)} values, got {len(args.inf_run_ids)}"
            )
        for p, s in zip(providers, args.inf_run_ids):
            inf_slugs[p] = s
    else:
        for p in providers:
            inf_slugs[p] = None
    if args.eval_run_ids:
        if len(args.eval_run_ids) != len(providers):
            parser.error(
                f"--eval-run-ids must have {len(providers)} values, got {len(args.eval_run_ids)}"
            )
        for p, s in zip(providers, args.eval_run_ids):
            eval_slugs[p] = s

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_output = Path(args.stats_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Resolve runs
    # ------------------------------------------------------------------
    print("Resolving runs...")
    provider_dfs: dict[str, pd.DataFrame] = {}
    run_info_map: dict[str, dict] = {}

    for p in providers:
        print(f"  {p}...", end=" ")
        result = _resolve_run(p, inf_slugs.get(p), eval_slugs.get(p))
        if result is None:
            print("skipped.")
            continue
        df, run_info = result
        # Pool segment-specific metrics
        df, _ = _pool_segment_metrics(df)
        provider_dfs[p] = df
        run_info_map[p] = run_info
        n_q = df["question"].nunique() if "question" in df.columns else len(df)
        print(f"  {n_q} questions (inf_run_id={run_info['inf_run_id']})")

    if not provider_dfs:
        print("ERROR: no provider data found. Register runs first.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Detect canonical metrics (union across all providers)
    # ------------------------------------------------------------------
    all_score_cols: set[str] = set()
    for df in provider_dfs.values():
        score_cols = [c for c in df.columns if c.endswith("_score") and c != "overall_score"]
        all_score_cols.update(score_cols)

    # Only keep canonical (non-segment-specific) columns
    canonical_metrics: list[str] = sorted(
        col[:-6] for col in all_score_cols
        if "_" not in col[:-6]  # strip _score and check no underscore
    )
    print(f"Canonical metrics: {canonical_metrics}")

    # ------------------------------------------------------------------
    # Step 3 & 4: Per-plugin and weighted scores
    # ------------------------------------------------------------------
    print("Computing scores...")
    plugin_scores: dict[str, dict[str, dict[str, float]]] = {}
    weighted_scores: dict[str, dict[str, float]] = {}

    for p, df in provider_dfs.items():
        ps = _compute_plugin_scores(df, canonical_metrics)
        plugin_scores[p] = ps
        weighted_scores[p] = _compute_weighted_scores(ps, canonical_metrics)

    # ------------------------------------------------------------------
    # Step 5: Significance tests
    # ------------------------------------------------------------------
    print("Running significance tests...")
    significance = _compute_significance(
        provider_dfs, providers, reference, canonical_metrics
    )

    # ------------------------------------------------------------------
    # Step 6: Reliability stats
    # ------------------------------------------------------------------
    print("Computing reliability stats...")
    reliability: dict[str, dict] = {}
    for p, run_info in run_info_map.items():
        reliability[p] = _compute_reliability(
            run_info.get("inference_file") or None,
            run_info.get("eval_file") or None,
        )

    # ------------------------------------------------------------------
    # Step 7: Plots
    # ------------------------------------------------------------------
    print("Generating plots...")
    overall_plot = str(output_dir / "scores_overall_weighted.png")
    by_plugin_plot = str(output_dir / "scores_by_plugin.png")

    _plot_overall(provider_dfs, providers, canonical_metrics, significance, overall_plot)
    _plot_by_plugin(provider_dfs, providers, canonical_metrics, significance, by_plugin_plot)

    # ------------------------------------------------------------------
    # Step 8: Diagnose cells
    # ------------------------------------------------------------------
    ref_plugin_scores = plugin_scores.get(reference, {})
    diagnose_cells: list[dict] = []
    for plugin, metric_map in ref_plugin_scores.items():
        for metric, score in metric_map.items():
            if score < args.diagnose_threshold:
                diagnose_cells.append({
                    "plugin": plugin,
                    "metric": metric,
                    f"{reference}_score": round(score, 4),
                })
    diagnose_cells.sort(key=lambda c: c[f"{reference}_score"])

    # ------------------------------------------------------------------
    # Assemble and write stats JSON
    # ------------------------------------------------------------------
    stats = {
        "providers": providers,
        "reference_provider": reference,
        "metrics": canonical_metrics,
        "run_info": run_info_map,
        "weighted_scores": {
            p: {m: round(v, 4) for m, v in weighted_scores[p].items()}
            for p in providers if p in weighted_scores
        },
        "plugin_scores": {
            p: {
                plugin: {m: round(v, 4) for m, v in ms.items()}
                for plugin, ms in plugin_scores[p].items()
            }
            for p in providers if p in plugin_scores
        },
        "significance": significance,
        "reliability": reliability,
        "diagnose_cells": diagnose_cells,
        "plots": {
            "overall": overall_plot,
            "by_plugin": by_plugin_plot,
        },
    }

    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {stats_output}")


if __name__ == "__main__":
    main()
