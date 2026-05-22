"""
compare_runs.py

Compute statistics and generate comparison plots across benchmark runs.
Runs can differ by provider, model, or configuration.  Produces:
  - results/images/scores_overall_weighted.png
  - results/images/scores_by_plugin.png
  - results/compare_stats.json

No LLM involvement — all computation is deterministic.

CLI
---
    uv run scripts/analysis/compare_runs.py \\
      [--inf-run-ids <finance_agent_slug> <claude_slug> <openai_slug>] \\
      [--runs finance_agent claude openai] \\
      [--diagnose-threshold 1.0] \\
      --output-dir results/images/ \\
      --stats-output results/compare_stats.json
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
from scipy.stats import norm as scipy_norm
import statsmodels.formula.api as smf

_REPO_ROOT = Path(__file__).parent.parent.parent

# Make project modules importable
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tracking.runs import query_runs, list_runs

# ---------------------------------------------------------------------------
# Colour palette and display names
# ---------------------------------------------------------------------------
_PALETTE = {
    "finance_agent": "#4C72B0",
    "claude": "#DD8452",
    "openai": "#55A868",
}

DISPLAY_NAMES = {
    "finance_agent": "Finance Agent",
    "claude": "Claude",
    "openai": "OpenAI",
}

PLUGIN_DISPLAY_NAMES = {
    "erp_qa": "Entity Financial Obligations",
    "finance_qa": "Entity Financial Performance",
    "business_brief": "Finance Business Briefs",
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

def _build_lmm_contrasts(
    param_names: list[str],
    other_provider: str,
    non_ref_plugins: list[str],
    target_plugin: Optional[str] = None,
) -> np.ndarray:
    """Build a contrast vector for MixedLMResults.t_test().

    If target_plugin is None: equal-weighted marginal mean across all plugins
    (1/k weight per plugin, preserving the same equal-plugin weighting as the
    weighted scores). If target_plugin is set: plugin-specific provider effect.

    Derivation for the overall contrast:
        equal_avg(other - ref) = (1/k) * sum_j [β_other + β_other:plugin_j]
                               = β_other + (1/k) * sum_{j≠ref} β_other:plugin_j
    So: weight on β_other = 1.0; weight on each interaction term = 1/k.
    """
    param_idx = {name: i for i, name in enumerate(param_names)}
    c = np.zeros(len(param_names))

    k_total = len(non_ref_plugins) + 1  # total plugin count (ref + non-ref)

    main_key = f"C(provider)[T.{other_provider}]"
    if main_key in param_idx:
        c[param_idx[main_key]] = 1.0

    if target_plugin is None:
        for plugin in non_ref_plugins:
            int_key = f"C(provider)[T.{other_provider}]:C(plugin)[T.{plugin}]"
            if int_key in param_idx:
                c[param_idx[int_key]] = 1.0 / k_total
    elif target_plugin in non_ref_plugins:
        int_key = f"C(provider)[T.{other_provider}]:C(plugin)[T.{target_plugin}]"
        if int_key in param_idx:
            c[param_idx[int_key]] = 1.0

    return c


def _lmm_metric_test(
    combined_df: pd.DataFrame,
    metric: str,
    ref_provider: str,
    runs: list[str],
    ref_plugin: str,
) -> dict:
    """Fit one LMM for a metric on all data and extract all provider contrasts.

    Model: score ~ C(provider) * C(plugin) + (1 | question)

    Overall provider effects use an equal-weighted marginal mean contrast
    (each plugin contributes equally, matching the weighted score aggregation).
    Per-plugin effects use interaction contrasts from the same fitted model.

    Returns {"overall": {...}, "by_plugin": {plugin: {...}}} or {} on failure.
    """
    col = f"{metric}_score"
    if col not in combined_df.columns:
        return {}

    present_providers = [p for p in runs if p in combined_df["provider"].unique()]
    non_ref_providers = [p for p in present_providers if p != ref_provider]
    if ref_provider not in present_providers or not non_ref_providers:
        return {}

    sub = combined_df[combined_df["provider"].isin(present_providers)][
        ["question", "provider", "plugin", col]
    ].dropna().copy()

    for p in present_providers:
        if (sub["provider"] == p).sum() < 5:
            return {}

    if sub[col].std() == 0.0:
        trivial = {"statistic": 0.0, "pvalue": 1.0, "n_pairs": len(sub), "direction": 0, "lmm_beta": 0.0, "lmm_se": 0.0}
        result: dict = {"overall": {}, "by_plugin": {}}
        for other in non_ref_providers:
            key = f"{ref_provider}_vs_{other}"
            result["overall"][key] = trivial.copy()
            for plugin in sub["plugin"].unique():
                result["by_plugin"].setdefault(plugin, {})[key] = trivial.copy()
        return result

    all_plugins = sorted(sub["plugin"].dropna().unique())
    ref_plugin_actual = ref_plugin if ref_plugin in all_plugins else all_plugins[0]
    non_ref_plugins = [p for p in all_plugins if p != ref_plugin_actual]

    sub["provider"] = pd.Categorical(
        sub["provider"], categories=[ref_provider] + [p for p in present_providers if p != ref_provider]
    )
    sub["plugin"] = pd.Categorical(
        sub["plugin"], categories=[ref_plugin_actual] + non_ref_plugins
    )

    use_interaction = len(all_plugins) >= 2
    formula = f"{col} ~ C(provider) * C(plugin)" if use_interaction else f"{col} ~ C(provider)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(formula, sub, groups=sub["question"])
            fit = model.fit(reml=False)
            if not getattr(fit, "converged", True):
                return {}
        except Exception:
            return {}

    param_names = list(fit.params.index)
    params = fit.params.values
    # cov_params() returns a DataFrame for MixedLM; extract numpy array
    cov = fit.cov_params()
    cov_arr = cov.values if hasattr(cov, "values") else np.array(cov)

    n_total = len(sub)
    result = {"overall": {}, "by_plugin": {plugin: {} for plugin in all_plugins}}

    def _contrast_stats(c: np.ndarray, n_obs: int) -> dict:
        """Compute z, p, beta, se from a contrast vector and model fit."""
        beta = float(c @ params)
        var = float(c @ cov_arr @ c)
        se = float(np.sqrt(max(var, 0.0)))
        z = beta / se if se > 1e-10 else 0.0
        pval = float(2.0 * scipy_norm.sf(abs(z)))
        return {
            "statistic": round(z, 4),
            "pvalue": float(pval),
            "n_pairs": n_obs,
            "direction": 1 if beta <= 0 else -1,
            "lmm_beta": round(beta, 4),
            "lmm_se": round(se, 4),
        }

    for other in non_ref_providers:
        key = f"{ref_provider}_vs_{other}"
        nrp = non_ref_plugins if use_interaction else []

        # Overall: equal-weighted marginal contrast across plugins
        c_overall = _build_lmm_contrasts(param_names, other, nrp, target_plugin=None)
        result["overall"][key] = _contrast_stats(c_overall, n_total)

        # By-plugin: interaction contrasts from the same model
        for plugin in all_plugins:
            n_plugin = int((sub["plugin"] == plugin).sum())
            c_plugin = _build_lmm_contrasts(param_names, other, nrp, target_plugin=plugin)
            result["by_plugin"][plugin][key] = _contrast_stats(c_plugin, n_plugin)

    return result


def _compute_significance(
    run_dfs: dict[str, pd.DataFrame],
    runs: list[str],
    reference: str,
    metrics: list[str],
) -> dict:
    """Compute significance tests for each (non-ref provider, metric) pair via LMM.

    Fits one LMM per metric on all data:
        score ~ C(provider) * C(plugin) + (1 | question)

    Overall significance uses an equal-weighted marginal mean contrast across plugins,
    matching the equal-plugin weighting used for the weighted scores.
    Per-plugin significance uses plugin-specific interaction contrasts from the same model.

    Returns structure:
    {
        "overall": {
            metric: {
                "{ref}_vs_{other}": {"statistic", "pvalue", "n_pairs", "direction",
                                     "lmm_beta", "lmm_se"}
            }
        },
        "by_plugin": {
            plugin: {
                metric: {
                    "{ref}_vs_{other}": {"statistic", "pvalue", "n_pairs", "direction",
                                         "lmm_beta", "lmm_se"}
                }
            }
        }
    }
    """
    result: dict = {"overall": {}, "by_plugin": {}}

    ref_df = run_dfs.get(reference)
    if ref_df is None:
        return result

    non_ref = [p for p in runs if p != reference and p in run_dfs]

    # Pre-compute weighted scores for direction override (avoids matched-sample bias)
    ref_plugin_scores = _compute_plugin_scores(ref_df, metrics)
    ref_weighted = _compute_weighted_scores(ref_plugin_scores, metrics)
    other_weighted: dict[str, dict[str, float]] = {}
    for other in non_ref:
        ops = _compute_plugin_scores(run_dfs[other], metrics)
        other_weighted[other] = _compute_weighted_scores(ops, metrics)

    # Build combined DataFrame; assign(provider=p) ensures the column is correct
    combined_df = pd.concat(
        [df.assign(provider=p) for p, df in run_dfs.items()],
        ignore_index=True,
    )

    all_providers = [reference] + non_ref
    all_plugins = sorted(combined_df["plugin"].dropna().unique())

    # Prefer "erp_qa" as the LMM reference plugin for stable parameterisation;
    # fall back to the first alphabetically if it's absent.
    lmm_ref_plugin = "erp_qa" if "erp_qa" in all_plugins else all_plugins[0]

    # Initialise by_plugin skeleton so every plugin appears even if LMM returns {}
    for plugin in all_plugins:
        result["by_plugin"][plugin] = {}

    for m in metrics:
        lmm = _lmm_metric_test(combined_df, m, reference, all_providers, lmm_ref_plugin)

        # Overall — override direction from independent weighted means
        overall_entry: dict = {}
        for other in non_ref:
            key = f"{reference}_vs_{other}"
            res = lmm.get("overall", {}).get(key)
            if res is not None:
                res = res.copy()
                ref_w = ref_weighted.get(m, 0.0)
                oth_w = other_weighted[other].get(m, 0.0)
                res["direction"] = 1 if ref_w >= oth_w else -1
            overall_entry[key] = res
        result["overall"][m] = overall_entry

        # Per-plugin — direction comes directly from the LMM contrast sign
        for plugin in all_plugins:
            plugin_m_entry: dict = {}
            for other in non_ref:
                key = f"{reference}_vs_{other}"
                plugin_m_entry[key] = (
                    lmm.get("by_plugin", {}).get(plugin, {}).get(key)
                )
            result["by_plugin"][plugin][m] = plugin_m_entry

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

    sub = df[["plugin", col]].dropna()
    if sub.empty:
        return (0.0, 0.0)

    # Convert to numpy once to avoid pandas overhead inside the loop
    scores = sub[col].to_numpy()
    labels = sub["plugin"].to_numpy()
    unique_plugins = np.unique(labels)
    n = len(scores)

    # Pre-generate all index arrays in one RNG call
    all_idx = rng.integers(0, n, size=(n_boot, n))

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = all_idx[i]
        s = scores[idx]
        p = labels[idx]
        pmeans = np.array([s[p == pl].mean() for pl in unique_plugins if (p == pl).any()])
        boot_means[i] = pmeans.mean() if len(pmeans) else 0.0

    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _sig_stars(pval) -> str:
    """Return significance stars for a p-value, or empty string if not significant."""
    if pval is None or pval >= 0.05:
        return ""
    return "***" if pval < 0.001 else "**" if pval < 0.01 else "*"


def _run_color(provider: str, runs: list[str]) -> str:
    if provider in _PALETTE:
        return _PALETTE[provider]
    # Fallback to seaborn default cycle
    palette = sns.color_palette("Set2", n_colors=len(runs))
    idx = runs.index(provider) if provider in runs else 0
    r, g, b = palette[idx % len(palette)]
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


_HATCH_NOTE = "† score based on limited or non-comparable source coverage"


def _plot_overall(
    run_dfs: dict[str, pd.DataFrame],
    runs: list[str],
    metrics: list[str],
    significance: dict,
    output_path: str,
    hatch_bars: set[tuple[str, str]] | None = None,
) -> None:
    """Plot A: weighted overall scores with bootstrap CI and significance brackets."""
    n_metrics = len(metrics)
    n_runs = len(runs)
    fig_w = max(8, n_metrics * n_runs * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bar_width = 0.8 / n_runs
    x = np.arange(n_metrics)

    # Precompute weighted scores and CIs
    weighted: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple]] = {}
    for p in runs:
        if p not in run_dfs:
            continue
        df = run_dfs[p]
        plugin_scores = _compute_plugin_scores(df, metrics)
        weighted[p] = _compute_weighted_scores(plugin_scores, metrics)
        cis[p] = {}
        for m in metrics:
            cis[p][m] = _bootstrap_ci(df, m)

    # Draw bars
    reference = runs[0] if runs else None
    sig_overall = significance.get("overall", {})
    bar_positions: dict[str, list[float]] = {}
    for pi, p in enumerate(runs):
        if p not in weighted:
            continue
        offsets = x + (pi - n_runs / 2 + 0.5) * bar_width
        bar_positions[p] = list(offsets)
        heights = [weighted[p].get(m, 0.0) for m in metrics]
        yerr_lower = [weighted[p].get(m, 0.0) - cis[p][m][0] for m in metrics]
        yerr_upper = [cis[p][m][1] - weighted[p].get(m, 0.0) for m in metrics]
        color = _run_color(p, runs)
        bars = ax.bar(
            offsets, heights,
            width=bar_width * 0.9,
            color=color,
            label=DISPLAY_NAMES.get(p, p.capitalize()),
            yerr=[yerr_lower, yerr_upper],
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        for mi, (bar, h, ci_up) in enumerate(zip(bars, heights, yerr_upper)):
            m = metrics[mi]
            is_hatched = bool(hatch_bars and (p, m) in hatch_bars)
            if is_hatched:
                bar.set_hatch("///")
                bar.set_alpha(0.5)
            if h <= 0:
                continue
            stars = ""
            if p != reference:
                key = f"{reference}_vs_{p}"
                sig_entry = sig_overall.get(m, {}).get(key)
                stars = _sig_stars(sig_entry["pvalue"] if sig_entry else None)
            suffix = "†" if is_hatched else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ci_up + 0.01,
                f"{h:.2f}{stars}{suffix}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").capitalize() for m in metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Overall weighted scores (avg of plugin averages)")
    ax.legend(title="System", fontsize=9, loc="lower right")
    if hatch_bars:
        fig.text(0.01, -0.02, _HATCH_NOTE, fontsize=8, color="gray", ha="left")
    sns.despine()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_erp_only(
    run_dfs: dict[str, pd.DataFrame],
    runs: list[str],
    metrics: list[str],
    significance: dict,
    plugin: str,
    output_path: str,
) -> None:
    """Dedicated bar chart for a set of metrics scoped to a single plugin (e.g. ERP QA accuracy)."""
    n_metrics = len(metrics)
    n_runs = len(runs)
    bar_width = 0.8 / n_runs
    x = np.arange(n_metrics)

    fig_w = max(5, n_metrics * n_runs * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    reference = runs[0] if runs else None
    sig_plugin = significance.get("by_plugin", {}).get(plugin, {})

    for pi, p in enumerate(runs):
        if p not in run_dfs:
            continue
        df = run_dfs[p][run_dfs[p]["plugin"] == plugin]
        if df.empty:
            continue
        offsets = x + (pi - n_runs / 2 + 0.5) * bar_width
        heights = []
        for m in metrics:
            col = f"{m}_score"
            vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
            heights.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        color = _run_color(p, runs)
        bars = ax.bar(
            offsets, heights,
            width=bar_width * 0.9,
            color=color,
            label=DISPLAY_NAMES.get(p, p.capitalize()),
        )
        for mi, (bar, h) in enumerate(zip(bars, heights)):
            if h <= 0:
                continue
            m = metrics[mi]
            stars = ""
            if p != reference:
                key = f"{reference}_vs_{p}"
                sig_entry = sig_plugin.get(m, {}).get(key)
                stars = _sig_stars(sig_entry["pvalue"] if sig_entry else None)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}{stars}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").capitalize() for m in metrics], rotation=0)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    plugin_label = PLUGIN_DISPLAY_NAMES.get(plugin, plugin.replace("_", " ").title())
    metrics_label = ", ".join(m.replace("_", " ").capitalize() for m in metrics)
    ax.set_title(f"{plugin_label}: {metrics_label}")
    ax.legend(title="System", fontsize=9)
    sns.despine()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_by_plugin(
    run_dfs: dict[str, pd.DataFrame],
    runs: list[str],
    metrics: list[str],
    significance: dict,
    output_path: str,
    hatch_bars: set[tuple[str, str]] | None = None,
) -> None:
    """Plot B: per-plugin grouped bar charts, one row of subplots per plugin."""
    all_plugins: list[str] = sorted({
        p
        for df in run_dfs.values()
        for p in df["plugin"].dropna().unique()
    })
    if not all_plugins:
        print("  WARNING: no plugin data found — skipping scores_by_plugin.png", file=sys.stderr)
        return

    n_plugins = len(all_plugins)
    n_metrics = len(metrics)
    n_runs = len(runs)
    bar_width = 0.8 / n_runs
    x = np.arange(n_metrics)

    fig, axes = plt.subplots(
        n_plugins, 1,
        figsize=(max(8, n_metrics * n_runs * 0.55 + 2), 4 * n_plugins),
        squeeze=False,
    )

    reference = runs[0] if runs else None

    for ri, plugin in enumerate(all_plugins):
        ax = axes[ri][0]
        bar_positions: dict[str, list[float]] = {}
        sig_plugin = significance.get("by_plugin", {}).get(plugin, {})

        for pi, p in enumerate(runs):
            if p not in run_dfs:
                continue
            df = run_dfs[p][run_dfs[p]["plugin"] == plugin]
            if df.empty:
                continue
            offsets = x + (pi - n_runs / 2 + 0.5) * bar_width
            bar_positions[p] = list(offsets)
            heights = []
            for m in metrics:
                col = f"{m}_score"
                vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
                heights.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            color = _run_color(p, runs)
            bars = ax.bar(
                offsets, heights,
                width=bar_width * 0.9,
                color=color,
                label=DISPLAY_NAMES.get(p, p.capitalize()),
            )
            for mi, (bar, h) in enumerate(zip(bars, heights)):
                m = metrics[mi]
                is_hatched = bool(hatch_bars and (p, m) in hatch_bars)
                if is_hatched:
                    bar.set_hatch("///")
                    bar.set_alpha(0.5)
                if h <= 0:
                    continue
                stars = ""
                if p != reference:
                    key = f"{reference}_vs_{p}"
                    sig_entry = sig_plugin.get(m, {}).get(key)
                    stars = _sig_stars(sig_entry["pvalue"] if sig_entry else None)
                suffix = "†" if is_hatched else ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}{stars}{suffix}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").capitalize() for m in metrics],
            rotation=20, ha="right",
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(PLUGIN_DISPLAY_NAMES.get(plugin, plugin.replace("_", " ").title()))
        sns.despine(ax=ax)

    # Shared legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title="System",
            loc="upper center",
            ncol=n_runs,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    if hatch_bars:
        fig.text(0.01, -0.01, _HATCH_NOTE, fontsize=8, color="gray", ha="left")
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
        # Slug uniquely identifies the run — skip provider name filter so display
        # labels (e.g. "claude-haiku", "claude-opus") can differ from the DB column.
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
        description="Compute benchmark run comparison statistics and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", nargs="+", default=["finance_agent", "claude", "openai"],
        metavar="RUN",
        help="Run labels to compare, in order. First is the reference. (default: finance_agent claude openai)",
    )
    parser.add_argument(
        "--inf-run-ids", nargs="+", default=None, dest="inf_run_ids",
        metavar="SLUG",
        help="Inference run-ID slugs, one per run in the same order as --runs.",
    )
    parser.add_argument(
        "--eval-run-ids", nargs="+", default=None, dest="eval_run_ids",
        metavar="SLUG",
        help="Eval run-ID slugs, one per provider (optional; pins evaluation run when multiple evals exist for the same inference).",
    )
    parser.add_argument(
        "--erp-only-metrics", nargs="*", default=[], dest="erp_only_metrics",
        metavar="METRIC",
        help="Metrics excluded from overall/by-plugin plots and shown only in a dedicated ERP QA plot.",
    )
    parser.add_argument(
        "--diagnose-threshold", type=float, default=1.0, dest="diagnose_threshold",
        metavar="T",
        help="(plugin, metric) cells where reference score >= T are excluded from diagnose_cells. (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", default="results/images", dest="output_dir",
        metavar="DIR",
        help="Directory for output PNGs. (default: results/images/)",
    )
    parser.add_argument(
        "--stats-output", default="results/compare_stats.json", dest="stats_output",
        metavar="FILE",
        help="Path for output JSON stats. (default: results/compare_stats.json)",
    )
    parser.add_argument(
        "--hatch", metavar="RUN:METRIC", action="append", default=None,
        help=(
            "Mark a (run, metric) bar as limited/non-comparable with crosshatch fill. "
            "Format: 'run:metric' using internal names (e.g. 'openai:groundedness'). "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--intersect", action="store_true", default=False,
        help="Filter each run's data to only questions present in all runs.",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        metavar="SECS",
        help="Zero-score entries with inference_time_secs above this value (default: 60).",
    )
    parser.add_argument(
        "--timeout-business-brief", type=float, default=300.0, dest="timeout_business_brief",
        metavar="SECS",
        help="Timeout for business_brief plugin (default: 300).",
    )
    args = parser.parse_args()

    runs: list[str] = args.runs
    reference = runs[0]

    inf_slugs: dict[str, Optional[str]] = {}
    eval_slugs: dict[str, Optional[str]] = {}
    if args.inf_run_ids:
        if len(args.inf_run_ids) != len(runs):
            parser.error(
                f"--inf-run-ids must have {len(runs)} values, got {len(args.inf_run_ids)}"
            )
        for p, s in zip(runs, args.inf_run_ids):
            inf_slugs[p] = s
    else:
        for p in runs:
            inf_slugs[p] = None
    if args.eval_run_ids:
        if len(args.eval_run_ids) != len(runs):
            parser.error(
                f"--eval-run-ids must have {len(runs)} values, got {len(args.eval_run_ids)}"
            )
        for p, s in zip(runs, args.eval_run_ids):
            eval_slugs[p] = s

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_output = Path(args.stats_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)

    hatch_bars: set[tuple[str, str]] | None = None
    if args.hatch:
        hatch_bars = set()
        for entry in args.hatch:
            if ":" not in entry:
                parser.error(f"--hatch value must be 'run:metric', got: {entry!r}")
            provider, _, metric = entry.partition(":")
            hatch_bars.add((provider.strip(), metric.strip()))

    # ------------------------------------------------------------------
    # Step 1: Resolve runs
    # ------------------------------------------------------------------
    print("Resolving runs...")
    run_dfs: dict[str, pd.DataFrame] = {}
    run_info_map: dict[str, dict] = {}

    for p in runs:
        print(f"  {p}...", end=" ")
        result = _resolve_run(p, inf_slugs.get(p), eval_slugs.get(p))
        if result is None:
            print("skipped.")
            continue
        df, run_info = result
        # Zero out scores for entries that exceeded the inference timeout.
        # business_brief uses a longer timeout; all other plugins use args.timeout.
        timeout_note = ""
        if "inference_time_secs" in df.columns and "plugin" in df.columns:
            bb_mask = df["plugin"] == "business_brief"
            timed_out = (
                df["inference_time_secs"].notna()
                & (
                    (bb_mask & (df["inference_time_secs"] > args.timeout_business_brief))
                    | (~bb_mask & (df["inference_time_secs"] > args.timeout))
                )
            )
            n_timed_out = int(timed_out.sum())
            if n_timed_out:
                score_cols = [c for c in df.columns if c.endswith("_score")]
                for col in score_cols:
                    df.loc[timed_out & df[col].notna(), col] = 0.0
                if "overall_score" in df.columns:
                    df.loc[timed_out & df["overall_score"].notna(), "overall_score"] = 0.0
                timeout_note = (
                    f", zeroed {n_timed_out} timed-out "
                    f"(>{args.timeout}s / >{args.timeout_business_brief}s for business_brief)"
                )
        # Pool segment-specific metrics
        df, _ = _pool_segment_metrics(df)
        run_dfs[p] = df
        run_info_map[p] = run_info
        n_q = df["question"].nunique() if "question" in df.columns else len(df)
        print(f"  {n_q} questions (inf_run_id={run_info['inf_run_id']}{timeout_note})")

    if not run_dfs:
        print("ERROR: no provider data found. Register runs first.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1b: Optionally filter to common questions across all runs
    # ------------------------------------------------------------------
    if args.intersect and len(run_dfs) > 1:
        common_qs = set.intersection(
            *[set(df["question"].dropna()) for df in run_dfs.values()]
        )
        print(f"Intersect mode: {len(common_qs)} questions common across all runs.")
        for p in list(run_dfs):
            before = run_dfs[p]["question"].nunique()
            run_dfs[p] = run_dfs[p][run_dfs[p]["question"].isin(common_qs)]
            after = run_dfs[p]["question"].nunique()
            if before != after:
                print(f"  {p}: {before} -> {after} questions after intersection.")

    # ------------------------------------------------------------------
    # Step 2: Detect canonical metrics (union across all runs)
    # ------------------------------------------------------------------
    all_score_cols: set[str] = set()
    for df in run_dfs.values():
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

    for p, df in run_dfs.items():
        ps = _compute_plugin_scores(df, canonical_metrics)
        plugin_scores[p] = ps
        weighted_scores[p] = _compute_weighted_scores(ps, canonical_metrics)

    # ------------------------------------------------------------------
    # Step 5: Significance tests
    # ------------------------------------------------------------------
    print("Running significance tests...")
    significance = _compute_significance(
        run_dfs, runs, reference, canonical_metrics
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

    erp_only_set = set(args.erp_only_metrics) if args.erp_only_metrics else set()
    plot_metrics = [m for m in canonical_metrics if m not in erp_only_set]

    _plot_overall(run_dfs, runs, plot_metrics, significance, overall_plot, hatch_bars=hatch_bars)
    _plot_by_plugin(run_dfs, runs, plot_metrics, significance, by_plugin_plot, hatch_bars=hatch_bars)

    erp_accuracy_plot = ""
    if erp_only_set:
        erp_only_list = [m for m in canonical_metrics if m in erp_only_set]
        if erp_only_list:
            erp_accuracy_plot = str(output_dir / "scores_erp_accuracy.png")
            _plot_erp_only(
                run_dfs, runs, erp_only_list, significance,
                "erp_qa", erp_accuracy_plot,
            )

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
        "runs": runs,
        "reference_run": reference,
        "metrics": canonical_metrics,
        "run_info": run_info_map,
        "weighted_scores": {
            p: {m: round(v, 4) for m, v in weighted_scores[p].items()}
            for p in runs if p in weighted_scores
        },
        "plugin_scores": {
            p: {
                plugin: {m: round(v, 4) for m, v in ms.items()}
                for plugin, ms in plugin_scores[p].items()
            }
            for p in runs if p in plugin_scores
        },
        "significance": significance,
        "reliability": reliability,
        "diagnose_cells": diagnose_cells,
        "plots": {
            "overall": overall_plot,
            "by_plugin": by_plugin_plot,
            "erp_accuracy": erp_accuracy_plot,
        },
    }

    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {stats_output}")


if __name__ == "__main__":
    main()
