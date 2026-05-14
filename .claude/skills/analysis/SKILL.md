---
name: analysis
description: Run a complete provider comparison: weighted score charts with CI, per-plugin distribution plots (latency, tool calls, citations), reliability stats, and Finance Agent failure pattern analysis. Replaces /report and /diagnose.
argument-hint: "[eval_file1 eval_file2 ...] [--labels L1 L2]  (defaults to all results/eval_results_*.json)"
---

# analysis skill

## Plugin display names

When writing any user-facing text (report section headers, labels, descriptions), use these display names:

| Internal identifier | Display name |
|---|---|
| `erp_qa` | Entity financial obligations with the user company research |
| `finance_qa` | Entity financial performance research |
| `business_brief` | Finance business briefs preparation |
| `finance_agent` (provider) | Finance Agent |

Use internal identifiers only in code contexts (CLI args, JSON keys, variable names, filename patterns).

## Step 1 — Resolve inputs

**If `$ARGUMENTS` is empty:** glob `results/eval_results_*.json` (top-level only, not subdirs). If none found, stop with a clear error message.

**If `$ARGUMENTS` has file paths:** use those files.

Parse `--labels <L1> <L2> ...` if present in `$ARGUMENTS`; otherwise auto-derive labels.

For each eval file:
- Read the JSON envelope's `metadata` field.
- Extract `provider` from `metadata.inference_metadata.provider`. If that field is absent, infer from the filename: `claude-*` → `claude`, `gpt-*` or `openai-*` → `openai`, `finance_agent-*` → `finance_agent`.
- Extract `inf_run_id` from `metadata.inf_run_id`.
- Extract `eval_run_id` from `metadata.eval_run_id`.
- Auto-derive label if not given: `"<Provider capitalized> (<model>)"` using `metadata.inference_metadata.model` (e.g. `"Claude (claude-sonnet-4-5)"`). When provider is `finance_agent`, use label `"Finance Agent"`.
- Derive the answers file: the eval filename is `eval_results_answers_{stem}_{eval_slug}.json`, so the answers file is `answers_{stem}.json` in the same directory. Check it exists; if not, stop with a clear error.

Collect ordered lists: `providers`, `inf_run_ids`, `eval_run_ids`, `eval_files`, `answers_files`, `labels`. Order: finance_agent first if present, then others alphabetically.

## Step 2 — Run `compare_runs.py`

Run the following command and stop with an error if it fails:

```
uv run scripts/analysis/compare_runs.py \
  --runs <runs...> \
  --inf-run-ids <inf_run_ids...> \
  --eval-run-ids <eval_run_ids...> \
  --output-dir results/images/ \
  --stats-output results/compare_stats.json \
  --diagnose-threshold 1.0 \
  [--hatch openai:groundedness]   ← include this flag if and only if openai is one of the providers
```

**Groundedness hatch rule:** If `openai` is in the providers list, always append `--hatch openai:groundedness` to the command. The OpenAI Responses API does not expose web search result content, so GPT groundedness scores are based on at most 1–2 Playwright-fetched URLs rather than the full source trail. The hatched bar (crosshatch fill, 0.xx†) signals that the score is not directly comparable to Claude or Finance Agent groundedness.

This reads from `runs.db` via slugs (not from file paths directly). CI error bars are already rendered in the output PNGs.

Outputs produced:
- `results/images/scores_overall_weighted.png` — weighted scores with CI bars
- `results/images/scores_by_plugin.png` — per-plugin breakdown (includes accuracy for erp_qa)
- `results/compare_stats.json` — machine-readable statistics

## Step 3 — Run plot scripts (parallel)

The three plot scripts are independent. Issue **all three as simultaneous bash tool calls in a single message** so they execute concurrently. Wait for all three to complete before proceeding.

```
uv run scripts/analysis/plot_latency.py <answers_files...> \
  --labels <labels...> \
  --by-plugin \
  --output results/images/latency_by_plugin.png
```

```
uv run scripts/analysis/plot_tool_calls.py <answers_files...> \
  --labels <labels...> \
  --by-plugin \
  --output results/images/tool_calls_by_plugin.png
```

```
uv run scripts/analysis/plot_citations.py <answers_files...> \
  --labels <labels...> \
  --by-plugin \
  --output results/images/citations_by_plugin.png
```

Notes:
- Providers without `inference_time_secs` (e.g. Finance Agent) will be skipped in the latency plot — this is expected.
- Finance Agent answers (plain text from scraping) will show zero citations — this is expected and meaningful.

## Step 6 — Load stats and identify Finance Agent analysis cells

Read `results/compare_stats.json`. Extract: `runs`, `metrics`, `run_info`, `weighted_scores`, `plugin_scores`, `significance`, `reliability`.

If `finance_agent` is not in `runs`, or `run_info["finance_agent"]["eval_file"]` is empty or missing: skip steps 6b–7 and omit the Finance Agent Performance Analysis section from the report entirely.

Otherwise, identify three cell lists:

**Win cells** (`win_cells`): for each (metric, plugin) pair in `plugin_scores["finance_agent"]`, check if `plugin_scores["finance_agent"][plugin][metric]` is more than 0.05 **above** the best non-Finance-Agent provider score for the same (metric, plugin). Collect as `{plugin, metric, finance_agent_score, gap}` sorted by gap descending.

**Loss cells** (`loss_cells`): for each (metric, plugin) pair, check if `plugin_scores["finance_agent"][plugin][metric]` is more than 0.05 **below** the best non-Finance-Agent provider score for the same (metric, plugin). Collect as `{plugin, metric, finance_agent_score, gap}` sorted by score ascending.

**Opportunity cells** (`opportunity_cells`): for each (metric, plugin) pair where `plugin_scores["finance_agent"][plugin][metric] < 1.0`, regardless of win/loss status. Collect as `{plugin, metric, finance_agent_score}` sorted by score ascending. These represent areas where Finance Agent's absolute score still leaves meaningful improvement headroom, whether Finance Agent is currently winning or losing that metric.

## Step 7 — Diagnose and synthesize Finance Agent performance patterns (LLM step)

Collect the unique set of (plugin, metric) pairs across all three cell lists.

Issue **all diagnose calls simultaneously as parallel bash tool calls in a single message** — one tool call per unique (plugin, metric) pair. Each call:

```
uv run scripts/analysis/sample_low_scoring.py <finance_agent_eval_file> <metric> --plugin <plugin>
```

where `<finance_agent_eval_file>` = `run_info["finance_agent"]["eval_file"]`.

Wait for all calls to complete, then capture each stdout. Store as a lookup `diagnose_outputs[(plugin, metric)] = output`.

The diagnose output now includes an **"Assertion failure rates"** table showing, for every assertion in the metric, its mean score, the fraction of entries scoring 0.0 (`p(0.0)`), and the fraction scoring 1.0 (`p(1.0)`) across **all** scored entries — not just the bottom sample. Use this table as your primary evidence for any pattern claim. A pattern must be visible in the aggregate stats (e.g. an assertion with `p(0.0) ≥ 50%`), not just in one or two sampled entries. If no assertion dominates or the data doesn't reveal a clear cause, say so explicitly — do not construct a plausible-sounding explanation that isn't traceable to the numbers.

**7a — Win synthesis**

For each (plugin, metric) pair in `win_cells`, draw on `diagnose_outputs[(plugin, metric)]`. Group by plugin. For each plugin with win cells, synthesize a "where Finance Agent wins" section:
- Lead from the assertion failure rates table: which assertions is Finance Agent scoring consistently high on? State the rate if notable.
- Name a root cause only if it is visible across the majority of entries. If the aggregate stats show high scores without a clear driver, write: "No single root cause identified from the data."
- Distinguish genuine capability advantage vs. rubric alignment vs. data characteristics — but only when the data supports the distinction.
- Format: bulleted list, one bullet per pattern, prefixed with `**<metric>**:`.

If `win_cells` is empty: write one line: "No cells met the win threshold (Finance Agent lead > 0.05 vs best other provider)."

**7b — Loss synthesis**

For each (plugin, metric) pair in `loss_cells`, draw on `diagnose_outputs[(plugin, metric)]`. Group by plugin. For each plugin with loss cells, synthesize a "where Finance Agent loses" section:
- Lead from the assertion failure rates table: identify the worst-performing assertions (highest `p(0.0)`, lowest mean) and state the figures.
- Name a root cause only if one or two assertions clearly dominate (e.g. `p(0.0) > 50%`). If failures are spread across many assertions without a dominant one, write that instead.
- Distinguish model limitation vs. data/rubric issue vs. evaluation setup problem — but only when the failure type makes this unambiguous. When unclear, say so.
- Format: bulleted list, one bullet per pattern, prefixed with `**<metric>**:`.

If `loss_cells` is empty: write one line: "No cells met the loss threshold (Finance Agent gap < 0.05 vs best other provider)."

**7c — Opportunity synthesis + example extraction**

For each (plugin, metric) pair in `opportunity_cells`:

1. **Identify the pattern from aggregate stats, then select an example.** Read `diagnose_outputs[(plugin, metric)]`. Start with the assertion failure rates table — find the assertion(s) with the highest `p(0.0)` or lowest mean. If one assertion stands out as the dominant driver, that is the pattern to report. Then, among the low-scoring entries in the diagnose output, pick the one whose judge reasoning most directly reflects that assertion failure. If no assertion clearly dominates, pick the lowest-scoring entry and note that no dominant pattern was found. Record the selected entry's `question`, `score`, and a one-sentence judge reasoning excerpt.

2. **Extract the full result object.** Find the full result object for that question in `<finance_agent_eval_file>` by matching on the `question` field inside the `results` array. Write it to `results/examples/<plugin>_<metric>_example.json` (create the `results/examples/` directory if needed). The file should contain the single result object as a JSON object (not an array).

3. **Write the opportunity bullet.** Group by plugin. For each plugin with opportunity cells, write a bulleted list:
   - Focus on **what could concretely be improved** and **how**: prompt/instruction changes, inference-side fixes, rubric alignment, or data quality issues.
   - State whether the gap is primarily a model issue, an inference setup issue, a rubric/evaluation issue, or a data quality issue — since this shapes what team owns the fix.
   - Include cells regardless of whether Finance Agent wins or loses that metric; a win at 0.60 is still an opportunity.
   - End each bullet with a blockquote `> **Example** ([full entry](examples/<plugin>_<metric>_example.json)): *"<question truncated to ~120 chars>"* — score <score>, judge: "<one-sentence excerpt from judge reasoning>"`.
   - The judge excerpt must be directly traceable to the root cause stated in the bullet — not a secondary or incidental failure in that entry.
   - Format: bulleted list, one bullet per pattern, prefixed with `**<plugin display name> / <metric>**:` (use the display name from the plugin display names table above).

If `opportunity_cells` is empty (all scores at or above threshold): write one line: "No cells below the improvement threshold."

## Step 8 — Write `results/provider_comparison.md`

Overwrite `results/provider_comparison.md` if it already exists. Use today's date. Image paths must be relative to `results/` (i.e. `images/scores_overall_weighted.png`, not `results/images/...`).

Use plugin display names (from the table at the top of this skill) in all section headers, labels, and descriptive text. Use internal identifiers only in code blocks.

Report structure:

```
# Finance Copilot Benchmark — Provider Comparison

**Date:** <today>
**Providers:** <comma-separated provider labels>
**Dataset:** <total unique questions across providers> questions
**Judge:** <judge from any run_info entry>

## Runs

| Provider | Model | Inference slug | Eval slug |
|---|---|---|---|
...one row per provider...

## Overall Scores

![Overall weighted scores (avg of plugin averages)](images/scores_overall_weighted.png)

<1–2 sentence headline finding from weighted_scores>

### Significance

LMM-based pairwise significance tests (equal-weighted marginal contrasts). ↑ = reference leads, ↓ = reference trails.

Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant

| Metric | <ref>_vs_<other1> | <ref>_vs_<other2> |
|---|---|---|
...one row per metric...

Format cells as: "\*\*\* p=0.001 (n=120, ↑)" for p<0.001 with ref leading, "\*\* p=0.008 (n=120, ↓)" for p<0.01 with ref trailing, "\* p=0.03 (n=120, ↑)" for p<0.05, "ns p=0.21 (↑)" for not significant, "—" if no test available.
The `direction` field in the stats JSON is +1 (ref leads) or -1 (ref trails); use ↑ for +1 and ↓ for -1.

### Score table

| Metric | <Provider 1> | <Provider 2> | <Provider 3> |
|---|---|---|---|
...one row per metric, scores to 2 d.p....

## Per-Plugin Scores

![Scores by plugin](images/scores_by_plugin.png)

<One sentence per plugin noting notable patterns — use plugin display names>

### Entity Financial Obligations Accuracy

![Entity financial obligations accuracy](images/scores_erp_accuracy.png)

<One sentence on accuracy comparison across providers for Entity financial obligations with the user company research, noting the bimodal pattern and which provider leads.>

## Distributions

### Inference Latency

![Latency by plugin](images/latency_by_plugin.png)

<One sentence on median/p90 differences. Note if a provider was skipped due to missing latency data.>

### Tool Calls

![Tool calls by plugin](images/tool_calls_by_plugin.png)

<One sentence on notable tool call patterns.>

### Citations

![Citations by plugin](images/citations_by_plugin.png)

<One sentence. Note if any provider shows all-zero counts and whether that is expected.>

## Reliability

| Provider | Total | Answered | Errors | Skipped (eval) | "Sorry" responses |
|---|---|---|---|---|---|
...one row per provider from reliability stats...

<Brief note on any reliability concerns.>

## Finance Agent Performance Analysis

### Where Finance Agent Wins

<One section per plugin that had win cells, or the one-line "no cells" note. Use plugin display names for section headers.>

#### <Plugin display name>

<Synthesized win patterns from Step 7a>

### Where Finance Agent Loses

<One section per plugin that had loss cells, or the one-line "no cells" note. Use plugin display names for section headers.>

#### <Plugin display name>

<Synthesized loss patterns from Step 7b>

### Opportunities

<One section per plugin that had opportunity cells (score < 0.75), or the one-line "no cells" note. Use plugin display names for section headers.>

#### <Plugin display name>

<Synthesized improvement opportunities from Step 7c>

## Key Takeaways

<Numbered list, 4–6 items. Lead with most important finding.>

---

## Reproducing This Report

All steps except the failure pattern analysis (Step 7) are pure Python and can be run without Claude Code.

**Step 2 — Weighted scores and significance:**
\```
uv run scripts/analysis/compare_runs.py \
  --runs <runs> \
  --inf-run-ids <inf_slugs> \
  --eval-run-ids <eval_slugs> \
  --output-dir results/images/ \
  --stats-output results/compare_stats.json \
  --diagnose-threshold 1.0 \
  --hatch openai:groundedness   # if openai is one of the providers
\```

**Step 3 — Latency by plugin:**
\```
uv run scripts/analysis/plot_latency.py <answers_files> \
  --labels <labels> --by-plugin --output results/images/latency_by_plugin.png
\```

**Step 4 — Tool calls by plugin:**
\```
uv run scripts/analysis/plot_tool_calls.py <answers_files> \
  --labels <labels> --by-plugin --output results/images/tool_calls_by_plugin.png
\```

**Step 5 — Citations by plugin:**
\```
uv run scripts/analysis/plot_citations.py <answers_files> \
  --labels <labels> --by-plugin --output results/images/citations_by_plugin.png
\```

**Step 6 (data only, no LLM) — Sample low-scoring Finance Agent entries per metric:**
\```
uv run scripts/analysis/sample_low_scoring.py <finance_agent_eval_file> <metric> --plugin <plugin>
\```

The failure pattern synthesis in Step 7 requires an LLM. All other outputs are deterministic.
Machine-readable statistics: `results/compare_stats.json`.
```

## Step 9 — Confirm and report

After writing the file, confirm it was written successfully. Report to the user:
- Path of the report written (`results/provider_comparison.md`)
- Which providers were included and their slugs
- Which plots were generated
- Win / loss / opportunity cell counts and which plugins they cover (use plugin display names)
- List of example JSON files written to `results/examples/` (one per opportunity cell)
- Any warnings or skipped steps (e.g. latency skipped for Finance Agent)
