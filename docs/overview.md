# Finance Agent Benchmark — Overview

## What this is

This repository is an evaluation harness for **Finance Agent**, measuring how well its plugins perform finance knowledge, reasoning, and data-grounded tasks relative to leading AI baselines (Anthropic Claude, OpenAI GPT).

It is designed to be:
- **Reproducible** — every run is tracked with a config hash; re-runs with identical config reuse existing inference results
- **Maintainable** — adding a new provider, question set, or rubric assertion follows a defined pattern


---

## What it evaluates

Three plugins, each reflecting a distinct Finance Agent capability:

| Plugin | What it tests | Data source |
|---|---|---|
| `erp_qa` | AP/AR queries over structured ERP data | Synthetic Dynamics 365 Finance (live MCP) |
| `finance_qa` | Public financial Q&A (earnings, ratios, filings) | Web search / MSN Finance |
| `business_brief` | Multi-source company briefing generation | Web search + ERP enrichment |

---

## How it scores

Each answer is judged on a subset of these metrics (depending on plugin):

| Metric | Description |
|---|---|
| **Accuracy** | Correctness against ground-truth assertions |
| **Groundedness** | Claims traceable to retrieved source content |
| **Relevance** | Answer addresses the question being asked |
| **Depth** | Coverage of relevant financial context |
| **Structure** | Formatting and organisation quality |
| **Recency** | Use of current data where relevant |
| **Citations** | Presence of source references |
| **Clarity** | Explanation quality and readability |

Scores are **continuous 0.0–1.0** (not binary). The judge is an OpenAI model running via DSPy. See [evaluation.md](evaluation.md) for the full methodology.

---

## Comparators evaluated

| Label | System | Inference path |
|---|---|---|
| Claude | Anthropic API (agentic loop) | `inference.py --provider claude` |
| OpenAI | OpenAI Responses API | `inference.py --provider openai` |

---

## Repo map

```
run_benchmark.py          # pipeline entry point (prep → inference → eval)
config.yaml               # central config (models, dataset, judge settings)
data/dataset.yaml         # combined question set (pre-built)
results/                  # inference + eval JSON outputs (source of truth); runs.db is a metadata index only
scripts/
  inference/              # per-provider inference (Claude, OpenAI)
  evaluation/             # LLM judge
  analysis/               # comparison, plots, run tracking
docs/
  getting-started.md      # step-by-step first run
  inference.md            # inference configuration reference
  evaluation.md           # scoring methodology reference
  analysis.md             # analysis tools reference
```
