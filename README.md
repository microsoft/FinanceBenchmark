# Finance Copilot Benchmark

A benchmark for evaluating **Copilot for Finance plugins** (ERP QA, Finance QA, Business Brief). It runs questions against live data sources via MCP-connected AI agents, captures answers and tool calls, and scores them with an LLM judge.

---

## Prerequisites

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- `.env` file containing:
  - `OPENAI_API_KEY` — used by the evaluation judge
  - ERP bearer token (written automatically by `refresh_erp_token.py`)

The ERP QA plugin requires a live Dynamics 365 Finance instance and an MCP server. If you don't have access to the one used in our evaluation, see [**Reproducing results **](#reproducing-results) below.

---

## Quickstart

```bash
uv sync
uv run refresh_erp_token.py
uv run run_benchmark.py --provider claude -n 10
```

---

## Output

| File | Description |
|---|---|
| `results/answers_{model}_{slug}.json` | Inference output |
| `results/eval_results_{answers_stem}_{slug}.json` | Evaluation output |

Both files use a metadata envelope:

```json
{"metadata": { ... }, "results": [ ... ]}
```

`metadata` contains run configuration and timestamps; `results` is the per-question array of answers or scores.

---

## Reproducing results

The Finance QA and Business Brief plugins only need web search — no special setup. The **ERP QA plugin** requires a Dynamics 365 Finance environment with data matching the benchmark questions, plus an MCP server exposing that data as tools.

Two guides walk you through the setup from scratch:

1. **[Importing Synthetic Data into Dynamics 365 Finance](docs/Importing_Synthetic_Data_into_Dynamics_365_Finance.md)** — load the synthetic AR/AP dataset into any D365 Finance trial or customer tenant
2. **[Setting Up a Basic MCP Server for Dynamics 365](docs/Setting_Up_a_Basic_MCP_Server_for_Dynamics_365.md)** — build a minimal MCP server that exposes D365 Finance data as tools the benchmark can call

Once both are in place, point `shared.mcp_server_label` in `config.yaml` at your server and run the benchmark as normal.

---

## Adding a new provider

Use `scripts/inference/_claude.py` and `scripts/inference/_openai.py` as the pattern. Each module exports two functions:

- `get_config()` — returns provider-specific runtime configuration
- `make_process_fn()` — returns the per-question async function that calls the model and returns an `InferenceResult`

Register the new provider in `inference.py` alongside the existing `claude` / `openai` branches.

---

## Running tests

```bash
uv run pytest tests/
```

---

## Comparing results

Use the `/analysis` Claude Code skill to run a multi-model comparison and generate a full markdown report with score charts, latency, and tool-call plots. `scripts/analysis/analysis.py` is deprecated in favour of the skill.

---

## Repo structure

| Path | Description |
|---|---|
| `config.yaml` | Central config (models, dataset triplets, judge settings) |
| `.mcp.json` | MCP server definitions; active server set by `shared.mcp_server_label` |
| `run_benchmark.py` | Pipeline entry point: prep → inference → eval |
| `data/` | Source question files and generated `dataset.yaml` |
| `results/` | Inference and eval output, `runs.db` run-tracking database |
| `scripts/inference/` | Inference pipeline (`inference.py`, `_claude.py`, `_openai.py`) |
| `scripts/evaluation/` | LLM judge (`evaluate.py`) |
| `scripts/analysis/` | Comparison, plotting, and run-tracking utilities |
| `scripts/data_prep/` | Dataset preprocessing and format conversion |
| `docs/` | Architecture, evaluation, and getting-started guides |
| `tests/` | Unit and integration tests |
