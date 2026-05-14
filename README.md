# Finance Copilot Benchmark

A benchmark for evaluating **Copilot for Finance plugins** (ERP QA, Finance QA, Business Brief). Runs questions against live data sources via MCP-connected AI agents, scores answers with an LLM judge, and compares providers (Finance Agent, Claude, OpenAI).

→ **[Overview and motivation](docs/overview.md)** — what this measures and why

![Results](result.png)

## Quickstart

```bash
uv sync
cp env.example .env   # add OPENAI_API_KEY and ERP_MCP_TOKEN
uv run refresh_erp_token.py
uv run run_benchmark.py --provider claude -n 5
```

See **[Getting Started](docs/getting-started.md)** for the full walkthrough.

---

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Inference | `scripts/inference/inference.py` | `results/answers_{model}_{slug}.json` |
| Evaluation | `scripts/evaluation/evaluate.py` | `results/eval_results_{answers_stem}_{slug}.json` |
| Analysis | `scripts/analysis/compare_runs.py` | `results/images/`, `results/compare_stats.json`, `results/provider_comparison.md` |

`data/dataset.yaml` is pre-built and included. All settings live in `config.yaml`. Run everything at once with `run_benchmark.py`.

---

## Documentation

| Doc | Contents |
|---|---|
| [Getting Started](docs/getting-started.md) | First run, output interpretation, troubleshooting |
| [Inference](docs/inference.md) | Configuration, resume, output schema |
| [Evaluation](docs/evaluation.md) | Scoring methodology, judge routing, rubric |
| [Analysis](docs/analysis.md) | Provider comparison, plots, run tracking queries |
| [Claude setup](docs/anthropic.md) | Claude-specific inference setup |
| [OpenAI setup](docs/openai.md) | OpenAI Responses API, Deep Research, ChatGPT proxy |

---

## Reproducing results

The Finance QA and Business Brief plugins only need web search — no special infrastructure. The **ERP QA plugin** requires a Dynamics 365 Finance instance with matching data and an MCP server.

Two guides cover external reproduction from scratch:

1. **[Importing Synthetic Data into Dynamics 365 Finance](docs/importing_synthetic_data_into_dynamics_365_finance.md)**
2. **[Setting Up an MCP Server for ERP QA](docs/setting_up_mcp_for_erp_qa.md)**

Once set up, copy `example.mcp.json` to `.mcp.json` and add your MCP server configuration, set `shared.mcp_server_label` in `config.yaml`, and run normally.

---

## Running tests

```bash
uv run pytest tests/
```
