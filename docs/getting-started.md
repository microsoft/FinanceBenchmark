# Getting Started

Step-by-step guide to running the benchmark for the first time.

---

## 1. Clone and install

```bash
git clone <repo-url> FinanceBenchmark
cd FinanceBenchmark
uv sync
```

`uv sync` installs all dependencies into a local virtual environment. All subsequent commands use `uv run` — no manual activation needed.

---

## 2. Set up credentials

```bash
cp env.example .env
```

You need at minimum:
- `OPENAI_API_KEY` — used by the LLM judge (evaluation)
- `ERP_MCP_TOKEN` — bearer token for the ERP MCP server (ERP QA plugin only)

To get the ERP token:
```bash
uv run refresh_erp_token.py
```

**Tokens expire.** Re-run `refresh_erp_token.py` before every inference session.


---

## 3. Run a small test

```bash
uv run run_benchmark.py --provider claude -n 5 --no-shuffle
```

This runs the full pipeline on the first 5 questions (deterministic order):

1. **Data prep** — builds `data/dataset.yaml` from source files in `config.yaml`
2. **Inference** — sends questions to Claude via MCP tools, writes `results/answers_claude-*_{slug}.json`
3. **Evaluation** — LLM judge scores each answer, writes `results/eval_results_*_{slug}.json`

The `{slug}` is an auto-generated coolname (e.g. `panda`) that uniquely identifies the run.

---

## 4. Interpret results

### Inference output: `results/answers_{model}_{slug}.json`

```json
{
  "metadata": { "model": "claude-sonnet-4-6", "provider": "claude", ... },
  "results": [
    {
      "question": "What is the overdue balance for customer ACME?",
      "answer": "...",
      "plugin": "erp_qa",
      "tool_calls": [{ "tool": "data_find_entities", "input": {}, "output": "...", "success": true }],
      "inference_time_secs": 4.2,
      "error": null
    }
  ]
}
```

### Evaluation output: `results/eval_results_*_{slug}.json`

```json
{
  "summary": { "overall_score": 0.73, "tag_scores": { "accuracy": 0.81, "citations": 0.65 } },
  "results": [{ "question": "...", "tag_scores": { "accuracy": 0.9 }, ... }]
}
```

Scores are **0.0–1.0** continuous (not binary). `summary.overall_score` is the mean across all questions and tags.

### View run history

```bash
uv run scripts/tracking/runs.py list
```

---

## 5. Run the full benchmark

```bash
# Both providers
uv run run_benchmark.py

# Single provider
uv run run_benchmark.py --provider claude
uv run run_benchmark.py --provider openai

# Re-evaluate without re-running inference
uv run run_benchmark.py --skip-prep --skip-inference
```

---

## 6. Troubleshooting

### ERP token expired
**Symptom:** 401/403 errors or "unauthorized" on ERP tool calls.
```bash
uv run refresh_erp_token.py
```

### MCP server unreachable
**Symptom:** inference hangs or connection-refused errors.

Check that `shared.mcp_server_label` in `config.yaml` matches a key in `.mcp.json`.

### Missing API key
**Symptom:** evaluation or inference fails immediately with an auth error.

Check your `.env` file.

### Azure CLI / wrong account
**Symptom:** `refresh_erp_token.py` fails, or infra commands fail.

