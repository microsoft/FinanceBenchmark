# Getting Started

Step-by-step setup guide for a new team member on the Finance Copilot Benchmark.

---

## 1. Clone and install

```bash
git clone <repo-url> FinanceBenchmark
cd FinanceBenchmark
uv sync
```

`uv sync` reads `pyproject.toml` and installs all dependencies into a local virtual environment. All subsequent commands use `uv run` to execute inside that environment — no manual activation needed.

---

## 2. Configure `.env`

```bash
cp env.example .env
```

Open `.env` and populate the required variables:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for the LLM judge (OpenAI). Used by `evaluate.py` to score answers. |
| `OPENAI_JUDGE_MODEL` | No | Override the judge deployment. Defaults to `gpt-5.2` (set in `config.yaml` under `evaluation.judge_deployment`). |

The ERP bearer token is also written to `.env` by `refresh_erp_token.py` (see next section) — you do not set it manually.

---

## 3. Refresh the ERP token

```bash
uv run refresh_erp_token.py
```

This script uses the Azure CLI (`az`) to obtain a bearer token for the ERP MCP server and writes it to `.env` so inference scripts can pick it up automatically.

**Tokens expire**, so you must re-run this before each inference session. If you see authentication errors during inference, this is the first thing to check.

> Note: Make sure your `az` session is authenticated with the account that has access to the ERP MCP resources (see Troubleshooting if you see authentication errors).

---

## 4. Run a small test

```bash
uv run run_benchmark.py --provider claude -n 5 --no-shuffle
```

This runs the full pipeline end-to-end on the first 5 questions in the dataset (no shuffle keeps the order deterministic for testing):

1. **Data prep** (`scripts/data_prep/preprocess.py`) — builds `data/dataset.yaml` from the source files listed in `config.yaml`.
2. **Inference** (`scripts/inference/inference.py`) — sends each question to the Claude agent (with ERP MCP + web search tools), captures answers and tool calls, writes `results/answers_claude-haiku-4-5_{slug}.json`.
3. **Evaluation** (`scripts/evaluation/evaluate.py`) — runs the LLM judge against each answer, writes `results/eval_results_answers_claude-haiku-4-5_{slug}_{eval_slug}.json`.

The `{slug}` is an auto-generated coolname (e.g. `panda`, `falcon`) that makes file names unique across runs.

---

## 5. Interpret results

### Inference output: `results/answers_{model}_{slug}.json`

```json
{
  "metadata": {
    "model": "claude-haiku-4-5",
    "provider": "claude",
    "inf_slug": "panda",
    ...
  },
  "results": [
    {
      "question": "What is the overdue balance for customer ACME?",
      "answer": "...",
      "plugin": "erp_qa",
      "tool_calls": [...],
      "latency": 4.2,
      "answered": true,
      ...
    },
    ...
  ]
}
```

Each entry in `results` is one question. The `tool_calls` array records every MCP/web tool invocation the agent made; each call has `name`, `input`, and `output` fields.

### Eval output: `results/eval_results_*.json`

```json
{
  "summary": {
    "overall_score": 0.73,
    "tag_scores": {
      "accuracy": 0.81,
      "citations": 0.65,
      "groundedness": 0.70,
      ...
    },
    "total_token_usage": { "prompt_tokens": 12400, "completion_tokens": 3100 }
  },
  "results": [
    {
      "question": "...",
      "tag_scores": {
        "accuracy": 0.9,
        "citations": 0.5,
        ...
      },
      "token_usage": { ... },
      ...
    }
  ]
}
```

Scores are on a continuous **0.0–1.0** scale (not binary). `summary.overall_score` is the mean across all questions and tags. `tag_scores` in each result entry shows per-tag scores for that individual question.

### View run history

```bash
uv run scripts/analysis/query_runs.py list
```

This queries `results/runs.db` (a SQLite database) and prints one row per run with model, provider, score, question count, and file paths.

---

## 6. Run the full pipeline

Once the small test passes, run the complete benchmark:

```bash
uv run run_benchmark.py
```

This processes all questions in `data/dataset.yaml` for both providers (Claude and OpenAI) with default settings (shuffle on, retry failed on). Output files land in `results/`.

To run a single provider:

```bash
uv run run_benchmark.py --provider claude
uv run run_benchmark.py --provider openai
```

To skip data prep or inference (e.g. re-evaluate existing answers):

```bash
uv run run_benchmark.py --skip-prep --skip-inference
```

---

## 7. Troubleshooting

### ERP token expired

**Symptom:** Inference fails with 401/403 errors or "unauthorized" messages from the ERP MCP tools.

**Fix:**
```bash
uv run refresh_erp_token.py
```

Re-run before each inference session — tokens have a short TTL.

---

### MCP server unreachable

**Symptom:** Inference hangs or errors with connection-refused / DNS errors on MCP tool calls.

**Fix:** Check that `shared.mcp_server_label` in `config.yaml` matches a server key defined in `.mcp.json`:

```yaml
# config.yaml
shared:
  mcp_server_label: your-mcp-server   # must match a key in .mcp.json
```

To switch MCP environments, change only the `mcp_server_label` value to a key that exists in `.mcp.json` — no other config changes are needed.

---

### Missing `OPENAI_API_KEY`

**Symptom:** Evaluation fails immediately with an authentication error or `OPENAI_API_KEY not set`.

**Fix:**
```bash
cp env.example .env
# then open .env and set OPENAI_API_KEY=<your key>
```

---

### Azure CLI logged into wrong account

**Symptom:** `refresh_erp_token.py` fails, or you see "Logged into Azure as demo account" when you need infrastructure access.

**Fix:** Log out of the demo account and log in with your real account:

```bash
az logout
az login
```

ACA resources (container apps, etc.) are deployed under the real user account (`wpauli@microsoft.com`), not the demo tenant. The ERP token fetch uses the demo account specifically — `refresh_erp_token.py` handles that internally, but other `az` commands need the real account.
