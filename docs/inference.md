# Inference

`scripts/inference/inference.py` is the single entry point for running benchmark questions against a live AI provider. It loads `data/dataset.yaml`, dispatches each question through the appropriate provider module, and writes results to `results/`.

---

## Running

```bash
# Run all questions with Claude
uv run scripts/inference/inference.py --provider claude

# Run all questions with OpenAI
uv run scripts/inference/inference.py --provider openai

# Limit to first N questions (useful for testing)
uv run scripts/inference/inference.py --provider claude -n 10

# Run in dataset order (shuffle is on by default)
uv run scripts/inference/inference.py --provider claude --no-shuffle

# Retry only questions that failed in a previous run (resume)
uv run scripts/inference/inference.py --provider claude

# Skip retrying failed questions from a previous run
uv run scripts/inference/inference.py --provider claude --no-retry-failed
```

Resume behaviour: if the output file already exists, successfully answered questions are preserved. Only failed/skipped entries are retried.

---

## Configuration

All inference settings live in `config.yaml`:

| Section | Key | Description |
|---|---|---|
| `shared` | `max_workers` | Number of concurrent questions |
| `inference` | `shuffle` | Randomise question order (default: true) |
| `inference` | `retry_failed` | Retry failed questions on resume (default: true) |
| `claude` | `model` | Anthropic model name |
| `claude` | `max_turns` | Max agentic loop iterations per question |
| `openai` | `model` | OpenAI model name |
| `openai` | `max_tool_calls` | Max tool calls per question |
| `openai` | `timeout` | Request timeout in seconds |

### Environment variables (`.env`)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required for `--provider openai` |
| `ERP_MCP_TOKEN` | Bearer token for the ERP MCP server (ERP QA plugin) |

`--provider claude` calls the `claude` CLI as a subprocess and uses Claude Code's own authentication — no API key env variable is needed.

---

## Provider modules

Provider-specific logic lives in `scripts/inference/_claude.py` and `scripts/inference/_openai.py`. Each exports two functions:

- `get_config(cfg)` — extracts provider-specific settings from the `config.yaml` dict and returns them as a flat dict passed to `make_process_fn`.
- `make_process_fn(config, dataset, mcp_config)` — returns an async callable `process_question(row, index, total) -> InferenceResult`. This function runs inference for a single question and must return an `InferenceResult` dict (defined in `scripts/inference/result_schema.py`).

### Adding a new provider

To benchmark a new model — including local models via Ollama, LM Studio, or a custom HTTP endpoint — implement the two functions above and register the provider name in `inference.py`:

```python
# inference.py
PROVIDERS = {
    "claude": _claude,
    "openai": _openai,
    "my_model": _my_model,   # add your module here
}
```

The `InferenceResult` TypedDict in `result_schema.py` documents every field the downstream evaluation and analysis scripts expect. At minimum, populate `question`, `plugin`, `answer`, `tool_call_count`, `successful_tool_calls`, `tool_calls`, and `inference_time_secs`. Use `make_error_result()` from the same module to produce correctly-shaped error records.

---

## Output format

Results are written to `results/answers_{model}_{slug}.json` using a metadata envelope:

```json
{
  "metadata": {
    "model": "claude-sonnet-4-6",
    "provider": "claude",
    "run_id": "panda",
    "run_timestamp": "2026-01-01T00:00:00Z",
    "config_hash": "a1b2c3d4e5f6",
    "config_snapshot": {}
  },
  "results": [
    {
      "question": "What are the aged balances for all customers?",
      "segment": "Aged balances",
      "plugin": "erp_qa",
      "answer": "...",
      "tool_calls": [
        {
          "tool": "data_find_entities",
          "input": { "entityType": "CustomerBalance" },
          "output": "...",
          "success": true
        }
      ],
      "tool_call_count": 2,
      "successful_tool_calls": 2,
      "inference_time_secs": 6.1,
      "error": null,
      "sources": { "https://example.com/article": "snippet text" }
    }
  ]
}
```

`config_hash` is the first 12 hex characters of the SHA-256 of `config_snapshot` (serialised with sorted keys). It is used for run deduplication in `results/runs.db`.

Failed questions include a non-null `"error"` field. The `"sources"` field captures web content fetched during inference for use in groundedness evaluation.

---

## Run slug

Each run gets an auto-generated coolname slug (e.g. `panda`, `falcon`) embedded in the output filename. This slug is also stored in `results/runs.db` for run tracking. See [`docs/analysis.md`](analysis.md) for details on querying runs.
