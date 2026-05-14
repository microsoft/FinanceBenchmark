# Running the Benchmark with OpenAI

This project supports three OpenAI-based inference paths. Choose based on your use case:

| | Responses API | Deep Research (API) | ChatGPT Web UI |
|---|---|---|---|
| **Script** | `inference.py --provider openai` | Deep Research adapter | n/a (interactive) |
| **Auth to MCP** | Bearer token passed directly | Via adapter server | OAuth 2.1 via proxy |
| **Infrastructure** | None | Local adapter server | HTTPS proxy (e.g. Azure Container Apps) |
| **Use case** | Batch / scripted benchmarking | Deep Research models | Interactive exploration |

---

## OpenAI Responses API (recommended for benchmarking)

For standard models (e.g. `gpt-4.1`, `gpt-5.2`), the inference script connects directly to your MCP server:

### Prerequisites

- OpenAI API key set in `.env`:
  ```
  OPENAI_API_KEY=sk-...
  ```
- For the **ERP QA plugin**: a bearer token for your ERP MCP server set in `.env`:
  ```
  ERP_MCP_TOKEN=<your-token>
  ```
- MCP server URL configured in `.mcp.json` and `shared.mcp_server_label` in `config.yaml`.

### Running

```bash
# All questions
uv run scripts/inference/inference.py --provider openai

# Limit to first N questions
uv run scripts/inference/inference.py --provider openai -n 10
```


## Output format

All three options produce the same result format. See [`docs/inference.md`](inference.md) for the full output schema.
