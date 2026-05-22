# Running the Benchmark with OpenAI Responses API

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
