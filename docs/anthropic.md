# Claude (Anthropic) Inference

`inference.py --provider claude` runs each question as an agentic loop using the Claude Code CLI (`claude`) with MCP tools. See [inference.md](inference.md) for the full pipeline reference and output format.

---

## Prerequisites

- Claude Code installed and authenticated (`claude` CLI available in PATH)
- For ERP QA: `ERP_MCP_TOKEN` set in `.env` (see [setting_up_mcp_for_erp_qa.md](setting_up_mcp_for_erp_qa.md))
- MCP server URL configured in `.mcp.json` and `shared.mcp_server_label` in `config.yaml`

---

## Running

```bash
# All questions
uv run scripts/inference/inference.py --provider claude

# First N questions (useful for testing)
uv run scripts/inference/inference.py --provider claude -n 10

# Resume an interrupted run
uv run scripts/inference/inference.py --provider claude
```

---

## Configuration

Key settings in `config.yaml` under the `claude` section:

| Key | Description |
|---|---|
| `model` | Anthropic model name (e.g. `claude-sonnet-4-6`) |
| `max_turns` | Maximum agentic loop iterations per question |

Concurrency is controlled by `shared.max_workers`.
