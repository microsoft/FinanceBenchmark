# Setting Up a Local Mock MCP Server (CSV / SQLite)

This guide walks you through running the ERP QA benchmark without a Dynamics 365 Finance environment. It uses `mcp-server-sqlite` — the official MCP reference implementation for SQLite — backed by the benchmark's synthetic CSV data.

See [erp_qa_setup.md](erp_qa_setup.md) for an overview of all setup options and a comparison of this approach with the full Dynamics 365 path.

---

## Prerequisites

- Python >= 3.11 and `uv` installed
- The benchmark repo cloned and dependencies installed (`uv sync`)
- The benchmark data archive **unzipped**:

  ```
  data/fno_benchmark_data_raw/FO Benchmark_Data.zip  →  data/fno_benchmark_data_raw/FO Benchmark_Data/
  ```

---

## Step 1: Build the SQLite database

Open and run [`notebooks/setup_mock_mcp_server.ipynb`](../notebooks/setup_mock_mcp_server.ipynb). It loads the customer, vendor, and AP/AR journal data into `data/erp_benchmark.db`.

---

## Step 2: Add the server to `.mcp.json`

The `example.mcp.json` already contains the entry. Copy it to `.mcp.json` (or merge it in) — no changes needed:

```json
{
  "mcpServers": {
    "erp-mcp-sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "data/erp_benchmark.db"]
    }
  }
}
```

Claude Code launches `mcp-server-sqlite` directly as a subprocess (stdio transport). No separate proxy process is needed.

> **For inference scripts** (which need HTTP): run `uvx mcp-proxy --port 8001 -- uvx mcp-server-sqlite --db-path data/erp_benchmark.db` in a separate terminal and set the URL in `config.yaml` accordingly.

---

## Step 3: Enable the server in `.claude/settings.local.json`

Claude Code only connects to MCP servers listed in `enabledMcpjsonServers`. Add the entry if it isn't already there:

```json
{
  "enabledMcpjsonServers": ["erp-mcp-sqlite"]
}
```

If the file already has other entries, append `"erp-mcp-sqlite"` to the existing array. After saving, start a new Claude Code session — the server will appear in `/mcp`.

---

## Step 4: Set the active MCP server

In `config.yaml`, update `shared.mcp_server_label` to point at the new entry:

```yaml
shared:
  mcp_server_label: erp-mcp-sqlite
```

No bearer token is required. You can leave `ERP_MCP_TOKEN` unset or empty in `.env`.

---

## Step 5: Run inference

```bash
uv run scripts/inference/inference.py --provider claude -n 5 --no-shuffle
```

The agent will call `read_query`, `list_tables`, and `describe_table` (the tools exposed by `mcp-server-sqlite`) instead of the Dynamics-specific tools.

---

## Limitations

| | Dynamics 365 MCP | SQLite mock server |
|---|---|---|
| Tool names | `data_find_entities`, `data_query`, ... | `read_query`, `list_tables`, `describe_table` |
| Query interface | High-level ERP operations | Raw SQL |
| Data scope | Full ERP state (posting logic, aging, etc.) | Flat tables from CSV export |
| AP/AR posting state | Fully posted, balance-aware | Raw journal lines only |
| Benchmark score comparability | Reference | Not directly comparable |

Because the tool interface is different, the agent's query patterns will differ from the Dynamics 365 baseline. Evaluation still works end to end and the scores are internally consistent, but they should not be compared against published benchmark numbers.

---

## Reverting to Dynamics 365

To switch back to the real ERP backend, restore `shared.mcp_server_label` in `config.yaml` to your Dynamics 365 MCP server label and re-run `refresh_erp_token.py` to get a fresh bearer token.
