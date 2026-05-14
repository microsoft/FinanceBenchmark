# ERP QA Setup Guide

The `erp_qa` plugin benchmarks an agent's ability to answer accounts payable and receivable questions by querying a structured ERP data store through MCP tools.

This guide describes the benchmark dataset and two options for connecting the agent to a backend.

---

## Benchmark Data

The ERP QA questions are grounded in a synthetic dataset of 1,000 customers, 1,000 vendors, and approximately 7,000 AP/AR journal lines. The data is fully fictitious and safe for public distribution.

Files are located at:

```
data/fno_benchmark_data_raw/FO Benchmark_Data.zip
```

| File | Contents |
|---|---|
| `Master Data/D365_Customers_DMF_Final.csv` | 1,000 synthetic customer accounts (SYNCUS-0001...1000) |
| `Master Data/D365_Vendors_DMF_Final.csv` | 1,000 synthetic vendor accounts (SYNVEN-0001...1000) |
| `Transactional Data/AP_Journal_Import_Updated.xlsx` | ~3,500 AP journal lines |
| `Transactional Data/AR_Journal_Import_Updated.xlsx` | ~3,500 AR journal lines |

Company names follow the standard Microsoft sample dataset (A. Datum Corporation, Adventure Works, Contoso, Fabrikam, Northwind Traders) — all fictitious.

---

## Backend Setup Options

### Option A — Dynamics 365 Finance (full fidelity)

This option connects the benchmark agent to a live Dynamics 365 Finance instance via the official ERP MCP server. It requires an active Dynamics 365 Finance environment (version >= 10.0.47) and an Azure Entra registration.

1. [Import the synthetic data into Dynamics 365 Finance](importing_synthetic_data_into_dynamics_365_finance.md)
2. [Set up the Dynamics 365 ERP MCP server](setting_up_mcp_for_erp_qa.md)

This path produces results directly comparable to the published benchmark scores.

### Option B — Local SQLite mock server (no Dynamics licence required)

This option converts the CSV data into a local SQLite database and serves it via the `mcp-server-sqlite` reference implementation. No Dynamics 365 licence or Azure setup is required.

- [Set up a local mock MCP server over CSV / SQLite](setting_up_mock_mcp_server.md)

The agent can query the data using SQL through generic MCP tools (`read_query`, `list_tables`, `describe_table`). Because the tool interface differs from the Dynamics 365 server, benchmark scores will not be directly comparable to published results, but the pipeline infrastructure — inference, evaluation, and analysis — works end to end.

---

## Choosing an Option

| | Option A | Option B |
|---|---|---|
| Requires Dynamics 365 | Yes | No |
| Requires Azure app registration | Yes | No |
| Results comparable to published scores | Yes | No (different tool interface) |
| Setup complexity | High | Low |
| Token refresh needed | Yes | No |

If you want to reproduce the published benchmark numbers, use Option A. If you want to experiment with the pipeline or do not have a Dynamics licence, use Option B.
