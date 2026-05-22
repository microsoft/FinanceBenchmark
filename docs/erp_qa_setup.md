# ERP QA Setup Guide

The `erp_qa` plugin benchmarks an agent's ability to answer accounts payable and receivable questions by querying a structured ERP data store through MCP tools.

This guide describes the benchmark dataset and the supported setup path for reproducing ERP QA runs.

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

## Supported Setup Path

ERP QA requires a Dynamics 365 Finance environment that already includes the standard demo data baseline.

Set up your environment by following the Microsoft Learn tutorial:

- https://learn.microsoft.com/en-us/power-platform/admin/unified-experience/tutorial-install-finance-operations-provisioning-app

Summary of what this means for this benchmark:

- The benchmark files in this repo are supplemental data only.
- They do not create a complete ERP baseline from scratch.
- You should provision a sandbox environment and select the demo data option first, then import the benchmark's additional data files.

After the environment is provisioned:

1. [Import the benchmark synthetic data into Dynamics 365 Finance](importing_synthetic_data_into_dynamics_365_finance.md)
2. [Set up the Dynamics 365 ERP MCP server](setting_up_mcp_for_erp_qa.md)

This is the only supported path and is the path intended for reproducing ERP QA results.
