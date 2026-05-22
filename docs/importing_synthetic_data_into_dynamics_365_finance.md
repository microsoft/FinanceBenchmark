# Importing Synthetic Data into Dynamics 365 Finance

This document covers only the benchmark-specific data import that must be done after provisioning a Dynamics 365 Finance sandbox with demo data.

For the environment provisioning steps, use Microsoft Learn as the source of truth:

- https://learn.microsoft.com/en-us/power-platform/admin/unified-experience/tutorial-install-finance-operations-provisioning-app

Do not skip this prerequisite. The benchmark import files assume the demo dataset baseline is already present.

---

## What This Import Adds

The benchmark data package adds four supplemental datasets used by ERP QA:

| File | Description |
|---|---|
| `Master Data/D365_Customers_DMF_Final.csv` | Synthetic customer master records |
| `Master Data/D365_Vendors_DMF_Final.csv` | Synthetic vendor master records |
| `Transactional Data/AP_Journal_Import_Updated.xlsx` | Synthetic AP journal lines |
| `Transactional Data/AR_Journal_Import_Updated.xlsx` | Synthetic AR journal lines |

Data package location:

```
data/fno_benchmark_data_raw/FO Benchmark_Data.zip
```

---

## Minimal Workflow

1. Provision a Dynamics 365 Finance sandbox and select demo data using the Microsoft Learn tutorial above.
2. Unzip `FO Benchmark_Data.zip`.
3. Import customer and vendor files in Data management (DMF).
4. Import AP and AR journal files and post the journals.

The package already includes short import guides for these benchmark-specific files:

- `Master_Data_Import_Guide.html`
- `General_Journal_Import_Guide.html`

This repo intentionally keeps detailed product setup instructions in Microsoft Learn to minimize duplicated documentation maintenance.

---

## Validation Checklist

- Sandbox was provisioned with demo data.
- Customer and vendor imports completed without blocking errors.
- AP and AR journals were imported and posted.

After this is complete, continue with [setting_up_mcp_for_erp_qa.md](setting_up_mcp_for_erp_qa.md).
