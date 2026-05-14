# Importing Synthetic Data into Dynamics 365 Finance

This document describes a **public, reproducible process** for importing **synthetic finance data** (customers, vendors, AP/AR journals) into **Dynamics 365 Finance** so that **external researchers and practitioners** can reproduce evaluation or benchmarking results.

See [erp_qa_setup.md](erp_qa_setup.md) for an overview of setup options, including a local mock server that does not require a Dynamics 365 licence.

The document is written for an **external audience** with **minimal prior knowledge of Dynamics 365**. It intentionally avoids internal tools, proprietary workflows, or product internals, and relies **only on publicly documented Dynamics features**.

---

**Important:** This guide enables *structural reproducibility* — the same agent–ERP interaction patterns and tool-calling behaviors. It does **not** guarantee bit-for-bit answer reproduction. Some benchmark questions reference company names specific to the original evaluation tenant (e.g., Forest Wholesales, Birch Company, Cave Wholesales); those questions will behave correctly with the provided data but may return "not found" for those specific companies. This is expected and still valid for evaluating agent behavior.

This guide does **not** assume access to Microsoft's internal MCP servers, prompts, evaluation harnesses, or tenants.

---

## Data Files

The benchmark data is provided in:

```
data/fno_benchmark_data_raw/FO Benchmark_Data.zip
```

The zip contains:

| File | Description |
|---|---|
| `Master Data/D365_Customers_DMF_Final.csv` | 1,000 synthetic customer accounts (SYNCUS-0001…1000) |
| `Master Data/D365_Vendors_DMF_Final.csv` | 1,000 synthetic vendor accounts (SYNVEN-0001…1000) |
| `Transactional Data/AP_Journal_Import_Updated.xlsx` | ~3,500 AP journal lines |
| `Transactional Data/AR_Journal_Import_Updated.xlsx` | ~3,500 AR journal lines |
| `Master_Data_Import_Guide.html` | Step-by-step guide for importing customers and vendors via DMF |
| `General_Journal_Import_Guide.html` | Step-by-step guide for importing AP/AR journals via Excel Add-in |

Company names use the standard Microsoft sample dataset (A. Datum Corporation, Adventure Works, Contoso, Fabrikam, Northwind Traders, etc.) and are fully fictitious.

The HTML guides inside the zip provide detailed, step-by-step screenshots for the import procedures described below.

---

## Scope and Assumptions

- Product: **Dynamics 365 Finance** (public cloud)
- Tenant: Any customer or trial tenant with Dynamics 365 Finance enabled
- Legal entity: **Single company** (e.g., `USMF` or equivalent)
- Data types supported:
    - Customers
    - Vendors
    - Accounts Payable (AP) journals
    - Accounts Receivable (AR) journals
- Data constraints:
    - Fully **synthetic / fictitious** data
    - Safe for public distribution
    - Read-only usage after import
- Goal:
    - Enable realistic balances and transaction volume
    - Support reproducible querying and evaluation experiments

---

## High-Level Workflow

1. Prepare a Dynamics 365 Finance environment
2. Import master data (customers and vendors) using Data Management
3. Import transactional data (AP/AR journals) using Excel
4. Post journals
5. Run aging and validation steps

Each step below maps to **public Microsoft documentation**.

---

## 1. Environment Preparation

### Prerequisites

External users must have:

- An active **Dynamics 365 Finance** environment
- Permissions to access:
    - **Data management** workspace
    - **General ledger** journals
    - **Accounts payable / receivable** modules

### Recommended Setup (Temporary)

- Disable **budget control** for the target legal entity
- Confirm that required **main accounts** and **posting profiles** exist

### Required Reference Data

The CSV files reference specific lookup values that must already exist in Dynamics before import:

| Reference type | Required values |
|---|---|
| Customer groups | 10, 20, 30, 80, 90 |
| Vendor groups | 10, 20, 30, 40, 50 |
| Payment terms | Net10, Net15, Net30, Net45, COD |
| Sales tax groups | No-Tax, EXMPT FOR, CA, TX, NY, FL, WA, IL |
| Payment methods | CHECK, ELECTRONIC, CC |

If any of these are missing, DMF will reject the rows that reference them.

---

## 2. Import Customers and Vendors (Master Data)

### Tooling (Public)

- **Data Management Framework (DMF)**

### Supported Data Entities

- Customers: `Customers V3`
- Vendors: `Vendors V2`

### Conceptual Import Procedure

1. Navigate to **Data management**
2. Create a new **Import project**
3. Add the appropriate entity (Customers or Vendors)
4. Upload the provided **CSV file**
5. Map source fields to target fields — **several mappings must be adjusted manually** (see tables below)
6. Execute the import job
7. Address validation errors, if any

> **Critical — Customer field mappings (Customers V3):** D365 auto-mapping does not correctly resolve these columns. You must map them manually via **View map** before importing.
>
> | Source column (CSV) | Target field (D365) |
> |---|---|
> | `CreditMax` | `Credit limit` |
> | `CurrencyCode` | `SalesCurrencyCode` |
> | `CustomerHoldStatus` | `CredManAccountStatusId` |
> | `PaymentTermsId` | `PaymentTerms` |
> | `PrimaryContactPersonFirstName` | `PersonFirstName` |

> **Critical — Vendor field mappings (Vendors V2):**
>
> | Source column (CSV) | Target field (D365) |
> |---|---|
> | `AddressState` | `AddressStateId` |
> | `PaymentMethodName` | `DefaultVendorPaymentMethodName` |
> | `PaymentTermsId` | `DefaultPaymentTermsName` |
> | `PrimaryContactEmail` | `PrimaryEmailAddress` |
> | `PrimaryContactPersonFirstName` | `PersonFirstName` |
> | `PrimaryContactPhone` | `PrimaryPhoneNumber` |
> | `SalesTaxGroup` | `SalesTaxGroupCode` |
> | `VendorHoldStatus` | `OnHoldStatus` |

The HTML guide bundled with the data (`Master_Data_Import_Guide.html`) contains the same tables with screenshots.

---

## 3. Import AP / AR Journals (Transactional Data)

### Tooling (Public)

- **General journal** UI
- **Excel Add-in for Dynamics 365**

### Pre-Requirements

- Customers and vendors must be imported first (journal lines reference SYNCUS / SYNVEN accounts)
- **Budget control must be disabled** — the journal totals will exceed default budget limits and the import will fail if budget control is active

### Conceptual Import Procedure

1. Open the relevant **General journal** (AP or AR)
2. Use **Open in Excel** (Excel Add-in)
3. Paste or load journal lines from the provided file
4. Publish data back to Dynamics
5. Validate journal lines
6. Post the journal

The HTML guide bundled with the data (`General_Journal_Import_Guide.html`) has the step-by-step navigation path and expected line counts (AR: 3,467 lines; AP: 3,474 lines).

---

## Result

- Reproducible synthetic Dynamics environment
- Deterministic, queryable data
- No dependency on internal Microsoft systems
