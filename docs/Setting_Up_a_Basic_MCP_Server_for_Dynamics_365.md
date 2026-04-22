# Setting Up a Basic MCP Server for Dynamics 365

This document describes how an **external user** can set up a **basic MCP (Model Context Protocol) server** that connects to **Dynamics 365 Finance** using **publicly documented APIs**.

The goal is **architectural reproducibility**, not fidelity with Microsoft-internal systems. This guide intentionally avoids internal prompts, orchestration logic, evaluation harnesses, or proprietary services.

> **Scope:** This MCP server enables *read-only querying* of Dynamics 365 Finance data using synthetic datasets imported via the companion guide.

---

## What This Guide Is (and Is Not)

### This Guide Enables

- A minimal MCP server that exposes Dynamics 365 Finance data as tools
- Deterministic, inspectable tool behavior
- External reproduction of qualitative agent–ERP interaction patterns

### This Guide Does NOT Attempt

- To replicate Microsoft-internal MCP servers
- To reproduce internal prompts, routing logic, or evaluation metrics
- To expose write, post, or mutate operations in Dynamics

---

## High-Level Architecture

```
LLM / Agent
   |
   |  MCP (JSON-RPC / HTTP)
   v
MCP Server (You own)
   |
   |  REST / OData
   v
Dynamics 365 Finance (Public APIs)
```

Key principle: **The MCP server contains no intelligence**. It only:

- Validates inputs
- Calls Dynamics APIs
- Returns structured results

---

## Prerequisites

External users need:

- A Dynamics 365 Finance tenant with synthetic data loaded
- An Azure AD app registration (for OAuth)
- Read-only access to required Dynamics entities
- A basic web service runtime (Node.js or Python)

All prerequisites are covered by public Microsoft documentation.

---

## Authentication Model

Use **Azure Active Directory OAuth 2.0**:

- Client Credentials or Delegated User Flow
- Scope limited to Dynamics 365 Finance APIs
- Tokens passed as `Authorization: Bearer <token>`

No secrets or credentials are stored in the MCP protocol itself.

---

## Public Dynamics Endpoints to Use

A basic MCP server typically wraps:

- **OData endpoints** for entities such as:
  - Customers
  - Vendors
  - Vendor balances
  - Customer balances
- **Documented query filters** (company, date, account)

Avoid undocumented endpoints or internal form postbacks.

---

## Example MCP Tools

A minimal MCP server might expose tools such as:

- `get_vendor_balance(vendor_id, as_of_date)`
- `get_customer_balance(customer_id, as_of_date)`
- `list_open_ap_transactions(vendor_id)`
- `list_open_ar_transactions(customer_id)`

Each tool:

- Maps 1:1 to a Dynamics API call
- Is read-only
- Returns normalized JSON

---

## MCP Server Responsibilities

The MCP server should:

- Perform input validation (types, required fields)
- Enforce legal entity scoping (e.g., single company)
- Enforce read-only behavior
- Log requests for debugging and reproducibility

The MCP server should **not**:

- Execute business rules
- Perform aggregations better suited for Dynamics
- Contain prompt logic or agent heuristics

---

## Reproducibility Guidance

For external reproducibility:

- Version control the MCP server code
- Version control the synthetic dataset
- Document:
  - API versions
  - Authentication scopes
  - Entity mappings

Two different MCP implementations are acceptable if:

- They expose equivalent tools
- They return semantically equivalent results

Bit-for-bit parity is **not required**.

---

## Security and Safety

Recommended constraints:

- Read-only permissions
- Least-privilege AAD roles
- Explicit API allowlists
- No write or post capabilities

---

## Result

Following this guide, an external user should be able to:

- Stand up a minimal MCP server
- Query Dynamics 365 Finance deterministically
- Reproduce agent–ERP interaction patterns
- Do so without access to any Microsoft-internal systems
