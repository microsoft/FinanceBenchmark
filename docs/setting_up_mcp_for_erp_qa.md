# Setting up the Dynamics 365 ERP MCP Server

This document describes the minimal steps required to connect an agent to **Dynamics 365 Finance** using the **Model Context Protocol (MCP)**.

See [erp_qa_setup.md](erp_qa_setup.md) for the end-to-end ERP QA setup flow.

Reference documentation:  
[Use Model Context Protocol for finance and operations apps](https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/copilot/copilot-mcp)

---

## Prerequisites

- Dynamics 365 Finance (version ≥ 10.0.47)
- Tier 2+ or Unified Developer Environment
- Environment provisioned with demo data using Microsoft Learn:
   - https://learn.microsoft.com/en-us/power-platform/admin/unified-experience/tutorial-install-finance-operations-provisioning-app
- Benchmark supplemental data imported:
   - [importing_synthetic_data_into_dynamics_365_finance.md](importing_synthetic_data_into_dynamics_365_finance.md)
- Access to:
  - Feature management
  - System administration

---

## 1. Enable MCP server

1. Open **Feature management**
2. Find:
   - `Dynamics 365 ERP Model Context Protocol server`
3. Enable the feature
4. Apply changes

---

## 2. Register your agent (Entra ID)

1. Go to **Azure portal → App registrations**
2. Click **New registration**
3. Save:
   - Application (client) ID
4. Add API permission:
   - `Dynamics ERP → mcp.tools`
5. Create a **client secret**

See [Azure AI Foundry — Connect to a Dynamics 365 ERP MCP server](https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/copilot/mcp/mcp-foundry) for the full registration walkthrough.

---

## 3. Allow the client in Dynamics

1. Go to:
   - **System administration → Setup → Allowed MCP clients**
2. Add your **Client ID**
3. Set:
   - `Allowed = true`

---

## 4. Connect from your agent

### Copilot Studio
1. Open your agent
2. Go to **Tools → Add a tool**
3. Select:
   - **Model Context Protocol**
4. Choose:
   - **Dynamics 365 ERP MCP server**
5. Create connection (Entra ID)
6. Add to agent

### Other agents
- Use your Entra app credentials
- Connect to the MCP endpoint exposed by Dynamics

---

## 5. Test

- Run a simple query (e.g., customers, vendors, journals)
- Confirm tool calls return results

---

## Notes

- Access is **role-based** (depends on the connected user)
- MCP exposes ERP functionality via tools (data, forms, actions)

---

Once connected, the agent can interact with Dynamics 365 Finance via MCP for ERP QA.