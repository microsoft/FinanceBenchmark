"""
refresh_erp_token.py

Fetches a new ERP MCP bearer token using UsernamePasswordCredential
and updates .env. Does not touch the global az CLI account state.

Usage:
  uv run refresh_erp_token.py

Required .env keys: AZ_USERNAME, AZ_PASSWORD, AZ_TENANT_ID
"""

import base64
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import yaml


def _load_env(path: str = ".env") -> dict[str, str]:
    env = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                # Strip surrounding quotes (single or double)
                val = v.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                env[k.strip()] = val
    except FileNotFoundError:
        pass
    return env


_cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
_mcp_config = json.loads((Path(__file__).parent / _cfg["shared"]["mcp_config_file"]).read_text(encoding="utf-8"))
_mcp_url = _mcp_config["mcpServers"][_cfg["shared"]["mcp_server_label"]].get("url")
_resource_candidate = _mcp_url.rstrip("/").removesuffix("/mcp") if _mcp_url else None
# localhost servers don't use Azure auth — treat as no-token-needed
RESOURCE = (
    _resource_candidate
    if _resource_candidate and not _resource_candidate.startswith(("http://localhost", "http://127.0.0.1"))
    else None
)
ENV_FILE = ".env"

# Well-known Azure CLI client ID (public, widely used for user auth flows)
_AZ_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"


def refresh_token() -> str:
    """Fetch a fresh ERP MCP bearer token and persist it.

    Uses UsernamePasswordCredential with AZ_USERNAME / AZ_PASSWORD / AZ_TENANT_ID
    from .env — does not affect the global az CLI account state.

    Returns:
        The new token string.

    Raises:
        SystemExit: If credentials are missing or the token fetch fails.
    """
    if not RESOURCE:
        print("No ERP URL configured for the active MCP server — token refresh not applicable.", file=sys.stderr)
        return

    from azure.identity import UsernamePasswordCredential

    env = _load_env(ENV_FILE)
    username  = env.get("AZ_USERNAME") or os.environ.get("AZ_USERNAME")
    password  = env.get("AZ_PASSWORD") or os.environ.get("AZ_PASSWORD")
    tenant_id = env.get("AZ_TENANT_ID") or os.environ.get("AZ_TENANT_ID")

    for name, val in [("AZ_USERNAME", username), ("AZ_PASSWORD", password), ("AZ_TENANT_ID", tenant_id)]:
        if not val:
            print(f"ERROR: {name} not found in .env", file=sys.stderr)
            sys.exit(1)

    print(f"Fetching access token for {RESOURCE} as {username} ...")
    cred  = UsernamePasswordCredential(
        client_id=_AZ_CLI_CLIENT_ID,
        tenant_id=tenant_id,
        username=username,
        password=password,
    )
    token_obj = cred.get_token(f"{RESOURCE}/.default")
    token = token_obj.token
    if not token:
        print("ERROR: received an empty token.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    if re.search(r"^ERP_MCP_TOKEN=", content, re.MULTILINE):
        content = re.sub(r"^ERP_MCP_TOKEN=.*$", f"ERP_MCP_TOKEN={token}", content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\nERP_MCP_TOKEN={token}\n"

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    os.environ["ERP_MCP_TOKEN"] = token
    print(f"Done. ERP_MCP_TOKEN updated in {ENV_FILE}.")
    return token


_token_refresh_lock = threading.Lock()


def maybe_refresh_token(buffer_seconds: int = 300) -> None:
    """Refresh ERP_MCP_TOKEN if it expires within buffer_seconds.

    Safe to call from concurrent async/threaded contexts — uses a lock to
    ensure only one refresh runs at a time.  Never raises; logs a warning
    and returns silently on any error so inference is not interrupted.
    """
    token = os.environ.get("ERP_MCP_TOKEN", "")
    if not token:
        return
    try:
        payload_b64 = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
        exp = payload.get("exp", 0)
        if exp - time.time() > buffer_seconds:
            return
    except Exception:
        print("[token] WARNING: could not decode ERP_MCP_TOKEN expiry — skipping refresh check.", file=sys.stderr)
        return

    with _token_refresh_lock:
        # Re-check inside the lock in case another thread already refreshed.
        try:
            token = os.environ.get("ERP_MCP_TOKEN", "")
            payload_b64 = token.split(".")[1]
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
            exp = payload.get("exp", 0)
            if exp - time.time() > buffer_seconds:
                return
        except Exception:
            pass
        try:
            print("[token] ERP token expiring soon — refreshing ...")
            token = refresh_token()
            if token:
                print("[token] ERP token refreshed.")
        except Exception as exc:
            print(f"[token] WARNING: token refresh failed: {exc}", file=sys.stderr)


def main():
    refresh_token()


if __name__ == "__main__":
    main()
