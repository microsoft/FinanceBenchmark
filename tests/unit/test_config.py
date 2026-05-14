"""
Setup verification tests — check that the local environment is configured correctly
before running inference or evaluation.

Run these after cloning and filling in config.yaml / .env:

    uv run pytest tests/unit/test_config.py -v
"""
import json
import os
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent.parent

load_dotenv(REPO_ROOT / ".env")


@pytest.fixture(scope="module")
def config() -> dict:
    path = REPO_ROOT / "config.yaml"
    assert path.exists(), "config.yaml not found at repo root"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Inference config ───────────────────────────────────────────────────────────

def test_config_loads(config):
    assert isinstance(config, dict)


def test_claude_config(config):
    assert "claude" in config, "Missing 'claude' section in config.yaml"
    assert config["claude"].get("model"), "claude.model must be set"
    assert config["claude"].get("max_turns"), "claude.max_turns must be set"


def test_openai_config(config):
    assert "openai" in config, "Missing 'openai' section in config.yaml"
    assert config["openai"].get("model"), "openai.model must be set"


def test_shared_config(config):
    shared = config.get("shared", {})
    for key in ("max_workers", "output_dir", "mcp_config_file"):
        assert key in shared, f"shared.{key} is missing from config.yaml"


def test_mcp_config_file_exists(config):
    path = REPO_ROOT / config["shared"]["mcp_config_file"]
    assert path.exists(), (
        f"MCP config file not found: {path}. "
        "Copy example.mcp.json to .mcp.json and fill in your server details."
    )


def test_mcp_config_valid(config):
    path = REPO_ROOT / config["shared"]["mcp_config_file"]
    if not path.exists():
        pytest.skip("MCP config file not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "mcpServers" in data, ".mcp.json must have a 'mcpServers' key"
    assert isinstance(data["mcpServers"], dict), "'mcpServers' must be an object"


def test_mcp_server_label_exists(config):
    label = config["shared"].get("mcp_server_label")
    if not label:
        pytest.skip("mcp_server_label not set in config.yaml — skipping")
    path = REPO_ROOT / config["shared"]["mcp_config_file"]
    if not path.exists():
        pytest.skip("MCP config file not found")
    servers = json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {})
    assert label in servers, (
        f"mcp_server_label '{label}' not found in {path}. "
        f"Available servers: {sorted(servers)}"
    )


# ── Evaluation config ──────────────────────────────────────────────────────────

def test_evaluation_config(config):
    assert "evaluation" in config, "Missing 'evaluation' section in config.yaml"
    assert config["evaluation"].get("judge_deployment"), (
        "evaluation.judge_deployment must be set in config.yaml"
    )


def test_openai_api_key_set():
    key = os.getenv("OPENAI_API_KEY")
    assert key, (
        "OPENAI_API_KEY is not set. Add it to .env — required for evaluation."
    )
