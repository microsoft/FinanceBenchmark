import sys
import os
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis.runs_db import init_db
from scripts.inference.result_schema import InferenceResult, ToolCall

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dataset() -> list[dict]:
    """Load tests/fixtures/dataset_fixture.yaml and return as a list of dicts."""
    with open(_FIXTURES_DIR / "dataset_fixture.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite runs database; yield connection; close on teardown."""
    db_path = tmp_path / "runs.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def mock_inference_result() -> InferenceResult:
    """Return a sample InferenceResult dict for use in tests."""
    tool_call: ToolCall = {
        "sequence_index": 0,
        "tool": "WebSearch",
        "input": {"query": "Contoso Ltd cash conversion cycle"},
        "output": "Contoso Ltd CCC: 42 days (FY2025).",
        "success": True,
    }
    result: InferenceResult = {
        "question": "What is the current cash conversion cycle for Contoso Ltd?",
        "segment": "financial performance & financial health",
        "answer": "The cash conversion cycle for Contoso Ltd is 42 days as of FY2025.",
        "answer_sequence_index": 1,
        "tool_call_count": 1,
        "successful_tool_calls": 1,
        "tool_calls": [tool_call],
        "inference_time_secs": 3.14,
        "error": None,
        "sources": {"https://example-finance-data.com/contoso": "CCC: 42 days"},
    }
    return result
