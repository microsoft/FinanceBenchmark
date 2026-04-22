import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference.result_schema import make_error_result, InferenceResult, ToolCall


def test_make_error_result_basic():
    result = make_error_result("What is the balance?", "timeout", 5.0)
    assert result["question"] == "What is the balance?"
    assert result["error"] == "timeout"
    assert result["answer"] is None
    assert result["tool_call_count"] == 0
    assert result["successful_tool_calls"] == 0
    assert result["tool_calls"] == []
    assert result["inference_time_secs"] == 5.0


def test_make_error_result_extra_fields():
    result = make_error_result("q", "err", 1.0, extra_fields={"plugin": "erp_qa"})
    assert result["plugin"] == "erp_qa"
    assert result["error"] == "err"


def test_make_error_result_segment_is_none():
    result = make_error_result("q", "err", 0.0)
    assert result["segment"] is None
    assert result["answer_sequence_index"] is None


def test_inference_result_has_required_keys():
    result = make_error_result("q", "err", 1.0)
    required = {
        "question", "segment", "answer", "answer_sequence_index",
        "tool_call_count", "successful_tool_calls", "tool_calls",
        "inference_time_secs", "error",
    }
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_tool_call_with_success_false():
    tc: ToolCall = {
        "sequence_index": 0,
        "tool": "erp_query",
        "input": {"query": "SELECT * FROM Vendors"},
        "output": "Connection refused",
        "success": False,
    }
    assert tc["success"] is False
    assert tc["output"] == "Connection refused"


def test_tool_call_output_field_present():
    tc: ToolCall = {
        "sequence_index": 1,
        "tool": "WebSearch",
        "input": {"query": "test"},
        "output": "some result",
        "success": True,
    }
    assert "output" in tc
