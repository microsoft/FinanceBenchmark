"""Smoke test: inference → evaluate pipeline using fixtures and mocks.

No real API or MCP calls are made. All LLM/tool interactions are patched.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference.inference import run_inference
from scripts.evaluation.evaluate import evaluate_entry

_FIXTURE_DATASET = Path(__file__).parent.parent / "fixtures" / "dataset_fixture.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_mock_process_fn(questions):
    """Return a mock async process_fn that echoes back a fake answer for each question."""
    async def process_fn(row, semaphore, index, total):
        async with semaphore:
            return {
                "question":             row["question"],
                "segment":              row.get("segment", ""),
                "plugin":               row.get("plugin", ""),
                "answer":               f"Mock answer for: {row['question'][:40]}",
                "answer_sequence_index": 1,
                "tool_call_count":      1,
                "successful_tool_calls": 1,
                "tool_calls": [
                    {"sequence_index": 0, "tool": "WebSearch",
                     "input": {"query": "test"}, "output": "mock result", "success": True}
                ],
                "inference_time_secs":  0.1,
                "error":                None,
                "sources":              {},
            }
    return process_fn


# ── Inference smoke test ───────────────────────────────────────────────────────


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_inference_output_has_envelope_format(mock_refresh, tmp_path, fixture_dataset):
    """run_inference() produces a file with metadata + results envelope."""
    # Convert fixture to the question-key format used by inference
    questions = []
    for entry in fixture_dataset:
        row = {k: v for k, v in entry.items() if k != "query"}
        row["question"] = entry.get("question", entry.get("query", ""))
        questions.append(row)

    output_file = str(tmp_path / "answers_test.json")
    metadata = {
        "model": "test-model",
        "provider": "claude",
        "inf_run_id": "smoke",
        "run_timestamp": "2026-04-22T00:00:00Z",
        "config_hash": "abc123",
        "config_snapshot": {},
    }

    asyncio.run(run_inference(
        questions=questions,
        process_fn=_make_mock_process_fn(questions),
        output_file=output_file,
        max_workers=2,
        metadata=metadata,
    ))

    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data, "Output missing metadata key"
    assert "results" in data, "Output missing results key"
    assert data["metadata"]["provider"] == "claude"
    assert len(data["results"]) == len(questions)


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_inference_result_fields(mock_refresh, tmp_path, fixture_dataset):
    """Each result has the expected fields."""
    questions = []
    for entry in fixture_dataset:
        row = dict(entry)
        row["question"] = row.pop("query", row.get("question", ""))
        questions.append(row)

    output_file = str(tmp_path / "answers_fields.json")

    asyncio.run(run_inference(
        questions=questions,
        process_fn=_make_mock_process_fn(questions),
        output_file=output_file,
        max_workers=1,
        metadata={"model": "m", "provider": "p", "config_snapshot": {}},
    ))

    with open(output_file) as f:
        data = json.load(f)

    for result in data["results"]:
        assert "question" in result
        assert "answer" in result
        assert "tool_calls" in result


# ── Evaluation smoke test ──────────────────────────────────────────────────────


def _mock_lm_call_for_eval(module, **kwargs):
    pred = MagicMock()
    pred.assertion_scores = "[1.0]"
    pred.reasoning = "looks good"
    return pred, {"input_tokens": 5, "output_tokens": 3}


@patch("scripts.evaluation.evaluate._lm_call", side_effect=_mock_lm_call_for_eval)
def test_evaluate_entry_has_tag_scores(mock_lm, fixture_dataset):
    """evaluate_entry() returns per-tag scores and token_usage for a fixture question."""
    judge              = MagicMock()
    groundedness_judge = MagicMock()

    # Pick the first erp_qa entry from fixture
    entry = next(e for e in fixture_dataset if e.get("plugin") == "erp_qa")
    question = entry.get("query") or entry.get("question")
    tags     = entry["tags"]

    answer_entry = {
        "question": question,
        "answer":   "The outstanding balance for vendor V-1042 is $45,230.00 USD.",
        "tool_calls": [
            {"tool": "erp_query", "output": "VendorId: V-1042, Balance: 45230.00", "success": True}
        ],
    }

    result = evaluate_entry(
        question=question,
        answer=answer_entry["answer"],
        plugin="erp_qa",
        tags=tags,
        answer_entry=answer_entry,
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    assert "tag_scores" in result
    assert "token_usage" in result
    assert "overall_score" in result
    assert result["overall_score"] is not None
    # Metric tags should have scores
    metric_tags = [t["tag"] for t in tags if t.get("metric")]
    for tag in metric_tags:
        assert tag in result["tag_scores"], f"Missing tag score for: {tag}"


@patch("scripts.evaluation.evaluate._lm_call", side_effect=_mock_lm_call_for_eval)
def test_evaluate_entry_token_usage_structure(mock_lm, fixture_dataset):
    """token_usage in result has input_tokens, output_tokens, total_tokens."""
    judge              = MagicMock()
    groundedness_judge = MagicMock()

    entry    = fixture_dataset[0]
    question = entry.get("query") or entry.get("question")

    result = evaluate_entry(
        question=question,
        answer="Some answer.",
        plugin=entry.get("plugin", "erp_qa"),
        tags=entry["tags"],
        answer_entry={"question": question, "answer": "Some answer.", "tool_calls": []},
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    usage = result["token_usage"]
    assert "input_tokens"  in usage
    assert "output_tokens" in usage
    assert "total_tokens"  in usage
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
