"""Tests for inference resume behavior.

Verifies that interrupted runs can be continued without duplicating entries,
and that error entries are retried by default.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference.inference import run_inference


_QUESTIONS = [
    {"question": "What is vendor V-1042 balance?",  "plugin": "erp_qa",      "segment": "accounts payable"},
    {"question": "List open POs over $10,000.",      "plugin": "erp_qa",      "segment": "accounts payable"},
    {"question": "What is Contoso Ltd CCC?",         "plugin": "finance_qa",  "segment": "financial performance"},
]


def _make_answer(question: str) -> dict:
    return {
        "question":              question,
        "segment":               "",
        "answer":                f"Answer: {question[:30]}",
        "answer_sequence_index": 1,
        "tool_call_count":       0,
        "successful_tool_calls": 0,
        "tool_calls":            [],
        "inference_time_secs":   0.05,
        "error":                 None,
    }


async def _mock_process_fn(row, semaphore, index, total):
    async with semaphore:
        return _make_answer(row["question"])


# ── Resume tests ──────────────────────────────────────────────────────────────


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_resume_no_duplicates(mock_refresh, tmp_path):
    """Resuming after 1 result is written produces exactly N results — no duplicates."""
    output_file = str(tmp_path / "answers_resume.json")

    # Pre-write one result
    partial = {
        "metadata": {"model": "m", "provider": "p", "config_snapshot": {}},
        "results":  [_make_answer(_QUESTIONS[0]["question"])],
    }
    with open(output_file, "w") as f:
        json.dump(partial, f)

    asyncio.run(run_inference(
        questions=_QUESTIONS,
        process_fn=_mock_process_fn,
        output_file=output_file,
        max_workers=1,
        metadata={"model": "m", "provider": "p", "config_snapshot": {}},
    ))

    with open(output_file) as f:
        data = json.load(f)

    results = data["results"]
    assert len(results) == len(_QUESTIONS), f"Expected {len(_QUESTIONS)}, got {len(results)}"

    questions_seen = [r["question"] for r in results]
    assert len(questions_seen) == len(set(questions_seen)), "Duplicate questions in results"


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_resume_preserves_existing_results(mock_refresh, tmp_path):
    """Pre-written result content is preserved unchanged after resume."""
    output_file = str(tmp_path / "answers_preserve.json")

    original = _make_answer(_QUESTIONS[0]["question"])
    original["answer"] = "ORIGINAL_ANSWER_DO_NOT_OVERWRITE"

    partial = {
        "metadata": {"model": "m", "provider": "p", "config_snapshot": {}},
        "results":  [original],
    }
    with open(output_file, "w") as f:
        json.dump(partial, f)

    asyncio.run(run_inference(
        questions=_QUESTIONS,
        process_fn=_mock_process_fn,
        output_file=output_file,
        max_workers=1,
        metadata={"model": "m", "provider": "p", "config_snapshot": {}},
    ))

    with open(output_file) as f:
        data = json.load(f)

    first = next(r for r in data["results"] if r["question"] == _QUESTIONS[0]["question"])
    assert first["answer"] == "ORIGINAL_ANSWER_DO_NOT_OVERWRITE"


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_resume_retries_error_entries_by_default(mock_refresh, tmp_path):
    """Error entries in a partial file are retried when no_retry_failed=False (default)."""
    output_file = str(tmp_path / "answers_retry.json")

    error_entry = {
        "question":              _QUESTIONS[0]["question"],
        "answer":                None,
        "error":                 "timeout",
        "tool_call_count":       0,
        "successful_tool_calls": 0,
        "tool_calls":            [],
        "inference_time_secs":   0.0,
    }
    partial = {
        "metadata": {"model": "m", "provider": "p", "config_snapshot": {}},
        "results":  [error_entry],
    }
    with open(output_file, "w") as f:
        json.dump(partial, f)

    asyncio.run(run_inference(
        questions=_QUESTIONS,
        process_fn=_mock_process_fn,
        output_file=output_file,
        max_workers=1,
        no_retry_failed=False,  # default behavior: retry errors
        metadata={"model": "m", "provider": "p", "config_snapshot": {}},
    ))

    with open(output_file) as f:
        data = json.load(f)

    retried = next(r for r in data["results"] if r["question"] == _QUESTIONS[0]["question"])
    assert retried.get("error") is None, "Error entry should have been retried and cleared"
    assert retried["answer"] is not None


@patch("scripts.inference.inference.maybe_refresh_token", return_value=None)
def test_resume_skips_error_entries_when_no_retry_failed(mock_refresh, tmp_path):
    """Error entries are skipped when no_retry_failed=True."""
    output_file = str(tmp_path / "answers_no_retry.json")

    error_entry = {
        "question":              _QUESTIONS[0]["question"],
        "answer":                None,
        "error":                 "timeout",
        "tool_call_count":       0,
        "successful_tool_calls": 0,
        "tool_calls":            [],
        "inference_time_secs":   0.0,
    }
    partial = {
        "metadata": {"model": "m", "provider": "p", "config_snapshot": {}},
        "results":  [error_entry, _make_answer(_QUESTIONS[1]["question"])],
    }
    with open(output_file, "w") as f:
        json.dump(partial, f)

    asyncio.run(run_inference(
        questions=_QUESTIONS,
        process_fn=_mock_process_fn,
        output_file=output_file,
        max_workers=1,
        no_retry_failed=True,
        metadata={"model": "m", "provider": "p", "config_snapshot": {}},
    ))

    with open(output_file) as f:
        data = json.load(f)

    not_retried = next(r for r in data["results"] if r["question"] == _QUESTIONS[0]["question"])
    assert not_retried.get("error") == "timeout", "Error entry should be preserved when no_retry_failed=True"
