"""Tests for plugin routing and citation-counting logic in evaluate.py.

LLM calls are mocked via unittest.mock.patch so no real API calls are made.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.evaluate import count_markdown_citations, evaluate_tag


# ── count_markdown_citations ────────────────────────────────────────────────


def test_citation_count_three_links():
    text = (
        "See [report](https://example.com/a), "
        "[data](https://example.com/b), and "
        "[filing](https://example.com/c)."
    )
    assert count_markdown_citations(text) == 3


def test_citation_count_deduplicated():
    text = "[foo](https://example.com/same) and [bar](https://example.com/same)"
    assert count_markdown_citations(text) == 1


def test_citation_count_malformed_empty_url():
    text = "[text]()"
    assert count_markdown_citations(text) == 0


def test_citation_count_empty_string():
    assert count_markdown_citations("") == 0


def test_citation_count_no_links():
    assert count_markdown_citations("No hyperlinks here.") == 0


# ── evaluate_tag routing ────────────────────────────────────────────────────


def _make_tag_entry(tag: str, assertions=None):
    if assertions is None:
        assertions = [{"text": "response covers the topic", "level": "critical"}]
    return {"tag": tag, "metric": True, "assertions": assertions}


def _make_mock_pred(scores=None):
    pred = MagicMock()
    pred.assertion_scores = "[1.0]"
    pred.reasoning = "looks good"
    return pred


def _mock_lm_call(module, **kwargs):
    return _make_mock_pred(), {"input_tokens": 10, "output_tokens": 5}


def test_finance_qa_citations_no_llm_called():
    """finance_qa citations uses markdown count — judge is never called."""
    judge = MagicMock()
    groundedness_judge = MagicMock()
    tag_entry = _make_tag_entry("citations")
    answer_with_link = "See [report](https://example.com/report) for details."

    result = evaluate_tag(
        tag_entry,
        question="How many citations?",
        answer=answer_with_link,
        plugin="finance_qa",
        answer_entry={},
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    judge.assert_not_called()
    groundedness_judge.assert_not_called()
    assert result["type"] == "scored"
    # All assertions should score 1.0 (has ≥1 citation)
    for score in result["assertion_scores"].values():
        assert score == 1.0


def test_finance_qa_citations_zero_when_no_links():
    judge = MagicMock()
    groundedness_judge = MagicMock()
    tag_entry = _make_tag_entry("citations")

    result = evaluate_tag(
        tag_entry,
        question="q",
        answer="No links here.",
        plugin="finance_qa",
        answer_entry={},
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    assert result["type"] == "scored"
    for score in result["assertion_scores"].values():
        assert score == 0.0


@patch("scripts.evaluation.evaluate._lm_call", side_effect=_mock_lm_call)
def test_erp_qa_citations_uses_llm(mock_lm):
    """erp_qa citations tag → standard LLM judge, not the special citation counter."""
    judge = MagicMock()
    groundedness_judge = MagicMock()
    tag_entry = _make_tag_entry("citations")

    result = evaluate_tag(
        tag_entry,
        question="q",
        answer="answer",
        plugin="erp_qa",
        answer_entry={},
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    assert mock_lm.called
    assert result["type"] == "scored"
    # The mocked _lm_call was invoked with the judge (first positional arg)
    call_args = mock_lm.call_args
    assert call_args[0][0] is judge


@patch("scripts.evaluation.evaluate._lm_call", side_effect=_mock_lm_call)
def test_groundedness_uses_groundedness_judge(mock_lm):
    """groundedness tag uses groundedness_judge, not the standard judge."""
    judge = MagicMock()
    groundedness_judge = MagicMock()
    tag_entry = _make_tag_entry("groundedness")
    answer_entry = {
        "tool_calls": [
            {"tool": "WebSearch", "output": "some source content", "success": True}
        ]
    }

    result = evaluate_tag(
        tag_entry,
        question="q",
        answer="answer",
        plugin="finance_qa",
        answer_entry=answer_entry,
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    assert mock_lm.called
    call_args = mock_lm.call_args
    assert call_args[0][0] is groundedness_judge


@patch("scripts.evaluation.evaluate._lm_call", side_effect=_mock_lm_call)
def test_groundedness_skipped_when_no_sources(mock_lm):
    """groundedness is skipped when no source content is available."""
    judge = MagicMock()
    groundedness_judge = MagicMock()
    tag_entry = _make_tag_entry("groundedness")

    result = evaluate_tag(
        tag_entry,
        question="q",
        answer="answer",
        plugin="finance_qa",
        answer_entry={},  # no tool_calls, no sources
        judge=judge,
        groundedness_judge=groundedness_judge,
    )

    assert result["type"] == "skipped"
    mock_lm.assert_not_called()
