"""
result_schema.py

Defines the canonical inference result schema and error result factory.

InferenceResult documents all fields that may appear in answers_*.json files.
make_error_result() creates error records with zeros/nulls for numeric/list fields.
RunMetadata documents the metadata envelope written at the top of each output file.
"""

from typing import TypedDict, Any, Optional


class ToolCall(TypedDict, total=False):
    """Individual tool call record."""
    sequence_index: int
    tool: str
    input: dict[str, Any]
    output: Optional[str]
    success: bool


class InferenceResult(TypedDict, total=False):
    """
    Canonical output record shape for inference scripts.

    Fields marked total=False means all are optional, though in practice
    most scripts populate at least question, segment, and answer.

    Fields:
        question: The input question/utterance
        segment: Question category (e.g., "Aged balances", "Invoices")
        answer: The model's text response (None if error)
        answer_sequence_index: Position of answer relative to tool calls
        tool_call_count: Total number of tool calls made
        successful_tool_calls: Number of successful tool calls
        tool_calls: List of tool call records with details
        inference_time_secs: Wall-clock time spent on inference
        error: Error message if inference failed (None if success)
        sources: Optional dict mapping cited URL → source snippet text, embedded
                 at inference/prep time to avoid stale Playwright fetches at eval
    """
    question: str
    segment: Optional[str]
    answer: Optional[str]
    answer_sequence_index: Optional[int]
    tool_call_count: int
    successful_tool_calls: int
    tool_calls: list[ToolCall]
    inference_time_secs: float
    error: Optional[str]
    sources: Optional[dict[str, str]]


class RunMetadata(TypedDict, total=False):
    """
    Metadata envelope written at the top of each inference output file.

    Fields:
        model: The model name used for inference (e.g., "claude-sonnet-4-5")
        provider: The inference provider ("claude" or "openai")
        run_id: Optional identifier for variance measurement runs; null if not provided
        run_timestamp: ISO 8601 UTC timestamp set once when inference starts
        config_hash: First 12 hex characters of the sha256 of config_snapshot
                     serialized with sorted keys
        config_snapshot: Flattened config values active for this run, combining
                         shared settings with provider-specific settings
    """
    model: str
    provider: str
    run_id: Optional[str]
    run_timestamp: str
    config_hash: str
    config_snapshot: dict[str, Any]


def make_error_result(
    question: str,
    error: str,
    inference_time_secs: float,
    extra_fields: Optional[dict[str, Any]] = None,
) -> InferenceResult:
    """
    Create an error inference result with zeros/nulls for numeric and list fields.

    Args:
        question: The input question that failed
        error: Error message describing the failure
        inference_time_secs: Time spent attempting inference
        extra_fields: Optional dict of feature-specific fields (e.g., {"segment": "..."})

    Returns:
        InferenceResult with error populated and defaults for all other fields
    """
    result: InferenceResult = {
        "question":              question,
        "segment":               None,
        "answer":                None,
        "answer_sequence_index": None,
        "tool_call_count":       0,
        "successful_tool_calls": 0,
        "tool_calls":            [],
        "inference_time_secs":   inference_time_secs,
        "error":                 error,
    }

    if extra_fields:
        result.update(extra_fields)  # type: ignore

    return result
