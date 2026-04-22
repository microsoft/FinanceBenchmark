"""
_openai.py

OpenAI-specific inference logic for the Finance Copilot Benchmark.

Exports:
    get_config(cfg: dict) -> dict
        Takes the full parsed config.yaml and returns a flat dict of
        OpenAI-specific settings.

    make_process_fn(provider_cfg: dict, shared_cfg: dict) -> Callable
        Returns an async process_question(row, semaphore, index, total) -> dict
        with all config closed over.
"""

import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

from scripts.inference.result_schema import make_error_result
from scripts.shared.url_fetcher import async_fetch_sources


def get_config(cfg: dict) -> dict:
    """Extract OpenAI-specific settings from the parsed config.yaml dict."""
    root = Path(__file__).parent.parent.parent
    mcp_config = json.loads((root / cfg["shared"]["mcp_config_file"]).read_text(encoding="utf-8"))
    mcp_server_url = mcp_config["mcpServers"][cfg["shared"]["mcp_server_label"]]["url"]
    return {
        "model":          cfg["openai"]["model"],
        "max_tool_calls": cfg["openai"]["max_tool_calls"],
        "timeout":        cfg["shared"]["timeout"],
        "mcp_server_url": mcp_server_url,
    }


def make_process_fn(provider_cfg: dict, shared_cfg: dict):
    """
    Return an async process_question function with config closed over.

    Args:
        provider_cfg: Dict returned by get_config().
        shared_cfg:   Shared settings dict (output_dir, max_workers, etc.).

    Returns:
        Async callable: process_question(row, semaphore, index, total) -> dict
    """
    model               = provider_cfg["model"]
    max_tool_calls      = provider_cfg["max_tool_calls"]
    timeout             = provider_cfg["timeout"]
    mcp_server_url      = provider_cfg["mcp_server_url"]
    system_instructions = shared_cfg["system_instructions"]
    url_cache_dir       = Path(shared_cfg["url_cache_dir"])

    client = AsyncOpenAI()

    async def process_question(
        row: dict,
        semaphore: asyncio.Semaphore,
        index: int,
        total: int,
    ) -> dict:
        question = row["question"]
        async with semaphore:
            t_start = time.monotonic()
            try:
                resp = await client.responses.create(
                    model=model,
                    instructions=system_instructions,
                    timeout=timeout,
                    max_tool_calls=max_tool_calls,
                    tools=[
                        {
                            "type": "mcp",
                            "server_label": "erp",
                            "server_url": mcp_server_url,
                            "require_approval": "never",
                            "headers": {
                                "Authorization": f"Bearer {os.getenv('ERP_MCP_TOKEN')}"
                            },
                        },
                        {"type": "web_search_preview"},
                    ],
                    input=question,
                )
                inference_time_secs = round(time.monotonic() - t_start, 3)

            except Exception as e:
                return make_error_result(
                    question=question,
                    error=str(e),
                    inference_time_secs=round(time.monotonic() - t_start, 3),
                    extra_fields={"segment": row.get("segment")},
                )

            tool_calls_log = []
            answer_sequence_index = None
            web_search_log_indices = []  # positions in tool_calls_log of web_search entries

            for seq_idx, item in enumerate(resp.output):
                item_dict = item.model_dump()
                item_type = item_dict.get("type")

                if item_type == "message":
                    answer_sequence_index = seq_idx
                    # Extract URL citations from annotations as a proxy for web search output.
                    # The API doesn't expose the raw query or results on web_search_call items,
                    # so we back-fill citations (shared across all web searches in this response).
                    citations = [
                        {"url": ann.get("url"), "title": ann.get("title")}
                        for block in item_dict.get("content", [])
                        for ann in block.get("annotations", [])
                        if ann.get("type") == "url_citation"
                    ]
                    if citations:
                        for idx in web_search_log_indices:
                            tool_calls_log[idx]["output"] = citations
                elif item_type == "mcp_call":
                    tool_calls_log.append({
                        "sequence_index": seq_idx,
                        "tool":           item_dict.get("name"),
                        "input":          item_dict.get("arguments"),
                        "output":         item_dict.get("output"),
                        "success":        item_dict.get("output") is not None,
                    })
                elif item_type == "web_search_call":
                    web_search_log_indices.append(len(tool_calls_log))
                    tool_calls_log.append({
                        "sequence_index": seq_idx,
                        "tool":           "web_search",
                        "input":          None,  # query not exposed by OpenAI Responses API
                        "output":         None,  # filled in from message annotations below
                        "success":        item_dict.get("status") == "completed",
                    })

            successful = sum(1 for tc in tool_calls_log if tc["success"])

            answer = resp.output_text or ""
            sources = {}
            if row.get("plugin") == "finance_qa" and answer:
                sources = await async_fetch_sources(answer, url_cache_dir)

            print(f"  [{index}/{total}] done: {question[:70]}...")
            result = {
                "question":              question,
                "plugin":                row.get("plugin"),
                "segment":               row.get("segment"),
                "answer":                answer,
                "answer_sequence_index": answer_sequence_index,
                "tool_call_count":       len(tool_calls_log),
                "successful_tool_calls": successful,
                "tool_calls":            tool_calls_log,
                "inference_time_secs":   inference_time_secs,
            }
            if sources:
                result["sources"] = sources
            if row.get("scenario") is not None:
                result["scenario"] = row["scenario"]
            return result

    return process_question
