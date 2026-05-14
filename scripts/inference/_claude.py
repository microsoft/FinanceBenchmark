"""
_claude.py

Claude-specific inference logic for the Finance Copilot Benchmark.

Exports:
    get_config(cfg: dict) -> dict
        Takes the full parsed config.yaml and returns a flat dict of
        Claude-specific settings.

    make_process_fn(provider_cfg: dict, shared_cfg: dict) -> Callable
        Returns an async process_question(row, semaphore, index, total) -> dict
        with all config closed over.
"""

import atexit
import asyncio
import json
import os
import tempfile
import time

from scripts.inference.result_schema import make_error_result
from scripts.shared.url_fetcher import is_useful_content


def get_config(cfg: dict) -> dict:
    """Extract Claude-specific settings from the parsed config.yaml dict."""
    return {
        "model":              cfg["claude"]["model"],
        "max_turns":          cfg["claude"]["max_turns"],
        "effort":             cfg["claude"].get("effort"),
        "timeout":            cfg["shared"]["timeout"],
        "mcp_config_file":    cfg["shared"]["mcp_config_file"],
        "mcp_server_label":   cfg["shared"].get("mcp_server_label"),
        "blocked_mcp_tools":  cfg["shared"].get("blocked_mcp_tools", []),
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
    max_turns           = provider_cfg["max_turns"]
    effort              = provider_cfg.get("effort")
    timeout             = provider_cfg["timeout"]
    mcp_config_file     = provider_cfg["mcp_config_file"]
    mcp_server_label    = provider_cfg.get("mcp_server_label")
    blocked_mcp_tools   = provider_cfg["blocked_mcp_tools"]
    system_instructions = shared_cfg["system_instructions"]

    # Build a filtered MCP config containing only the selected server.
    # The subprocess runs from a temp directory so it can't auto-discover the
    # project's .mcp.json (which would load all servers regardless of --mcp-config).
    with open(mcp_config_file) as _f:
        _full_mcp = json.load(_f)
    _servers = _full_mcp.get("mcpServers", {})
    if mcp_server_label and mcp_server_label in _servers:
        _filtered = {"mcpServers": {mcp_server_label: _servers[mcp_server_label]}}
    else:
        _filtered = _full_mcp
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_filtered, _tmp)
    _tmp.close()
    atexit.register(os.unlink, _tmp.name)
    effective_mcp_config = os.path.abspath(_tmp.name)

    # Subprocess cwd: a neutral temp dir with no .mcp.json so the project's
    # multi-server config isn't auto-loaded alongside --mcp-config.
    _subprocess_cwd = tempfile.mkdtemp()
    atexit.register(lambda d=_subprocess_cwd: os.rmdir(d) if os.path.isdir(d) else None)

    async def process_question(
        row: dict,
        semaphore: asyncio.Semaphore,
        index: int,
        total: int,
    ) -> dict:
        question = row["question"]
        async with semaphore:
            row_timeout = row.get("timeout", timeout)
            deny_list = ["Read", "Write", "Edit", "Bash", "Grep", "Glob"] + list(blocked_mcp_tools)
            settings = json.dumps({"permissions": {"allow": ["WebSearch"], "deny": deny_list}})
            cmd = [
                "claude", "-p", question,
                "--output-format", "stream-json",
                "--mcp-config", effective_mcp_config,
                "--model", model,
                "--max-turns", str(max_turns),
                "--verbose",
                "--dangerously-skip-permissions",
                "--system-prompt", system_instructions,
                "--settings", settings,
            ]
            if effort:
                cmd += ["--effort", effort]

            t_start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_subprocess_cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=row_timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return make_error_result(
                    question=question,
                    error=f"inference timeout after {row_timeout}s",
                    inference_time_secs=round(time.monotonic() - t_start, 3),
                    extra_fields={"segment": row.get("segment")},
                )
            except BaseException:
                # CancelledError (Ctrl+C) or any other unexpected interruption —
                # kill the subprocess so it doesn't become a zombie, then re-raise.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                raise
            inference_time_secs = round(time.monotonic() - t_start, 3)

            tool_calls_log = []
            answer = ""
            answer_sequence_index = None
            seq_idx = 0

            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            answer = block["text"]
                            answer_sequence_index = seq_idx
                            seq_idx += 1
                        elif block.get("type") == "tool_use":
                            tool_calls_log.append({
                                "_id":            block["id"],
                                "sequence_index": seq_idx,
                                "tool":           block["name"],
                                "input":          block["input"],
                                "output":         None,
                                "success":        False,
                            })
                            seq_idx += 1

                elif event_type == "user":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "tool_result":
                            for tc in tool_calls_log:
                                if tc.get("_id") == block.get("tool_use_id"):
                                    content = block.get("content", "")
                                    output = (
                                        content[0].get("text", "") if isinstance(content, list)
                                        else str(content)
                                    )
                                    tc["output"] = output
                                    tc["success"] = bool(output) and not output.startswith("Request failed with status code")
                                    break

                elif event_type == "result" and event.get("is_error"):
                    return make_error_result(
                        question=question,
                        error=event.get("result", "Unknown error"),
                        inference_time_secs=inference_time_secs,
                        extra_fields={"segment": row.get("segment")},
                    )

            if proc.returncode != 0 and not answer:
                return make_error_result(
                    question=question,
                    error=stderr.decode().strip(),
                    inference_time_secs=inference_time_secs,
                    extra_fields={"segment": row.get("segment")},
                )

            for tc in tool_calls_log:
                tc.pop("_id", None)

            successful = sum(1 for tc in tool_calls_log if tc["success"])

            # Extract sources from successful WebFetch tool calls
            sources = {}
            if row.get("plugin") == "finance_qa":
                for tc in tool_calls_log:
                    if tc.get("tool") == "WebFetch" and tc.get("success"):
                        url = tc.get("input", {}).get("url", "")
                        content = tc.get("output", "")
                        if url and is_useful_content(content):
                            sources[url] = content

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
