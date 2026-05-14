"""
inference.py

Single entry point for Finance Agent Benchmark inference.

Usage:
  uv run scripts/inference/inference.py --provider claude          # all questions
  uv run scripts/inference/inference.py --provider openai -n 10   # first 10 only

Output: results/answers_{model}_{inf_slug}.json  (e.g. answers_gpt-5.2_panda.json)
Each result includes: question, segment, answer, answer_sequence_index,
tool_call_count, successful_tool_calls, tool_calls, inference_time_secs.
"""

import argparse
import asyncio
import base64
import sys
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional

import yaml
from dotenv import load_dotenv

from scripts.inference.dataloader import load_questions
from scripts.inference.result_schema import make_error_result

_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_ROOT))
from refresh_erp_token import maybe_refresh_token  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_azure_account() -> None:
    """Exit with an error if logged into the account ERP_BLOCKED_AZ_USER.

    The ERP demo account lacks access to the ACA resources required for MCP calls.
    Silently skips if az is unavailable.
    """
    import subprocess

    _BLOCKED_USER = os.getenv("ERP_BLOCKED_AZ_USER", "").lower()
    if not _BLOCKED_USER:
        return
    try:
        result = subprocess.run(
            "az account show", shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return
        account = json.loads(result.stdout)
        logged_in_user = account.get("user", {}).get("name", "")
        if logged_in_user.lower() == _BLOCKED_USER:
            print(
                f"ERROR: Logged into Azure as the demo account ({logged_in_user}).\n"
                "Inference requires ACA resources deployed under the real user account.\n"
                "Run `az login` to switch accounts before running inference."
            )
            raise SystemExit(1)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        print("WARNING: Could not verify Azure account. Make sure you're logged into the correct account with `az login`.")
        pass  # az CLI not available or timed out — don't block


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _shared_cfg(cfg: dict) -> dict:
    output_dir = cfg["shared"]["output_dir"]
    return {
        "input_file":          cfg["data_prep"]["output"],
        "output_dir":          output_dir,
        "url_cache_dir":       str(Path(output_dir) / "url_cache"),
        "max_workers":         cfg["shared"]["max_workers"],
        "system_instructions": cfg["shared"]["system_instructions"],
    }


def _build_metadata(
    provider: str,
    model: str,
    inf_slug: str,
    snapshot: dict,
) -> dict:
    """Build the RunMetadata envelope dict for this run."""
    config_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]
    return {
        "model":           model,
        "provider":        provider,
        "inf_run_id":      inf_slug,
        "run_timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash":     config_hash,
        "config_snapshot": snapshot,
    }


def compute_inference_config_hash(cfg: dict, provider: str) -> str:
    """Compute the inference config_hash that would be produced for a given config/provider.

    Mirrors the snapshot dict built in ``main()`` so that ``run_benchmark.py``
    can predict the hash before running inference and look it up in runs.db.

    Args:
        cfg:      Parsed config.yaml dict (as returned by ``_load_cfg()``).
        provider: ``"claude"`` or ``"openai"``.

    Returns:
        First 12 hex characters of the sha256 of the serialised snapshot (same
        as the ``config_hash`` field stored in the inference metadata envelope).
    """
    shared = _shared_cfg(cfg)

    if provider == "claude":
        from scripts.inference._claude import get_config as _get_cfg
    else:
        from scripts.inference._openai import get_config as _get_cfg

    provider_cfg = _get_cfg(cfg)

    # Resolve run-time defaults from config (shuffle / retry_failed)
    inference_cfg = cfg.get("inference", {})
    shuffle      = inference_cfg.get("shuffle", True)
    retry_failed = inference_cfg.get("retry_failed", True)

    snapshot = {
        "input_file":          shared["input_file"],
        "output_dir":          shared["output_dir"],
        "max_workers":         shared["max_workers"],
        "shuffle":             shuffle,
        "retry_failed":        retry_failed,
        "system_instructions": shared["system_instructions"],
        **{k: v for k, v in provider_cfg.items()},
    }
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Orchestrator (inlined from the former orchestrator.py)
# ---------------------------------------------------------------------------

def _write_output(output_file: str, results: list, metadata: Optional[dict]) -> None:
    """Write results to output_file using the envelope format if metadata is provided."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    payload = {"metadata": metadata, "results": results} if metadata is not None else results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_existing(output_file: str) -> tuple[dict[str, dict], Optional[dict]]:
    """Load existing results from output_file.

    Returns:
        (existing_by_question, existing_metadata) where existing_metadata is None
        if the file used the old flat-list format.
    """
    existing: dict[str, dict] = {}
    existing_metadata: Optional[dict] = None

    if not os.path.exists(output_file):
        return existing, existing_metadata

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "results" in data:
            items = data["results"]
            existing_metadata = data.get("metadata")
        else:
            return existing, existing_metadata

        for item in items:
            if item.get("question"):
                existing[item["question"]] = item

    except (json.JSONDecodeError, KeyError, OSError):
        pass

    return existing, existing_metadata


async def run_inference(
    questions: list[dict],
    process_fn: Callable,
    output_file: str,
    max_workers: int,
    *,
    limit: Optional[int] = None,
    no_retry_failed: bool = False,
    question_key: str = "question",
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run inference on a list of questions with resume and retry support.

    Args:
        questions:        List of question dicts. Each dict must contain question_key.
        process_fn:       Async callable (row, semaphore, index, total) -> dict.
        output_file:      Path to JSON file where results are written.
        max_workers:      Number of concurrent workers (semaphore limit).
        limit:            If provided, only process the first N questions.
        no_retry_failed:  If True, skip questions that previously errored.
        question_key:     Key in question dict that identifies the question text.
        metadata:         Optional metadata dict for the output envelope.
    """
    if limit is not None:
        questions = questions[:limit]

    existing, _existing_metadata = _load_existing(output_file)
    if existing:
        print(f"Resuming: {len(existing)} results already in {output_file}")

    def is_done(row: dict) -> bool:
        item = existing.get(row[question_key])
        if item is None:
            return False
        if not no_retry_failed and item.get("error"):
            return False
        return True

    done_results = [existing[r[question_key]] for r in questions if is_done(r)]
    pending_rows = [r for r in questions if not is_done(r)]

    retry_count = sum(
        1 for r in questions
        if r[question_key] in existing and existing[r[question_key]].get("error") and not no_retry_failed
    )
    status = f"{len(done_results)} already done"
    if retry_count:
        status += f", {retry_count} error(s) queued for retry"
    print(
        f"Processing {len(pending_rows)} remaining questions "
        f"({status}) with {max_workers} concurrent workers...\n"
    )

    # completed_by_question tracks results as they arrive, for ordered flushing.
    completed_by_question: dict[str, dict] = {r.get("question"): r for r in done_results}
    write_lock = asyncio.Lock()

    def _flush() -> None:
        """Write results in original dataset order, with completed entries filled in."""
        ordered = [completed_by_question[r[question_key]] for r in questions if r[question_key] in completed_by_question]
        _write_output(output_file, ordered, metadata)

    async def process_and_save(row: dict, semaphore: asyncio.Semaphore, index: int, total: int) -> dict:
        maybe_refresh_token()
        try:
            result = await process_fn(row, semaphore, index, total)
        except asyncio.CancelledError:
            # Ctrl+C or external cancellation — let it propagate so the gather
            # can shut down cleanly. Progress already flushed by completed tasks.
            raise
        except Exception as exc:
            print(f"  ERROR on question {index}: {type(exc).__name__}: {exc}")
            extra = {k: v for k, v in row.items() if k != question_key}
            result = make_error_result(
                question=row[question_key],
                error=f"{type(exc).__name__}: {exc}",
                inference_time_secs=0.0,
                extra_fields=extra,
            )
        async with write_lock:
            completed_by_question[row[question_key]] = result
            _flush()
        return result

    semaphore = asyncio.Semaphore(max_workers)
    total     = len(questions)
    tasks     = [
        process_and_save(row, semaphore, len(done_results) + i + 1, total)
        for i, row in enumerate(pending_rows)
    ]
    cancelled = False
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        cancelled = True
        n_done = len(completed_by_question) - len(done_results)
        n_pending = len(pending_rows) - n_done
        print(f"\n  Interrupted — {n_done} new questions completed before stop; {n_pending} will be retried on resume.")
        _flush()

    if cancelled:
        return

    # Final reconciliation: any pending question still missing gets an explicit error entry.
    # This catches edge cases where a task vanished without raising a catchable exception.
    silent_failures = [row for row in pending_rows if row[question_key] not in completed_by_question]
    if silent_failures:
        print(f"\n  WARNING: {len(silent_failures)} question(s) produced no result and no exception — recording as errors:")
        for row in silent_failures:
            print(f"    - {row[question_key][:80]}")
            extra = {k: v for k, v in row.items() if k != question_key}
            completed_by_question[row[question_key]] = make_error_result(
                question=row[question_key],
                error="silent failure: task produced no result and no exception",
                inference_time_secs=0.0,
                extra_fields=extra,
            )
        _flush()

    final_results = [completed_by_question[r[question_key]] for r in questions if r[question_key] in completed_by_question]
    expected = len(questions)
    n_errors = sum(1 for r in final_results if r.get("error"))
    print(f"\nDone. {len(final_results)}/{expected} questions recorded ({n_errors} errors) -> {output_file}\n")
    if len(final_results) < expected:
        print(f"  WARNING: {expected - len(final_results)} question(s) from the dataset were not processed and are absent from the output.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    provider: str,
    provider_cfg: dict,
    shared: dict,
    process_fn: Callable,
    limit: Optional[int],
    retry_failed: bool,
    shuffle: bool,
    output_file: Optional[str],
) -> None:
    model      = provider_cfg["model"]
    output_dir = shared["output_dir"]
    max_workers = shared["max_workers"]
    input_file  = shared["input_file"]

    if output_file is None:
        from scripts.tracking.db import generate_unique_slug
        inf_slug = generate_unique_slug(Path(output_dir), f"answers_{model}_")
        output_file = f"{output_dir}/answers_{model}_{inf_slug}.json"
    else:
        # Output file explicitly provided — derive slug from filename for metadata
        from scripts.tracking.db import _extract_slug_from_filename
        inf_slug = _extract_slug_from_filename(output_file, "answers_") or Path(output_file).stem

    questions = load_questions(input_file, shuffle=shuffle)

    # Build config snapshot for metadata (shared keys first, then provider-specific keys).
    # system_instructions lives in shared config and is included via the shared keys.
    snapshot = {
        "input_file":          input_file,
        "output_dir":          output_dir,
        "max_workers":         max_workers,
        "shuffle":             shuffle,
        "retry_failed":        retry_failed,
        "system_instructions": shared["system_instructions"],
        **{k: v for k, v in provider_cfg.items()},
    }
    metadata = _build_metadata(provider, model, inf_slug, snapshot)

    # If resuming, restore inf_slug from the existing file
    _, existing_meta = _load_existing(output_file)
    if existing_meta:
        inf_slug = existing_meta.get("inf_run_id") or inf_slug
        metadata["inf_run_id"]     = inf_slug

    await run_inference(
        questions=questions,
        process_fn=process_fn,
        output_file=output_file,
        max_workers=max_workers,
        limit=limit,
        no_retry_failed=not retry_failed,
        metadata=metadata,
    )

    # Register in runs table (eval_file = NULL until evaluation completes)
    try:
        db_path = _ROOT / "results" / "runs.db"
        from scripts.tracking.db import init_db, register_inference_only
        db_conn = init_db(db_path)
        row_id = register_inference_only(db_conn, output_file, metadata)
        db_conn.close()
        print(f"  [runs.db] Registered inference run id={row_id}: {output_file}")
    except Exception as exc:
        print(f"  [warn] Could not register inference run in runs.db: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark inference for a given provider."
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "openai"],
        required=False,
        help="Inference provider to use",
        default="openai"
    )
    parser.add_argument("-n", "--limit", type=int, default=None, metavar="N",
                        help="Only process the first N questions (default: all)")
    parser.add_argument("--retry-failed", default=None, action=argparse.BooleanOptionalAction,
                        help="Retry questions that previously errored (default: from config.yaml inference.retry_failed)")
    parser.add_argument("--shuffle", default=None, action=argparse.BooleanOptionalAction,
                        help="Shuffle the questions before processing (default: from config.yaml inference.shuffle)")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Override the model from config.yaml")
    parser.add_argument("--input-file", default=None, metavar="FILE",
                        help="Override the input dataset path from config.yaml")
    parser.add_argument("--output-file", default=None, metavar="FILE",
                        help="Override the output file path entirely")
    parser.add_argument("--max-workers", type=int, default=None, metavar="N",
                        help="Override the number of concurrent workers from config.yaml")
    args = parser.parse_args()

    load_dotenv()
    _check_azure_account()

    cfg = _load_cfg()
    shared = _shared_cfg(cfg)

    # Resolve effective boolean run-time settings: CLI flag wins if passed, else config default.
    inference_cfg = cfg.get("inference", {})
    shuffle = (args.shuffle) if args.shuffle is not None else inference_cfg.get("shuffle", True)
    retry_failed = (args.retry_failed) if args.retry_failed is not None else inference_cfg.get("retry_failed", True)

    # Apply CLI overrides to shared settings
    if args.input_file is not None:
        shared["input_file"] = args.input_file
    if args.max_workers is not None:
        shared["max_workers"] = args.max_workers

    # Load provider module and extract config
    if args.provider == "claude":
        from scripts.inference._claude import get_config, make_process_fn
    else:
        from scripts.inference._openai import get_config, make_process_fn

    provider_cfg = get_config(cfg)

    # Apply model override
    if args.model is not None:
        provider_cfg["model"] = args.model

    # If resuming an existing output file, restore all config from its metadata.
    # This ensures the same model, workers, and settings are used across a resumed run.
    if args.output_file and os.path.exists(args.output_file):
        _, existing_meta = _load_existing(args.output_file)
        if existing_meta:
            snap = existing_meta.get("config_snapshot") or {}
            for key, value in snap.items():
                if key in shared:
                    shared[key] = value
                elif key in provider_cfg:
                    provider_cfg[key] = value
                elif key == "shuffle":
                    shuffle = value
                elif key == "retry_failed":
                    retry_failed = value

    process_fn = make_process_fn(provider_cfg, shared)

    asyncio.run(main(
        provider=args.provider,
        provider_cfg=provider_cfg,
        shared=shared,
        process_fn=process_fn,
        limit=args.limit,
        retry_failed=retry_failed,
        shuffle=shuffle,
        output_file=args.output_file,
    ))
