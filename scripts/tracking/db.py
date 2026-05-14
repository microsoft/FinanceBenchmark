"""
db.py  (scripts/tracking/db.py)

SQLite helper for run tracking.  DB lives at results/runs.db.

Schema — single table, one row per inference run (eval_file is NULL until
evaluation completes):

    runs(
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        inf_run_id            TEXT,   -- inference run identifier; NULL until eval completes
        eval_run_id           TEXT,   -- eval run identifier; NULL until eval completes
        timestamp             TEXT,
        provider              TEXT,
        model                 TEXT,
        judge                 TEXT,
        inference_config_hash TEXT,
        eval_config_hash      TEXT,
        inference_file        TEXT,   -- relative path from repo root
        eval_file             TEXT,   -- relative path from repo root; NULL until eval completes
        shuffle               INTEGER,
        retry_failed          INTEGER,
        max_workers           INTEGER,
        max_turns             INTEGER, -- claude-specific, NULL for openai
        max_tool_calls        INTEGER, -- openai-specific, NULL for claude
        timeout               INTEGER  -- openai-specific, NULL for claude
    )

Key functions:
    init_db(db_path)                    -> sqlite3.Connection
    generate_unique_slug(output_dir, prefix)
                                        -> str
    register_inference_only(conn, inference_file, metadata)
                                        -> int  (row id)
    register_run(conn, inference_file, eval_file, inf_slug, eval_slug)
                                        -> int  (row id)
    find_reusable_inference(conn, inference_config_hash)
                                        -> str | None
    find_reusable_eval(conn, inference_config_hash)
                                        -> tuple[str, str] | None
    find_latest_inference(conn, inference_config_hash)
                                        -> tuple[str, str] | None
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from coolname import generate_slug

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "results" / "runs.db"

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    inf_run_id            TEXT,
    eval_run_id           TEXT,
    timestamp             TEXT,
    provider              TEXT,
    model                 TEXT,
    judge                 TEXT,
    inference_config_hash TEXT,
    eval_config_hash      TEXT,
    inference_file        TEXT,
    eval_file             TEXT,
    shuffle               INTEGER,
    retry_failed          INTEGER,
    max_workers           INTEGER,
    max_turns             INTEGER,
    max_tool_calls        INTEGER,
    timeout               INTEGER
);
"""


def init_db(db_path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    """Open (or create) the runs SQLite database and return the connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL_RUNS)
    conn.commit()

    # Drop the now-redundant inference_runs table if it was created by a
    # previous version of this module
    conn.execute("DROP TABLE IF EXISTS inference_runs")
    conn.commit()

    return conn


def generate_unique_slug(output_dir: Path, prefix: str, word_index: int = 0) -> str:
    """Generate a unique single-word coolname slug for use in output filenames.

    coolname requires a minimum of 2 words, so a 2-word slug is generated
    internally and one word is extracted via ``word_index``.  Use ``word_index=0``
    (the default) for inference slugs and ``word_index=1`` for eval slugs.

    Checks that no file matching ``{output_dir}/{prefix}*{word}*.json`` already
    exists.  Retries up to 10 times on collision, then raises RuntimeError.

    Args:
        output_dir:  Directory where output files are written.
        prefix:      Filename prefix to check against (e.g. ``"answers_gpt-5.2_"``).
        word_index:  Which word of the generated 2-word slug to use (0 or 1).

    Returns:
        A single lowercase slug word that is not already in use.
    """
    output_dir = Path(output_dir)
    for _ in range(10):
        word = generate_slug(2).split("-")[word_index]
        # Check that no existing file contains this word
        matches = list(output_dir.glob(f"{prefix}*{word}*.json"))
        if not matches:
            return word
    raise RuntimeError(
        f"Could not generate a unique slug for prefix '{prefix}' in {output_dir} after 10 attempts"
    )


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_relative(path: str) -> str:
    """Return *path* relative to repo root when possible, else as-is."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError:
        return path


def _extract_inference_metadata(data: dict) -> dict:
    """Pull the inference metadata block out of an inference envelope."""
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"] or {}
    return {}


def _extract_eval_metadata(data: dict) -> dict:
    """Pull the top-level metadata block out of an eval result envelope."""
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"] or {}
    return {}


def _extract_slug_from_filename(filename: str, prefix: str) -> Optional[str]:
    """Try to extract a slug from a filename like ``{prefix}{model}_{slug}.json``.

    Splits stem on ``_`` and takes the last segment.  Returns None when the
    last segment looks like a version number (e.g. ``5.2``) rather than a slug.
    """
    stem = Path(filename).stem  # e.g. "answers_gpt-5.2_panda"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]  # e.g. "gpt-5.2_panda"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        candidate = parts[1]
        # Reject if it looks like a version number or contains only digits/dots
        if candidate and not all(c in "0123456789." for c in candidate):
            return candidate
    return None


def register_inference_only(
    conn: sqlite3.Connection,
    inference_file: str,
    metadata: dict,
) -> int:
    """Register a completed inference run in the runs table (eval_file = NULL).

    Idempotent: if a row already exists for this inference_file with a NULL
    eval_file it is updated in-place; otherwise a new row is inserted.

    Args:
        conn:           Open DB connection.
        inference_file: Path to the inference output JSON file.
        metadata:       The metadata dict from the inference envelope.

    Returns:
        The row id.
    """
    inf_rel = _to_relative(inference_file)

    inf_run_id = metadata.get("inf_run_id") or metadata.get("inf_slug") or metadata.get("run_id")
    timestamp = metadata.get("run_timestamp", "")
    provider  = metadata.get("provider")
    model     = metadata.get("model")
    inf_config_hash = metadata.get("config_hash")

    snapshot = metadata.get("config_snapshot") or {}
    shuffle        = snapshot.get("shuffle")
    retry_failed   = snapshot.get("retry_failed")
    max_workers    = snapshot.get("max_workers")
    max_turns      = snapshot.get("max_turns")
    max_tool_calls = snapshot.get("max_tool_calls")
    timeout        = snapshot.get("timeout")

    existing = conn.execute(
        "SELECT id FROM runs WHERE inference_file = ? AND eval_file IS NULL",
        (inf_rel,),
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE runs SET
                inf_run_id            = ?,
                timestamp             = ?,
                provider              = ?,
                model                 = ?,
                inference_config_hash = ?,
                shuffle               = ?,
                retry_failed          = ?,
                max_workers           = ?,
                max_turns             = ?,
                max_tool_calls        = ?,
                timeout               = ?
            WHERE id = ?
            """,
            (
                inf_run_id, timestamp, provider, model, inf_config_hash,
                int(shuffle) if shuffle is not None else None,
                int(retry_failed) if retry_failed is not None else None,
                max_workers, max_turns, max_tool_calls, timeout,
                existing["id"],
            ),
        )
        conn.commit()
        return existing["id"]
    else:
        cur = conn.execute(
            """
            INSERT INTO runs (
                inf_run_id,
                timestamp, provider, model,
                inference_config_hash,
                inference_file,
                shuffle, retry_failed, max_workers,
                max_turns, max_tool_calls, timeout
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inf_run_id,
                timestamp, provider, model,
                inf_config_hash,
                inf_rel,
                int(shuffle) if shuffle is not None else None,
                int(retry_failed) if retry_failed is not None else None,
                max_workers, max_turns, max_tool_calls, timeout,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def register_run(
    conn: sqlite3.Connection,
    inference_file: str,
    eval_file: str,
    inf_run_id: Optional[str] = None,
    eval_run_id: Optional[str] = None,
) -> int:
    """Register a completed inference+eval pair in the runs table.

    Upsert strategy: look for an existing row matching inference_file
    (regardless of eval_file).  If found, UPDATE it with eval fields.
    If not found, INSERT a new complete row.

    Args:
        conn:           Open DB connection.
        inference_file: Path to the inference output JSON file.
        eval_file:      Path to the eval results JSON file.
        inf_run_id:     Optional inference run ID (auto-extracted from filename if None).
        eval_run_id:    Optional eval run ID (auto-extracted from filename if None).

    Returns:
        The row id.
    """
    inf_rel = _to_relative(inference_file)
    eval_rel = _to_relative(eval_file)

    inf_data = _load_json(inference_file)
    eval_data = _load_json(eval_file)

    inf_meta = _extract_inference_metadata(inf_data)
    eval_meta = _extract_eval_metadata(eval_data)

    # Prefer inference_metadata embedded in eval file (always has it when
    # produced by evaluate.py), fall back to reading the inference file direct.
    embedded_inf_meta = eval_meta.get("inference_metadata") or inf_meta

    provider  = embedded_inf_meta.get("provider")
    model     = embedded_inf_meta.get("model")
    timestamp = eval_meta.get("run_timestamp") # or embedded_inf_meta.get("run_timestamp")

    inf_config_hash  = embedded_inf_meta.get("config_hash")
    eval_config_hash = eval_meta.get("config_hash")

    judge = eval_meta.get("judge_deployment")

    # Flattened inference config params
    snapshot = embedded_inf_meta.get("config_snapshot") or {}
    shuffle        = snapshot.get("shuffle")
    retry_failed   = snapshot.get("retry_failed")
    max_workers    = snapshot.get("max_workers")
    max_turns      = snapshot.get("max_turns")       # claude-specific
    max_tool_calls = snapshot.get("max_tool_calls")  # openai-specific
    timeout        = snapshot.get("timeout")         # openai-specific

    # Resolve run IDs: use provided values, fall back to extraction from filenames
    if inf_run_id is None:
        inf_run_id = (
            embedded_inf_meta.get("inf_run_id")
            or embedded_inf_meta.get("inf_slug")
            or embedded_inf_meta.get("run_id")
            or _extract_slug_from_filename(inference_file, "answers_")
        )
    if eval_run_id is None:
        eval_run_id = (
            eval_meta.get("eval_run_id")
            or eval_meta.get("eval_slug")
            or _extract_slug_from_filename(eval_file, "eval_results_")
        )

    # Upsert by (inference_file, eval_file) so multiple evals against the same
    # inference file each get their own row rather than overwriting each other.
    existing = conn.execute(
        "SELECT id FROM runs WHERE inference_file = ? AND eval_file = ?",
        (inf_rel, eval_rel),
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE runs SET
                inf_run_id            = ?,
                eval_run_id           = ?,
                timestamp             = ?,
                provider              = ?,
                model                 = ?,
                judge                 = ?,
                inference_config_hash = ?,
                eval_config_hash      = ?,
                eval_file             = ?,
                shuffle               = ?,
                retry_failed          = ?,
                max_workers           = ?,
                max_turns             = ?,
                max_tool_calls        = ?,
                timeout               = ?
            WHERE id = ?
            """,
            (
                inf_run_id, eval_run_id, timestamp, provider, model, judge,
                inf_config_hash, eval_config_hash, eval_rel,
                int(shuffle) if shuffle is not None else None,
                int(retry_failed) if retry_failed is not None else None,
                max_workers, max_turns, max_tool_calls, timeout,
                existing["id"],
            ),
        )
        conn.commit()
        return existing["id"]
    else:
        cur = conn.execute(
            """
            INSERT INTO runs (
                inf_run_id, eval_run_id,
                timestamp, provider, model, judge,
                inference_config_hash, eval_config_hash,
                inference_file, eval_file,
                shuffle, retry_failed, max_workers,
                max_turns, max_tool_calls, timeout
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inf_run_id, eval_run_id,
                timestamp, provider, model, judge,
                inf_config_hash, eval_config_hash,
                inf_rel, eval_rel,
                int(shuffle) if shuffle is not None else None,
                int(retry_failed) if retry_failed is not None else None,
                max_workers, max_turns, max_tool_calls, timeout,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def find_reusable_inference(
    conn: sqlite3.Connection,
    inference_config_hash: str,
) -> Optional[str]:
    """Return the inference_file path for an existing run with a matching hash.

    Queries the runs table ordered by timestamp DESC.

    Returns the relative path (from repo root) or None if no match is found.
    """
    row = conn.execute(
        "SELECT inference_file FROM runs WHERE inference_config_hash = ? ORDER BY timestamp DESC LIMIT 1",
        (inference_config_hash,),
    ).fetchone()
    return row["inference_file"] if row else None


def find_reusable_eval(
    conn: sqlite3.Connection,
    inference_config_hash: str,
    eval_config_hash: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Return (eval_file, inference_file) for the most recent run with a completed eval.

    Only returns a result when eval_file IS NOT NULL and both config hashes match.
    If eval_config_hash is None, only inference_config_hash is checked.
    Returns None if no such run exists.
    """
    if eval_config_hash is not None:
        row = conn.execute(
            "SELECT eval_file, inference_file FROM runs "
            "WHERE inference_config_hash = ? AND eval_config_hash = ? AND eval_file IS NOT NULL "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (inference_config_hash, eval_config_hash),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT eval_file, inference_file FROM runs "
            "WHERE inference_config_hash = ? AND eval_file IS NOT NULL "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (inference_config_hash,),
        ).fetchone()
    if row:
        return row["eval_file"], row["inference_file"]
    return None


def find_latest_inference(
    conn: sqlite3.Connection,
    inference_config_hash: str,
) -> Optional[tuple[str, str]]:
    """Return the (inference_file, inf_slug) for the most recent run matching the hash.

    Queries the runs table ordered by timestamp DESC.

    Returns a tuple of (inference_file, inf_slug) or None if no match is found.
    """
    row = conn.execute(
        "SELECT inference_file, inf_run_id FROM runs "
        "WHERE inference_config_hash = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (inference_config_hash,),
    ).fetchone()
    if row:
        return row["inference_file"], row["inf_run_id"]
    return None
