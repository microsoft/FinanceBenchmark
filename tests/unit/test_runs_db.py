import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analysis.runs_db import (
    init_db,
    generate_unique_slug,
    register_inference_only,
    find_reusable_inference,
    register_run,
)


def test_init_db_creates_runs_table(tmp_db):
    cursor = tmp_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    assert cursor.fetchone() is not None


def test_init_db_runs_columns(tmp_db):
    cursor = tmp_db.execute("PRAGMA table_info(runs)")
    columns = {row["name"] for row in cursor.fetchall()}
    expected = {
        "id", "inf_run_id", "eval_run_id", "timestamp", "provider", "model",
        "judge", "inference_config_hash", "eval_config_hash",
        "inference_file", "eval_file",
        "shuffle", "retry_failed", "max_workers", "max_turns",
        "max_tool_calls", "timeout",
    }
    assert expected.issubset(columns)


def test_generate_unique_slug(tmp_path):
    slug = generate_unique_slug(tmp_path, "answers_test_")
    assert isinstance(slug, str)
    assert len(slug) > 0


def test_register_inference_only_returns_id(tmp_db):
    metadata = {
        "provider": "claude",
        "model": "claude-sonnet-4-6",
        "run_timestamp": "2026-04-22T00:00:00Z",
        "config_hash": "abc123",
        "config_snapshot": {"shuffle": True, "max_turns": 5},
    }
    row_id = register_inference_only(tmp_db, "results/answers_test.json", metadata)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_find_reusable_inference_returns_none_when_absent(tmp_db):
    result = find_reusable_inference(tmp_db, "nonexistent_hash")
    assert result is None


def test_find_reusable_inference_returns_file_when_present(tmp_db):
    metadata = {
        "provider": "claude",
        "model": "claude-sonnet-4-6",
        "run_timestamp": "2026-04-22T00:00:00Z",
        "config_hash": "testhash999",
    }
    register_inference_only(tmp_db, "results/answers_hash_match.json", metadata)
    result = find_reusable_inference(tmp_db, "testhash999")
    assert result is not None
    assert "answers_hash_match.json" in result


def test_register_inference_only_is_idempotent(tmp_db):
    metadata = {
        "provider": "openai",
        "model": "gpt-5.2",
        "run_timestamp": "2026-04-22T00:00:00Z",
        "config_hash": "idempotent_hash",
    }
    id1 = register_inference_only(tmp_db, "results/answers_idem.json", metadata)
    id2 = register_inference_only(tmp_db, "results/answers_idem.json", metadata)
    assert id1 == id2  # same row updated, not a new insert

    count = tmp_db.execute(
        "SELECT COUNT(*) FROM runs WHERE inference_file LIKE '%answers_idem.json' AND eval_file IS NULL"
    ).fetchone()[0]
    assert count == 1
