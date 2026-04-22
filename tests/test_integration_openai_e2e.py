"""
test_integration_openai_e2e.py

End-to-end integration test: runs the full benchmark pipeline for the OpenAI
provider with --limit 10 against live services (OpenAI API + ERP MCP + Azure
judge).

Run explicitly with:
    uv run pytest tests/test_integration_openai_e2e.py -v -m integration

Skipped automatically when required credentials are absent:
    ERP_MCP_TOKEN        — ERP MCP bearer token
    AZURE_OPENAI_ENDPOINT — Azure OpenAI endpoint for the eval judge
"""
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

pytestmark = pytest.mark.integration


def _credentials_available() -> bool:
    return bool(os.getenv("ERP_MCP_TOKEN") and os.getenv("AZURE_OPENAI_ENDPOINT"))


@pytest.fixture(scope="module", autouse=True)
def require_credentials():
    if not _credentials_available():
        pytest.skip(
            "Integration test requires ERP_MCP_TOKEN and AZURE_OPENAI_ENDPOINT env vars. "
            "Run `uv run refresh_erp_token.py` and ensure Azure credentials are set."
        )


@pytest.fixture(scope="module")
def pipeline_run():
    """Run the full pipeline once and return (CompletedProcess, answers_file, eval_file)."""
    # Snapshot files that exist before we run so we can identify new ones
    before_answers = set(RESULTS_DIR.glob("answers_gpt-*.json"))
    before_evals = set(RESULTS_DIR.glob("eval_results_*.json"))
    before_db_rowid = _latest_openai_rowid()

    uv_bin = shutil.which("uv") or "uv"
    result = subprocess.run(
        [
            uv_bin, "run",
            "run_benchmark.py",
            "--provider", "openai",
            "-n", "10",
            "--no-shuffle",
            "--skip-prep",
            "--no-reuse",
        ],
        cwd=REPO_ROOT,
        capture_output=False,  # stream stdout/stderr so progress is visible
        timeout=600,           # 10 min ceiling for 10 questions + eval
    )

    # Identify files created by this run
    after_answers = set(RESULTS_DIR.glob("answers_gpt-*.json"))
    after_evals = set(RESULTS_DIR.glob("eval_results_*.json"))

    new_answers = after_answers - before_answers
    new_evals = after_evals - before_evals

    answers_file = next(iter(new_answers)) if new_answers else None
    eval_file = next(iter(new_evals)) if new_evals else None

    return result, answers_file, eval_file, before_db_rowid


def _latest_openai_rowid() -> int | None:
    db_path = RESULTS_DIR / "runs.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    # Use positional access — sqlite3 without row_factory returns plain tuples
    row = conn.execute(
        "SELECT rowid FROM runs WHERE provider = 'openai' ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None


class TestIntegrationOpenAIE2E:
    def test_pipeline_exits_cleanly(self, pipeline_run):
        result, *_ = pipeline_run
        assert result.returncode == 0, (
            f"run_benchmark.py exited with code {result.returncode}"
        )

    def test_answers_file_created(self, pipeline_run):
        """A new answers_gpt-*.json file was written to results/ by this run."""
        _, answers_file, _, _ = pipeline_run
        assert answers_file is not None, "No new answers file created by this run"
        assert answers_file.exists()

    def test_answers_file_has_ten_results(self, pipeline_run):
        """The answers file has exactly 10 result entries."""
        _, answers_file, _, _ = pipeline_run
        assert answers_file is not None, "No answers file to check"
        with open(answers_file, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    def test_answers_file_has_no_errors(self, pipeline_run):
        """All 10 answers succeeded (no error fields)."""
        _, answers_file, _, _ = pipeline_run
        assert answers_file is not None, "No answers file to check"
        with open(answers_file, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        errors = [r for r in results if r.get("error")]
        assert not errors, f"{len(errors)} inference error(s): {[e['error'] for e in errors]}"

    def test_eval_file_created(self, pipeline_run):
        """A new eval_results_*.json file was written to results/ by this run."""
        _, _, eval_file, _ = pipeline_run
        assert eval_file is not None, "No new eval file created by this run"
        assert eval_file.exists()

    def test_eval_file_has_scores(self, pipeline_run):
        """The eval file has a non-null overall_score."""
        _, _, eval_file, _ = pipeline_run
        assert eval_file is not None, "No eval file to check"
        with open(eval_file, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        assert summary.get("overall_score") is not None, "overall_score is None"
        assert summary.get("evaluated", 0) > 0, "No questions were evaluated"

    def test_run_registered_in_db(self, pipeline_run):
        """runs.db has a new row for this run with both inference_file and eval_file set."""
        _, _, _, before_rowid = pipeline_run
        db_path = RESULTS_DIR / "runs.db"
        assert db_path.exists(), "runs.db not found"

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT rowid AS rid, inference_file, eval_file FROM runs "
            "WHERE provider = 'openai' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        conn.close()

        assert row is not None, "No openai run found in runs.db"
        assert row["rid"] != before_rowid, "No new DB row was created by this run"
        assert row["inference_file"] is not None
        assert row["eval_file"] is not None

    def test_eval_uses_matching_inference_file(self, pipeline_run):
        """The eval file's answers_file metadata points to the inference file in the DB."""
        _, answers_file, eval_file, _ = pipeline_run
        assert eval_file is not None, "No eval file to check"

        with open(eval_file, encoding="utf-8") as f:
            eval_data = json.load(f)

        answers_in_eval = eval_data.get("metadata", {}).get("answers_file", "")
        assert Path(answers_in_eval).resolve() == answers_file.resolve(), (
            f"Eval metadata.answers_file ({answers_in_eval}) does not match "
            f"the answers file created by this run ({answers_file})"
        )
