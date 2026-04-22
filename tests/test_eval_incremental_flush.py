"""
test_eval_incremental_flush.py

Tests for incremental result flushing in evaluate.py:
  1. The output file is written after each evaluation completes (N+1 writes for N questions).
  2. Resume works: when the process is restarted with the same --output file, only
     the unevaluated questions are processed.
  3. The final output file has the correct top-level structure and content.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_QUESTIONS = 5


def _make_answers_file(tmp_path: Path, n: int = N_QUESTIONS) -> Path:
    answers_dir = tmp_path / "results"
    answers_dir.mkdir(exist_ok=True)
    answers_file = answers_dir / "answers_testmodel_slug.json"
    data = {
        "metadata": {
            "model": "testmodel",
            "provider": "openai",
            "inf_slug": "slug",
            "run_id": "slug",
            "run_timestamp": "2026-01-01T00:00:00Z",
            "config_hash": "abc123",
            "config_snapshot": {},
        },
        "results": [
            {
                "question": f"Q{i}",
                "answer": f"A{i}",
                "plugin": "erp_qa",
                "tool_call_count": 0,
                "successful_tool_calls": 0,
                "tool_calls": [],
                "inference_time_secs": 0.1,
            }
            for i in range(n)
        ],
    }
    answers_file.write_text(json.dumps(data), encoding="utf-8")
    return answers_file


def _make_eval_yaml(tmp_path: Path, n: int = N_QUESTIONS) -> Path:
    eval_file = tmp_path / "dataset.yaml"
    entries = [
        {
            "query": f"Q{i}",
            "plugin": "erp_qa",
            "segment": "test",
            "tags": [{"tag": "accuracy", "assertions": [{"text": "assert1"}]}],
        }
        for i in range(n)
    ]
    eval_file.write_text(yaml.dump(entries), encoding="utf-8")
    return eval_file


def _fake_evaluate_entry(question, answer, plugin, tags, answer_entry, judge, groundedness_judge, **kwargs):
    """Fake evaluate_entry: returns a scored result without calling any judge."""
    return {
        "question": question,
        "plugin": plugin,
        "actual_answer": answer,
        "overall_score": 0.8,
        "tag_scores": {"accuracy": 0.8},
        "assertion_scores": {"accuracy": {"assert1": 1}},
        "tag_reasoning": {"accuracy": "ok"},
        "skipped": False,
    }


def _run_evaluate_main(argv: list, evaluate_module) -> None:
    """Run evaluate.main() with patched sys.argv and mocked judge/DB."""
    mock_judge = MagicMock()

    with (
        patch.object(sys, "argv", argv),
        patch(f"{evaluate_module.__name__}.dspy") as mock_dspy,
        patch(f"{evaluate_module.__name__}.evaluate_entry", side_effect=_fake_evaluate_entry),
        patch("scripts.analysis.runs_db.init_db"),
        patch("scripts.analysis.runs_db.register_run"),
    ):
        mock_dspy.LM.return_value = MagicMock()
        mock_dspy.ChainOfThought.return_value = mock_judge
        mock_dspy.settings = MagicMock()
        evaluate_module.main()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluate_module():
    """Import evaluate.py, mocking Azure auth to avoid credential failures."""
    mock_credential = MagicMock()
    mock_token_provider = MagicMock()
    with (
        patch("azure.identity.DefaultAzureCredential", return_value=mock_credential),
        patch("azure.identity.get_bearer_token_provider", return_value=mock_token_provider),
    ):
        import importlib
        import scripts.evaluation.evaluate as ev
        importlib.reload(ev)  # re-run module-level code under the patches
        yield ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncrementalFlush:

    def test_output_file_written_after_each_question(self, tmp_path, evaluate_module):
        """Output file is written N+1 times for N questions (once per flush + final write)."""
        answers_file = _make_answers_file(tmp_path)
        eval_file = _make_eval_yaml(tmp_path)
        output_file = tmp_path / "results" / "eval_out.json"

        write_calls = []
        original_open = open

        def tracking_open(file, mode="r", **kwargs):
            fh = original_open(file, mode, **kwargs)
            if str(file) == str(output_file) and "w" in mode:
                write_calls.append(str(file))
            return fh

        argv = [
            "evaluate.py",
            "--answers", str(answers_file),
            "--eval-file", str(eval_file),
            "--output", str(output_file),
            "--threads", "1",
        ]
        with patch("builtins.open", side_effect=tracking_open):
            _run_evaluate_main(argv, evaluate_module)

        # N incremental flushes + 1 final write
        assert len(write_calls) == N_QUESTIONS + 1, (
            f"Expected {N_QUESTIONS + 1} writes, got {len(write_calls)}"
        )

    def test_resume_skips_already_evaluated_questions(self, tmp_path, evaluate_module):
        """When output file already has some results, only missing questions are evaluated."""
        n_already_done = 3
        answers_file = _make_answers_file(tmp_path)
        eval_file = _make_eval_yaml(tmp_path)
        output_file = tmp_path / "results" / "eval_out.json"

        # Pre-populate output file as if a previous run completed 3 of 5 questions
        partial_results = [
            {
                "question": f"Q{i}",
                "plugin": "erp_qa",
                "actual_answer": f"A{i}",
                "overall_score": 0.7,
                "tag_scores": {"accuracy": 0.7},
                "assertion_scores": {},
                "tag_reasoning": {},
                "skipped": False,
            }
            for i in range(n_already_done)
        ]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps({"metadata": {}, "summary": {}, "results": partial_results}),
            encoding="utf-8",
        )

        evaluated_questions = []
        original_fake = _fake_evaluate_entry

        def tracking_evaluate_entry(question, *args, **kwargs):
            evaluated_questions.append(question)
            return original_fake(question, *args, **kwargs)

        argv = [
            "evaluate.py",
            "--answers", str(answers_file),
            "--eval-file", str(eval_file),
            "--output", str(output_file),
            "--threads", "1",
        ]
        mock_judge = MagicMock()
        with (
            patch.object(sys, "argv", argv),
            patch(f"{evaluate_module.__name__}.dspy") as mock_dspy,
            patch(f"{evaluate_module.__name__}.evaluate_entry", side_effect=tracking_evaluate_entry),
            patch("scripts.analysis.runs_db.init_db"),
            patch("scripts.analysis.runs_db.register_run"),
        ):
            mock_dspy.LM.return_value = MagicMock()
            mock_dspy.ChainOfThought.return_value = mock_judge
            mock_dspy.settings = MagicMock()
            evaluate_module.main()

        remaining = N_QUESTIONS - n_already_done
        assert len(evaluated_questions) == remaining, (
            f"Expected {remaining} evaluate_entry calls (resume), got {len(evaluated_questions)}: {evaluated_questions}"
        )
        # Verify the resumed questions are the ones not already in the file
        already_done = {f"Q{i}" for i in range(n_already_done)}
        for q in evaluated_questions:
            assert q not in already_done, f"Question {q} was already done but got re-evaluated"

    def test_final_output_has_correct_structure(self, tmp_path, evaluate_module):
        """Final output file has metadata, summary, and results with expected content."""
        answers_file = _make_answers_file(tmp_path)
        eval_file = _make_eval_yaml(tmp_path)
        output_file = tmp_path / "results" / "eval_out.json"

        argv = [
            "evaluate.py",
            "--answers", str(answers_file),
            "--eval-file", str(eval_file),
            "--output", str(output_file),
            "--threads", "1",
        ]
        _run_evaluate_main(argv, evaluate_module)

        assert output_file.exists(), "Output file was not created"
        data = json.loads(output_file.read_text(encoding="utf-8"))

        # Top-level keys
        assert set(data.keys()) >= {"metadata", "summary", "results"}

        # Metadata fields
        meta = data["metadata"]
        assert "judge_deployment" in meta
        assert "eval_run_id" in meta
        assert "config_hash" in meta
        assert "answers_file" in meta

        # Summary fields
        summary = data["summary"]
        assert summary["total"] == N_QUESTIONS
        assert summary["evaluated"] == N_QUESTIONS
        assert summary["skipped"] == 0
        assert summary["overall_score"] == pytest.approx(0.8)

        # Results: one entry per question, in order
        results = data["results"]
        assert len(results) == N_QUESTIONS
        for i, r in enumerate(results):
            assert r["question"] == f"Q{i}"
            assert not r["skipped"]
            assert r["overall_score"] == pytest.approx(0.8)
