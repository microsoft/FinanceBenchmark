"""
test_pipeline_e2e.py

Smoke test: run_benchmark.main() with --limit 10 runs the full pipeline
(data prep skipped, inference + eval mocked) and produces the expected
output files + DB entries.

No live API calls are made.  subprocess calls to inference.py and
evaluate.py are intercepted by fake_run_step, which writes plausible
output files and registers them in a temp DB.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import make_fake_answers, make_fake_eval, make_test_config


class TestPipelineE2E:
    def _setup(self, tmp_path):
        """Common setup: temp results dir, DB, config, config_hash."""
        from scripts.analysis.runs_db import init_db
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"

        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        conn = init_db(db_path)
        return results_dir, db_path, cfg, config_hash, model, conn

    def _make_fake_run_step(self, results_dir, db_path, model, config_hash):
        """Return a fake run_step that writes output files and updates the DB."""
        from scripts.analysis.runs_db import init_db, register_inference_only, register_run
        from coolname import generate_slug

        inference_calls = []
        eval_calls = []

        def fake_run_step(cmd, label):
            cmd_str = " ".join(str(c) for c in cmd)
            if "inference.py" in cmd_str:
                inference_calls.append(list(cmd))
                slug = generate_slug(2)
                answers_path = results_dir / f"answers_{model}_{slug}.json"
                inf_meta = make_fake_answers(answers_path, model, config_hash)
                conn = init_db(db_path)
                register_inference_only(conn, str(answers_path), inf_meta)
                conn.close()

            elif "evaluate.py" in cmd_str:
                eval_calls.append(list(cmd))
                idx = list(cmd).index("--answers") + 1
                answers_path = Path(cmd[idx])
                with open(answers_path) as f:
                    inf_data = json.load(f)
                slug = generate_slug(2)
                eval_path = results_dir / f"eval_results_{answers_path.stem}_{slug}.json"
                make_fake_eval(eval_path, answers_path, inf_data["metadata"])
                conn = init_db(db_path)
                register_run(conn, str(answers_path), str(eval_path))
                conn.close()

        return fake_run_step, inference_calls, eval_calls

    def test_pipeline_creates_answers_and_eval_files(self, tmp_path):
        """Full pipeline creates one answers file and one eval file."""
        results_dir, db_path, cfg, config_hash, model, db_conn = self._setup(tmp_path)
        fake_run_step, inf_calls, eval_calls = self._make_fake_run_step(
            results_dir, db_path, model, config_hash
        )

        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=db_conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        db_conn.close()

        answers_files = list(results_dir.glob("answers_*.json"))
        eval_files = list(results_dir.glob("eval_results_*.json"))

        assert len(answers_files) == 1, f"Expected 1 answers file, got {answers_files}"
        assert len(eval_files) == 1, f"Expected 1 eval file, got {eval_files}"
        assert len(inf_calls) == 1
        assert len(eval_calls) == 1

    def test_answers_file_has_ten_results(self, tmp_path):
        """The answers file contains 10 result entries."""
        results_dir, db_path, cfg, config_hash, model, db_conn = self._setup(tmp_path)
        fake_run_step, _, _ = self._make_fake_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=db_conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        db_conn.close()

        answers_file = next(results_dir.glob("answers_*.json"))
        with open(answers_file) as f:
            data = json.load(f)
        assert len(data["results"]) == 10

    def test_limit_flag_forwarded_to_inference(self, tmp_path):
        """-n 10 is forwarded to the inference subprocess command."""
        results_dir, db_path, cfg, config_hash, model, db_conn = self._setup(tmp_path)
        fake_run_step, inf_calls, _ = self._make_fake_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=db_conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        db_conn.close()

        assert len(inf_calls) == 1
        cmd = inf_calls[0]
        assert "-n" in cmd
        assert "10" in cmd

    def test_db_has_complete_run_row(self, tmp_path):
        """After pipeline, runs.db has a row with both inference_file and eval_file set."""
        results_dir, db_path, cfg, config_hash, model, db_conn = self._setup(tmp_path)
        fake_run_step, _, _ = self._make_fake_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=db_conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        db_conn.close()

        from scripts.analysis.runs_db import init_db

        conn = init_db(db_path)
        rows = conn.execute(
            "SELECT inference_file, eval_file FROM runs WHERE eval_file IS NOT NULL"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["inference_file"] is not None
        assert rows[0]["eval_file"] is not None

    def test_eval_step_uses_inference_file_from_db(self, tmp_path):
        """Eval receives the --answers file that was registered in the DB by inference."""
        results_dir, db_path, cfg, config_hash, model, db_conn = self._setup(tmp_path)
        fake_run_step, _, eval_calls = self._make_fake_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=db_conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        db_conn.close()

        assert len(eval_calls) == 1
        cmd = eval_calls[0]
        assert "--answers" in cmd
        # The answers file passed to eval must exist on disk
        idx = cmd.index("--answers") + 1
        assert Path(cmd[idx]).exists()
