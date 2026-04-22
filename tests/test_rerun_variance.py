"""
test_rerun_variance.py

Tests for variance-analysis re-run behavior:
  1. compute_inference_config_hash changes when relevant config keys change.
  2. --no-reuse always runs fresh inference (bypasses DB hash check).
  3. Two consecutive runs with --no-reuse produce two distinct inference files
     and two distinct eval files.
  4. Each run's eval step uses that run's own inference file.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import make_fake_answers, make_fake_eval, make_test_config, load_real_config


# ---------------------------------------------------------------------------
# Unit tests: config hash behaviour
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_hash_changes_when_system_instructions_changes(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["shared"]["system_instructions"] += " CHANGED"
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 != h2

    def test_hash_changes_when_model_changes(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["openai"]["model"] = "different-model"
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 != h2

    def test_hash_stable_for_same_config(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        assert (
            compute_inference_config_hash(cfg, "openai")
            == compute_inference_config_hash(cfg, "openai")
        )

    def test_hash_differs_by_provider(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        assert (
            compute_inference_config_hash(cfg, "openai")
            != compute_inference_config_hash(cfg, "claude")
        )

    def test_eval_config_change_does_not_affect_inference_hash(self):
        """judge_deployment is not part of the inference config hash."""
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["evaluation"]["judge_deployment"] = "some-other-judge"
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 == h2


# ---------------------------------------------------------------------------
# Unit tests: DB reuse helpers
# ---------------------------------------------------------------------------

class TestDBReuse:
    def test_find_reusable_inference_empty_db(self, tmp_path):
        from scripts.analysis.runs_db import init_db, find_reusable_inference

        conn = init_db(tmp_path / "runs.db")
        assert find_reusable_inference(conn, "abc123") is None
        conn.close()

    def test_find_reusable_inference_finds_registered_file(self, tmp_path):
        from scripts.analysis.runs_db import init_db, find_reusable_inference, register_inference_only

        db_path = tmp_path / "runs.db"
        conn = init_db(db_path)

        metadata = {
            "model": "gpt-5.2",
            "provider": "openai",
            "inf_slug": "panda",
            "run_id": "panda",
            "run_timestamp": "2026-01-01T00:00:00Z",
            "config_hash": "abc123def456",
            "config_snapshot": {},
        }
        register_inference_only(conn, "results/answers_gpt-5.2_panda.json", metadata)

        result = find_reusable_inference(conn, "abc123def456")
        conn.close()

        assert Path(result) == Path("results/answers_gpt-5.2_panda.json")

    def test_find_latest_inference_returns_newest(self, tmp_path):
        """find_latest_inference returns most recent entry (by timestamp) for the hash."""
        from scripts.analysis.runs_db import init_db, find_latest_inference, register_inference_only

        db_path = tmp_path / "runs.db"
        conn = init_db(db_path)

        for slug, ts in [("panda", "2026-01-01T00:00:00Z"), ("falcon", "2026-01-02T00:00:00Z")]:
            register_inference_only(
                conn,
                f"results/answers_gpt-5.2_{slug}.json",
                {
                    "model": "gpt-5.2",
                    "provider": "openai",
                    "inf_slug": slug,
                    "run_id": slug,
                    "run_timestamp": ts,
                    "config_hash": "myhash123456",
                    "config_snapshot": {},
                },
            )

        result = find_latest_inference(conn, "myhash123456")
        conn.close()

        assert result is not None
        assert "falcon" in result[0]


# ---------------------------------------------------------------------------
# Integration tests: --no-reuse produces new results
# ---------------------------------------------------------------------------

class TestNoReuseVariance:
    def _make_run_step(self, results_dir, db_path, model, config_hash):
        """Return a fake run_step that writes output files and registers in DB."""
        from scripts.analysis.runs_db import init_db, register_inference_only, register_run
        from coolname import generate_slug

        inf_calls = []
        eval_calls = []

        def fake_run_step(cmd, label):
            cmd_str = " ".join(str(c) for c in cmd)
            if "inference.py" in cmd_str:
                inf_calls.append(list(cmd))
                slug = generate_slug(2)
                answers_path = results_dir / f"answers_{model}_{slug}.json"
                meta = make_fake_answers(answers_path, model, config_hash)
                c = init_db(db_path)
                register_inference_only(c, str(answers_path), meta)
                c.close()
            elif "evaluate.py" in cmd_str:
                eval_calls.append(list(cmd))
                idx = list(cmd).index("--answers") + 1
                ap = Path(cmd[idx])
                with open(ap) as f:
                    inf_data = json.load(f)
                slug = generate_slug(2)
                ep = results_dir / f"eval_results_{ap.stem}_{slug}.json"
                make_fake_eval(ep, ap, inf_data["metadata"])
                c = init_db(db_path)
                register_run(c, str(ap), str(ep))
                c.close()

        return fake_run_step, inf_calls, eval_calls

    def _run_once(self, rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False):
        from scripts.analysis.runs_db import init_db

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        if no_reuse:
            argv.append("--no-reuse")

        conn = init_db(db_path)
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        conn.close()

    def test_no_reuse_produces_second_inference_file(self, tmp_path):
        """Two runs with --no-reuse produce two distinct inference files."""
        from scripts.analysis.runs_db import init_db
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"
        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        fake_run_step, inf_calls, _ = self._make_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False)
        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=True)

        answers_files = list(results_dir.glob("answers_*.json"))
        assert len(answers_files) == 2, f"Expected 2 inference files, got {answers_files}"
        assert len(inf_calls) == 2, f"Expected inference called twice, got {len(inf_calls)}"

    def test_no_reuse_produces_second_eval_file(self, tmp_path):
        """Two runs with --no-reuse produce two distinct eval files."""
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"
        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        fake_run_step, _, eval_calls = self._make_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False)
        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=True)

        eval_files = list(results_dir.glob("eval_results_*.json"))
        assert len(eval_files) == 2, f"Expected 2 eval files, got {eval_files}"
        assert len(eval_calls) == 2

    def test_run2_eval_uses_run2_inference_file(self, tmp_path):
        """Run 2's eval must use Run 2's inference file, not Run 1's."""
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"
        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        fake_run_step, _, eval_calls = self._make_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False)

        # Record which answers files exist after run 1
        after_run1 = set(results_dir.glob("answers_*.json"))

        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=True)

        # Run 2's answers file is the new one that appeared after run 1
        after_run2 = set(results_dir.glob("answers_*.json"))
        new_files = after_run2 - after_run1
        assert len(new_files) == 1, f"Expected exactly 1 new answers file from run 2, got {new_files}"
        run2_answers = str(next(iter(new_files)))

        assert len(eval_calls) == 2
        run2_eval_answers = eval_calls[1][eval_calls[1].index("--answers") + 1]
        assert run2_eval_answers == run2_answers, (
            f"Run 2 eval should use run 2 inference file.\n"
            f"Expected: {run2_answers}\nGot: {run2_eval_answers}"
        )

    def test_without_no_reuse_inference_is_skipped(self, tmp_path):
        """Without --no-reuse, a second run reuses the existing inference file."""
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"
        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        fake_run_step, inf_calls, _ = self._make_run_step(results_dir, db_path, model, config_hash)

        import run_benchmark as rb

        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False)
        self._run_once(rb, fake_run_step, cfg, tmp_path, db_path, no_reuse=False)

        # Inference ran only once; the second run reused the first result
        assert len(inf_calls) == 1, (
            f"Inference should run only once when --no-reuse is not set, ran {len(inf_calls)} times"
        )
