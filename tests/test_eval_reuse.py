"""
test_eval_reuse.py

Tests for eval-only re-run when eval config changes but inference config is unchanged:
  1. Changing evaluation.judge_deployment does NOT affect the inference config hash.
  2. When a matching inference run exists in runs.db, the pipeline skips inference
     and reuses the existing file.
  3. After changing only eval settings, the pipeline reuses inference and produces
     a fresh eval file.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import make_fake_answers, make_fake_eval, make_test_config, load_real_config


# ---------------------------------------------------------------------------
# Unit tests: eval config does not influence inference hash
# ---------------------------------------------------------------------------

class TestInferenceHashIsolation:
    def test_judge_deployment_change_does_not_affect_hash(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["evaluation"]["judge_deployment"] = "totally-different-judge"
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 == h2, (
            "Changing evaluation.judge_deployment must not change the inference config hash"
        )

    def test_eval_threads_change_does_not_affect_hash(self):
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["evaluation"]["threads"] = 99
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 == h2

    def test_inference_hash_includes_system_instructions(self):
        """Sanity check: inference hash DOES change when shared inference settings change."""
        from scripts.inference.inference import compute_inference_config_hash

        cfg = load_real_config()
        h1 = compute_inference_config_hash(cfg, "openai")

        cfg["shared"]["max_workers"] = 999
        h2 = compute_inference_config_hash(cfg, "openai")

        assert h1 != h2


# ---------------------------------------------------------------------------
# Integration tests: inference reuse when only eval config changes
# ---------------------------------------------------------------------------

class TestEvalOnlyRerun:
    def _setup(self, tmp_path):
        from scripts.analysis.runs_db import init_db
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"

        cfg = make_test_config(results_dir)
        config_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]

        return results_dir, db_path, cfg, config_hash, model

    def _preregister_inference(self, results_dir, db_path, model, config_hash):
        """Create a pre-existing inference file and register it in the temp DB."""
        from scripts.analysis.runs_db import init_db, register_inference_only

        answers_path = results_dir / f"answers_{model}_panda.json"
        inf_meta = make_fake_answers(answers_path, model, config_hash)
        conn = init_db(db_path)
        register_inference_only(conn, str(answers_path), inf_meta)
        conn.close()
        return answers_path, inf_meta

    def _make_run_step(self, results_dir, db_path):
        from scripts.analysis.runs_db import init_db, register_run
        from coolname import generate_slug

        inf_calls = []
        eval_calls = []

        def fake_run_step(cmd, label):
            cmd_str = " ".join(str(c) for c in cmd)
            if "inference.py" in cmd_str:
                inf_calls.append(list(cmd))
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

    def test_inference_skipped_when_hash_matches(self, tmp_path):
        """When runs.db already has a matching inference hash, inference is skipped."""
        results_dir, db_path, cfg, config_hash, model = self._setup(tmp_path)
        answers_path, _ = self._preregister_inference(results_dir, db_path, model, config_hash)
        fake_run_step, inf_calls, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
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

        assert len(inf_calls) == 0, (
            f"Inference should be skipped when hash matches, but ran {len(inf_calls)} times"
        )
        assert len(eval_calls) == 1

    def test_eval_uses_preexisting_inference_file(self, tmp_path):
        """Eval is called with the pre-existing answers file when inference is reused."""
        results_dir, db_path, cfg, config_hash, model = self._setup(tmp_path)
        answers_path, _ = self._preregister_inference(results_dir, db_path, model, config_hash)
        fake_run_step, _, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
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

        assert len(eval_calls) == 1
        cmd = eval_calls[0]
        answers_arg = cmd[cmd.index("--answers") + 1]
        assert answers_arg == str(answers_path), (
            f"Eval should use pre-existing answers file.\n"
            f"Expected: {answers_path}\nGot: {answers_arg}"
        )

    def test_changing_judge_deployment_still_reuses_inference(self, tmp_path):
        """Changing judge_deployment in config reuses inference (hash unchanged)."""
        results_dir, db_path, cfg, config_hash, model = self._setup(tmp_path)
        answers_path, _ = self._preregister_inference(results_dir, db_path, model, config_hash)
        fake_run_step, inf_calls, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        # Simulate: user changed judge_deployment between runs
        cfg_changed = dict(cfg)
        cfg_changed["evaluation"] = dict(cfg["evaluation"])
        cfg_changed["evaluation"]["judge_deployment"] = "gpt-4.1-turbo"

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
        conn = init_db(db_path)
        with (
            patch.object(rb, "run_step", fake_run_step),
            patch.object(rb, "load_config", return_value=cfg_changed),
            patch.object(rb, "REPO_ROOT", tmp_path),
            patch.object(rb, "_open_db", return_value=conn),
            patch.object(sys, "argv", argv),
        ):
            rb.main()
        conn.close()

        # Inference should still be skipped
        assert len(inf_calls) == 0, (
            "Changing judge_deployment should not trigger new inference"
        )
        # Eval should run fresh
        assert len(eval_calls) == 1

    def test_new_eval_file_created_after_eval_only_rerun(self, tmp_path):
        """After an eval-only re-run (inference reused), a new eval file exists."""
        results_dir, db_path, cfg, config_hash, model = self._setup(tmp_path)
        self._preregister_inference(results_dir, db_path, model, config_hash)
        fake_run_step, _, _ = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = [
            "run_benchmark.py", "--provider", "openai",
            "-n", "10", "--no-shuffle", "--skip-prep",
        ]
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

        eval_files = list(results_dir.glob("eval_results_*.json"))
        assert len(eval_files) == 1
        # Inference file was not duplicated
        answers_files = list(results_dir.glob("answers_*.json"))
        assert len(answers_files) == 1


# ---------------------------------------------------------------------------
# Integration tests: eval reuse when both inference and eval config match
# ---------------------------------------------------------------------------

class TestEvalReuseByHash:
    """Tests for skipping eval when a completed matching eval already exists in DB."""

    def _compute_eval_config_hash(self, cfg: dict) -> str:
        import hashlib, json as _json
        snapshot = {
            "judge_deployment": cfg["evaluation"]["judge_deployment"],
            "eval_file":        cfg["data_prep"]["output"],
        }
        return hashlib.sha256(_json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:12]

    def _setup(self, tmp_path):
        from scripts.analysis.runs_db import init_db
        from scripts.inference.inference import compute_inference_config_hash

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        db_path = results_dir / "runs.db"
        cfg = make_test_config(results_dir)
        inf_hash = compute_inference_config_hash(cfg, "openai")
        model = cfg["openai"]["model"]
        return results_dir, db_path, cfg, inf_hash, model

    def _preregister_full_run(self, results_dir, db_path, model, inf_hash, eval_hash):
        """Write fake inference + eval files and register the pair in the DB."""
        from scripts.analysis.runs_db import init_db, register_run

        answers_path = results_dir / f"answers_{model}_panda.json"
        inf_meta = make_fake_answers(answers_path, model, inf_hash)
        eval_path = results_dir / f"eval_results_answers_{model}_panda_falcon.json"
        make_fake_eval(eval_path, answers_path, inf_meta, eval_config_hash=eval_hash)
        conn = init_db(db_path)
        register_run(conn, str(answers_path), str(eval_path))
        conn.close()
        return answers_path, eval_path

    def _make_run_step(self, results_dir, db_path):
        from scripts.analysis.runs_db import init_db, register_run
        from coolname import generate_slug

        inf_calls = []
        eval_calls = []

        def fake_run_step(cmd, label):
            cmd_str = " ".join(str(c) for c in cmd)
            if "inference.py" in cmd_str:
                inf_calls.append(list(cmd))
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

    def test_eval_skipped_when_matching_eval_exists(self, tmp_path):
        """Both inference and eval are skipped when both hashes match a completed run."""
        results_dir, db_path, cfg, inf_hash, model = self._setup(tmp_path)
        eval_hash = self._compute_eval_config_hash(cfg)
        self._preregister_full_run(results_dir, db_path, model, inf_hash, eval_hash)
        fake_run_step, inf_calls, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = ["run_benchmark.py", "--provider", "openai", "-n", "10", "--no-shuffle", "--skip-prep"]
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

        assert len(inf_calls) == 0, "Inference should be skipped"
        assert len(eval_calls) == 0, "Eval should be skipped when matching eval exists"

    def test_eval_reruns_when_eval_config_hash_differs(self, tmp_path):
        """Eval re-runs when eval_config_hash doesn't match, even if inference is reused."""
        results_dir, db_path, cfg, inf_hash, model = self._setup(tmp_path)
        # Register with a stale eval hash (e.g. old judge)
        self._preregister_full_run(results_dir, db_path, model, inf_hash, "old_eval_000")
        fake_run_step, inf_calls, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = ["run_benchmark.py", "--provider", "openai", "-n", "10", "--no-shuffle", "--skip-prep"]
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

        assert len(inf_calls) == 0, "Inference should still be skipped"
        assert len(eval_calls) == 1, "Eval should re-run when eval_config_hash differs"

    def test_no_reuse_forces_both_inference_and_eval(self, tmp_path):
        """--no-reuse bypasses both inference and eval reuse."""
        results_dir, db_path, cfg, inf_hash, model = self._setup(tmp_path)
        eval_hash = self._compute_eval_config_hash(cfg)
        self._preregister_full_run(results_dir, db_path, model, inf_hash, eval_hash)
        fake_run_step, inf_calls, eval_calls = self._make_run_step(results_dir, db_path)

        from scripts.analysis.runs_db import init_db
        import run_benchmark as rb

        argv = ["run_benchmark.py", "--provider", "openai", "-n", "10", "--no-shuffle", "--skip-prep", "--no-reuse"]
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

        assert len(inf_calls) == 1, "Inference should re-run with --no-reuse"
        assert len(eval_calls) == 1, "Eval should re-run with --no-reuse"
