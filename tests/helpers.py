"""Shared helpers for benchmark pipeline tests."""
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent


def load_real_config() -> dict:
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_test_config(tmp_results_dir: Path) -> dict:
    """Real config with output_dir redirected to a temp dir."""
    cfg = load_real_config()
    cfg = dict(cfg)
    cfg["shared"] = dict(cfg["shared"])
    cfg["shared"]["output_dir"] = str(tmp_results_dir)
    # Keep shuffle=False for determinism in tests
    cfg["inference"] = dict(cfg.get("inference", {}))
    cfg["inference"]["shuffle"] = False
    return cfg


def make_fake_answers(path: Path, model: str, config_hash: str, n: int = 10) -> dict:
    """Write a valid inference envelope and return its metadata dict."""
    slug = path.stem.rsplit("_", 1)[-1]
    metadata = {
        "model": model,
        "provider": "openai",
        "inf_slug": slug,
        "run_id": slug,
        "run_timestamp": "2026-01-01T00:00:00Z",
        "config_hash": config_hash,
        "config_snapshot": {
            "model": model,
            "shuffle": False,
            "retry_failed": True,
            "max_workers": 2,
            "max_tool_calls": 20,
            "timeout": 120,
            "system_instructions": "test",
            "input_file": "data/dataset.yaml",
            "output_dir": str(path.parent),
        },
    }
    data = {
        "metadata": metadata,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return metadata


def make_fake_eval(
    path: Path,
    answers_path: Path,
    inf_metadata: dict,
    eval_config_hash: str = "evalcfg00000",
) -> None:
    """Write a valid eval result envelope."""
    slug = path.stem.rsplit("_", 1)[-1]
    inf_slug = str(answers_path.stem).rsplit("_", 1)[-1]
    data = {
        "metadata": {
            "judge_deployment": "gpt-4.1",
            "eval_slug": slug,
            "inf_slug": inf_slug,
            "run_timestamp": "2026-01-01T00:00:00Z",
            "config_hash": eval_config_hash,
            "eval_file": "data/dataset.yaml",
            "answers_file": str(answers_path),
            "inference_metadata": inf_metadata,
        },
        "summary": {
            "total": 10,
            "evaluated": 10,
            "skipped": 0,
            "overall_score": 0.8,
            "tag_scores": {},
        },
        "results": [
            {"question": f"Q{i}", "overall_score": 0.8, "tag_scores": {}, "skipped": False}
            for i in range(10)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
