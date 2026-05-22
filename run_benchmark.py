"""
run_benchmark.py - Finance Agent Benchmark pipeline entry point

Chains: data prep -> inference -> evaluation.

Inference runs register themselves in results/runs.db (runs table, eval_file=NULL)
immediately on completion.  Evaluation runs fill in the eval fields on the same row.

Before running inference, the pipeline checks whether an existing inference
result with a matching config hash is already registered in runs.
If one is found the inference step is skipped automatically and the existing
file is reused.  Pass --no-reuse to always re-run inference.

The pipeline resolves the correct inference file for each provider by config hash
(via find_latest_inference) and passes --answers explicitly to evaluate.py, so
running --provider all always evaluates the matching provider's inference file.

Usage:
  uv run run_benchmark.py
  uv run run_benchmark.py --provider claude -n 10
  uv run run_benchmark.py --skip-prep --skip-inference
  uv run run_benchmark.py --provider openai --skip-eval
  uv run run_benchmark.py --no-reuse   # force fresh inference even if hash matches
"""

import argparse
import os
import subprocess
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent


def load_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(cmd: list, step_label: str) -> None:
    """Run a subprocess command, printing the step label. Raises on non-zero exit."""
    print(f"\n{'=' * 60}")
    print(f"  {step_label}")
    print(f"{'=' * 60}\n")
    print(f"  Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def build_inference_cmd(provider: str, args: argparse.Namespace) -> list:
    """Build the uv run command for inference.py, forwarding relevant args."""
    cmd = ["uv", "run", "scripts/inference/inference.py", "--provider", provider]
    if args.limit is not None:
        cmd += ["-n", str(args.limit)]
    if args.no_shuffle:
        cmd += ["--no-shuffle"]
    if args.model is not None:
        cmd += ["--model", args.model]
    return cmd


def build_eval_cmd(args: argparse.Namespace, answers_file: str) -> list:
    """Build the uv run command for evaluate.py."""
    cmd = ["uv", "run", "scripts/evaluation/evaluate.py", "--answers", answers_file]
    return cmd


def _try_compute_hash(config: dict, provider: str, args: argparse.Namespace) -> str | None:
    """Compute the inference config hash for a provider without running inference.

    Applies the same CLI overrides (--no-shuffle, --model) that will be forwarded
    to inference.py so the computed hash matches what inference.py will store.

    Returns None if the computation fails (e.g. missing MCP config file).
    """
    try:
        if provider == "finance_agent":
            from scripts.data_prep.prepare_finance_agent_answers import compute_finance_agent_config_hash  # type: ignore
            finance_agent_cfg = config.get("data_prep", {}).get("finance_agent", {})
            return compute_finance_agent_config_hash(finance_agent_cfg)
        from scripts.inference.inference import compute_inference_config_hash  # type: ignore
        # Build an effective config that reflects CLI overrides
        effective = dict(config)
        if args.no_shuffle:
            effective = dict(effective)
            effective["inference"] = dict(effective.get("inference", {}))
            effective["inference"]["shuffle"] = False
        if args.model:
            effective = dict(effective)
            effective[provider] = dict(effective.get(provider, {}))
            effective[provider]["model"] = args.model
        return compute_inference_config_hash(effective, provider)
    except Exception as exc:
        print(f"  [warn] Could not compute inference config hash for {provider}: {exc}")
        return None


def _open_db():
    """Open (or create) runs.db.  Returns None on failure."""
    try:
        from scripts.tracking.db import init_db  # type: ignore
        db_path = REPO_ROOT / "results" / "runs.db"
        return init_db(db_path)
    except Exception as exc:
        print(f"  [warn] Could not open runs.db: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance Agent Benchmark pipeline: data prep -> inference -> evaluation."
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "openai", "finance_agent", "all"],
        default="all",
        help="Which inference provider to run (default: all; 'all' runs claude+openai only)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N questions (passed to inference scripts)",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        default=False,
        help="Skip the data prep step",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        default=False,
        help="Skip the inference step",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip the evaluation step",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        default=False,
        help="Do not shuffle questions before inference (default: shuffle)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model override passed to inference (only sensible with a single provider)",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        default=False,
        help="Force re-running inference even if a matching result already exists in runs.db",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip the analysis step (compare_runs.py)",
    )
    args = parser.parse_args()

    # Load .env and check for ERP token (not needed for finance_agent-only runs)
    load_dotenv()
    if args.provider != "finance_agent" and not os.getenv("ERP_MCP_TOKEN"):
        print(
            "\nWARNING: ERP_MCP_TOKEN is not set. "
            "Run `uv run refresh_erp_token.py` to fetch a token. "
            "Inference will fail if the ERP MCP server is required.\n"
        )

    # Load config for default model names
    config = load_config()

    # Determine active providers
    if args.provider == "all":
        providers = ["claude", "openai"]
    else:
        providers = [args.provider]

    analysis_providers = [p for p in providers if p != "finance_agent"]
    run_analysis = not args.skip_analysis and not args.skip_eval and bool(analysis_providers)

    total_steps = (
        (0 if args.skip_prep else 1)
        + (0 if args.skip_inference else len(providers))
        + (0 if args.skip_eval else len(providers))
        + (0 if not run_analysis else 1)
    )
    step = 0

    # Open runs.db once for the whole pipeline run (needed for reuse lookup)
    db_conn = _open_db()

    # Step: Data prep
    if not args.skip_prep:
        step += 1
        run_step(
            ["uv", "run", "scripts/data_prep/preprocess.py"],
            f"Step {step}/{total_steps}: Data Prep",
        )

    # Step: Inference
    if not args.skip_inference:
        for provider in providers:
            step += 1
            reused_file: str | None = None
            config_hash: str | None = _try_compute_hash(config, provider, args)

            if not args.no_reuse and db_conn is not None and config_hash:
                from scripts.tracking.db import find_reusable_inference  # type: ignore
                match = find_reusable_inference(db_conn, config_hash)
                if match and (REPO_ROOT / match).exists():
                    reused_file = match
                    print(f"\n{'=' * 60}")
                    print(f"  Step {step}/{total_steps}: Inference ({provider}) — SKIPPED")
                    print(f"{'=' * 60}")
                    print(f"  Reusing existing inference results: {match}\n")

            if reused_file is None:
                if provider == "finance_agent":
                    cmd = ["uv", "run", "scripts/data_prep/prepare_finance_agent_answers.py"]
                else:
                    cmd = build_inference_cmd(provider, args)
                run_step(cmd, f"Step {step}/{total_steps}: Inference ({provider})")

    # Step: Evaluation
    if not args.skip_eval:
        for provider in providers:
            step += 1
            if provider == "finance_agent":
                model_name = "finance_agent"
            elif args.model and len(providers) == 1:
                model_name = args.model
            elif provider == "claude":
                model_name = config["claude"]["model"]
            else:
                model_name = config["openai"]["model"]
            config_hash = _try_compute_hash(config, provider, args)
            answers_file: str | None = None
            if db_conn is not None and config_hash:
                from scripts.tracking.db import find_latest_inference  # type: ignore
                result = find_latest_inference(db_conn, config_hash)
                if result:
                    answers_file = str(REPO_ROOT / result[0])
                    if not Path(answers_file).exists():
                        raise RuntimeError(
                            f"Inference file for provider '{provider}' exists in DB but not on disk: {answers_file}"
                        )
            if not answers_file:
                raise RuntimeError(
                    f"No inference file found for provider '{provider}'. "
                    "Run inference first, or use evaluate.py --answers directly."
                )

            if not args.no_reuse and db_conn is not None and config_hash:
                import hashlib as _hashlib, json as _json
                from scripts.tracking.db import find_reusable_eval  # type: ignore
                _eval_snapshot = {
                    "judge_deployment": config["evaluation"]["judge_deployment"],
                    "eval_file": config["data_prep"]["output"],
                }
                eval_config_hash = _hashlib.sha256(
                    _json.dumps(_eval_snapshot, sort_keys=True).encode()
                ).hexdigest()[:12]
                eval_match = find_reusable_eval(db_conn, config_hash, eval_config_hash)
                if eval_match and (REPO_ROOT / eval_match[0]).exists():
                    print(f"\n{'=' * 60}")
                    print(f"  Step {step}/{total_steps}: Evaluation ({provider} / {model_name}) — SKIPPED")
                    print(f"{'=' * 60}")
                    print(f"  Reusing existing eval results: {eval_match[0]}\n")
                    continue

            eval_cmd = build_eval_cmd(args, answers_file)
            run_step(eval_cmd, f"Step {step}/{total_steps}: Evaluation ({provider} / {model_name})")

    # Step: Analysis
    if run_analysis:
        step += 1
        run_step(
            ["uv", "run", "scripts/analysis/compare_runs.py", "--runs"] + analysis_providers,
            f"Step {step}/{total_steps}: Analysis",
        )

    if db_conn is not None:
        db_conn.close()

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
