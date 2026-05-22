"""
register.py  (scripts/tracking/register.py)

CLI tool for retroactively registering existing result pairs into runs.db.

Usage:
    # Auto-discover all answer/eval pairs in results/
    uv run scripts/tracking/register.py

    # Register specific files
    uv run scripts/tracking/register.py results/answers_gpt-5.2.json \\
        results/eval_results_answers_gpt-5.2_ground_truth.json

Auto-discovery scans results/ for answers_*.json files and matches them to
eval_results_{stem}_*.json files by glob. Multiple eval runs per inference file
are each registered as a separate row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent

from scripts.tracking.db import init_db, register_run, _DEFAULT_DB


def _find_eval_files(answers_path: Path, results_dir: Path) -> list[Path]:
    """Return all eval results files matching *answers_path* (one per eval slug)."""
    stem = answers_path.stem  # e.g. "answers_gpt-5.2_panda"
    return sorted(results_dir.glob(f"eval_results_{stem}_*.json"))


def _discover_pairs(results_dir: Path) -> list[tuple[Path, Path]]:
    """Scan results_dir for answers_*.json files and match to eval files."""
    pairs: list[tuple[Path, Path]] = []
    for answers_path in sorted(results_dir.glob("answers_*.json")):
        eval_paths = _find_eval_files(answers_path, results_dir)
        if eval_paths:
            for eval_path in eval_paths:
                pairs.append((answers_path, eval_path))
        else:
            print(f"  [skip] No matching eval file found for {answers_path.name}")
    return pairs


def _register_pair(
    conn,
    inference_file: Path,
    eval_file: Path,
    dry_run: bool = False,
) -> bool:
    """Register one pair.  Returns True on success."""
    inf_str  = str(inference_file)
    eval_str = str(eval_file)

    if dry_run:
        print(f"  [dry-run] Would register: {inference_file.name} + {eval_file.name}")
        return True

    try:
        row_id = register_run(conn, inf_str, eval_str)
        print(f"  [ok] id={row_id}: {inference_file.name} + {eval_file.name}")
        return True
    except Exception as exc:
        print(f"  [error] {inference_file.name} + {eval_file.name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retroactively register existing result pairs into runs.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "Explicit files to register: provide pairs of "
            "(answers_*.json eval_results_*.json) or omit to auto-discover."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Directory to scan for auto-discovery (default: results/)",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=f"Path to runs.db (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be registered without writing to the DB",
    )
    args = parser.parse_args()

    db_path      = Path(args.db) if args.db else _DEFAULT_DB
    results_dir  = Path(args.results_dir) if args.results_dir else (_REPO_ROOT / "results")

    conn = init_db(db_path)

    if args.files:
        # Explicit pairs: expect even number of files, alternating inf/eval
        if len(args.files) % 2 != 0:
            parser.error(
                "When providing explicit files, supply pairs: "
                "answers_X.json eval_results_X.json [answers_Y.json eval_results_Y.json ...]"
            )
        pairs: list[tuple[Path, Path]] = []
        it = iter(args.files)
        for inf_f, eval_f in zip(it, it):
            pairs.append((Path(inf_f), Path(eval_f)))
    else:
        print(f"Auto-discovering pairs in {results_dir} ...")
        pairs = _discover_pairs(results_dir)

    if not pairs:
        print("No pairs to register.")
        conn.close()
        return

    ok = 0
    for inf_path, eval_path in pairs:
        success = _register_pair(conn, inf_path, eval_path, dry_run=args.dry_run)
        if success:
            ok += 1

    conn.close()

    action = "Would register" if args.dry_run else "Registered"
    print(f"\n{action} {ok}/{len(pairs)} pair(s).")


if __name__ == "__main__":
    main()
