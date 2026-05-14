"""
runs.py  (scripts/tracking/runs.py)

Data layer for run tracking: joins runs.db metadata with per-question
inference and eval results into a flat pandas DataFrame.

One row per (run, question).

Per-tag scores from eval results are pivoted into columns:
  accuracy_score, structure_score, depth_score, citations_score, ...

Python API
----------
    from scripts.tracking.runs import query_runs

    df = query_runs(provider="claude")

CLI
---
    # List matching runs (one row per run)
    uv run scripts/tracking/runs.py list [--provider claude] [--judge gpt-4.1]
    uv run scripts/tracking/runs.py list --inf-run-id octopus buffalo
    uv run scripts/tracking/runs.py list --eval-run-id tuna marmot

    # Export flat per-(run, question) table
    uv run scripts/tracking/runs.py export [--provider claude] [--output out.csv]

    # Preview runs to delete (dry-run)
    uv run scripts/tracking/runs.py delete --provider claude
    uv run scripts/tracking/runs.py delete --id 5

    # Actually delete (add --yes; optionally remove result files from disk too)
    uv run scripts/tracking/runs.py delete --provider claude --yes
    uv run scripts/tracking/runs.py delete --id 5 --yes --files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent

from scripts.tracking.db import init_db, _DEFAULT_DB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_results(path: str) -> tuple[list[dict], dict]:
    """Load JSON file; return (results_list, metadata_dict)."""
    full = _REPO_ROOT / path
    if not full.exists():
        return [], {}
    with open(full, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("results", []), data.get("metadata") or {}


def _build_inference_index(results: list[dict]) -> dict[str, dict]:
    return {r["question"]: r for r in results if r.get("question")}


def _build_eval_index(results: list[dict]) -> dict[str, dict]:
    # Include skipped entries that carry zero scores (overall_score is not None),
    # so aggregate metrics reflect the full question set rather than answered-only.
    return {
        r["question"]: r
        for r in results
        if r.get("question") and (not r.get("skipped") or r.get("overall_score") is not None)
    }


def _rows_for_run(run: dict) -> list[dict]:
    """Return a list of flat dicts — one per question — for a single DB run row."""
    inf_results, _inf_meta = _load_json_results(run["inference_file"])
    eval_results, _eval_meta = _load_json_results(run["eval_file"])

    inf_index  = _build_inference_index(inf_results)
    eval_index = _build_eval_index(eval_results)

    all_questions = set(inf_index) | set(eval_index)
    rows = []

    for question in all_questions:
        inf = inf_index.get(question, {})
        ev  = eval_index.get(question, {})

        row: dict = {
            # Run-level metadata (from DB)
            "run_db_id":            run["id"],
            "inf_run_id":           run["inf_run_id"],
            "eval_run_id":          run["eval_run_id"],
            "timestamp":            run["timestamp"],
            "provider":             run["provider"],
            "model":                run["model"],
            "judge":                run["judge"],
            "inference_config_hash": run["inference_config_hash"],
            "eval_config_hash":     run["eval_config_hash"],
            "inference_file":       run["inference_file"],
            "eval_file":            run["eval_file"],
            # Question
            "question":             question,
            "plugin":               inf.get("plugin") or ev.get("plugin"),
            "segment":              inf.get("segment") or ev.get("segment"),
            # Inference metrics
            "tool_call_count":      inf.get("tool_call_count"),
            "successful_tool_calls": inf.get("successful_tool_calls"),
            "inference_time_secs":  inf.get("inference_time_secs"),
            "answer_length":        len(inf["answer"]) if inf.get("answer") else None,
            "answer":              inf.get("answer") if inf.get("answer") else None,
            "tag_reasoning":           ev.get("tag_reasoning") if ev.get("tag_reasoning") else None,

            # Overall eval score
            "overall_score":        ev.get("overall_score"),
        }

        # Per-tag scores — pivot tag_scores dict into {tag}_score columns
        tag_scores: dict = ev.get("tag_scores") or {}
        for tag, score in tag_scores.items():
            row[f"{tag}_score"] = score

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_where_clause(
    provider: Optional[str | list[str]] = None,
    model: Optional[str | list[str]] = None,
    judge: Optional[str | list[str]] = None,
    inf_run_id: Optional[str | list[str]] = None,
    inference_config_hash: Optional[str] = None,
    run_id: Optional[int] = None,
    eval_run_id: Optional[str | list[str]] = None,
) -> tuple[str, list]:
    """Build a SQL WHERE clause from the given filters.

    Returns (where_string, params_list). where_string is empty when no filters
    are provided, otherwise starts with 'WHERE '.

    inf_run_id and eval_run_id accept either a single string or a list of strings;
    lists are translated to IN (?, ...) clauses.
    """
    clauses: list[str] = []
    params:  list      = []
    if run_id is not None:
        clauses.append("id = ?")
        params.append(run_id)
    if provider is not None:
        clauses.append("provider = ?")
        params.append(provider)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    if judge is not None:
        clauses.append("judge = ?")
        params.append(judge)
    if inf_run_id is not None:
        ids = [inf_run_id] if isinstance(inf_run_id, str) else inf_run_id
        placeholders = ",".join("?" * len(ids))
        clauses.append(f"inf_run_id IN ({placeholders})")
        params.extend(ids)
    if eval_run_id is not None:
        ids = [eval_run_id] if isinstance(eval_run_id, str) else eval_run_id
        placeholders = ",".join("?" * len(ids))
        clauses.append(f"eval_run_id IN ({placeholders})")
        params.extend(ids)
    if inference_config_hash is not None:
        clauses.append("inference_config_hash = ?")
        params.append(inference_config_hash)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def query_runs(
    db_path: Optional[Path] = None,
    provider: Optional[str | list[str]] = None,
    model: Optional[str | list[str]] = None,
    judge: Optional[str | list[str]] = None,
    inf_run_id: Optional[str | list[str]] = None,
    inference_config_hash: Optional[str] = None,
    eval_run_id: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """Return a flat DataFrame with one row per (run, question).

    Filters are ANDed together; None means no filter on that column.
    inf_run_id and eval_run_id accept a single string or a list of strings.
    """
    db_path = db_path or _DEFAULT_DB
    conn = init_db(db_path)

    where, params = _build_where_clause(provider, model, judge, inf_run_id, inference_config_hash, eval_run_id=eval_run_id)
    runs  = conn.execute(f"SELECT * FROM runs {where}", params).fetchall()

    if not runs:
        return pd.DataFrame()

    all_rows: list[dict] = []
    for run in runs:
        all_rows.extend(_rows_for_run(dict(run)))

    conn.close()
    return pd.DataFrame(all_rows)


def list_runs(
    db_path: Optional[Path] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    judge: Optional[str] = None,
    inf_run_id: Optional[str | list[str]] = None,
    eval_run_id: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """Return one row per registered run (DB metadata only, no per-question expansion)."""
    db_path = db_path or _DEFAULT_DB
    conn = init_db(db_path)

    where, params = _build_where_clause(provider, model, judge, inf_run_id, eval_run_id=eval_run_id)
    rows  = conn.execute(
        f"SELECT * FROM runs {where} ORDER BY timestamp DESC, id DESC", params
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def delete_runs(
    db_path: Optional[Path] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    judge: Optional[str] = None,
    inf_run_id: Optional[str | list[str]] = None,
    eval_run_id: Optional[str | list[str]] = None,
    run_id: Optional[int] = None,
    dry_run: bool = True,
    delete_files: bool = False,
) -> int:
    """Delete matching runs from runs.db.

    Args:
        dry_run:      If True (default), print what would be deleted without touching the DB.
        delete_files: If True, also remove the inference and eval result files from disk.

    Returns:
        Number of rows deleted (or that would be deleted in dry-run mode).
    """
    db_path = db_path or _DEFAULT_DB
    conn = init_db(db_path)

    where, params = _build_where_clause(provider, model, judge, inf_run_id, run_id=run_id, eval_run_id=eval_run_id)
    rows = conn.execute(f"SELECT * FROM runs {where} ORDER BY timestamp DESC, id DESC", params).fetchall()

    if not rows:
        conn.close()
        return 0

    for row in rows:
        r = dict(row)
        tag = "(dry-run) would delete" if dry_run else "deleting"
        print(f"  {tag} id={r['id']}  {r['provider'] or '?'}  {r['inf_run_id'] or '?'}  {r.get('timestamp', '')[:19]}")
        print(f"    inf:  {r['inference_file']}")
        if r['eval_file']:
            print(f"    eval: {r['eval_file']}")

    if not dry_run:
        ids = [dict(r)["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", ids)
        conn.commit()

        if delete_files:
            for row in rows:
                r = dict(row)
                for key in ("inference_file", "eval_file"):
                    rel = r.get(key)
                    if rel:
                        p = _REPO_ROOT / rel
                        if p.exists():
                            p.unlink()
                            print(f"  deleted file: {rel}")

    conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id",       default=None, type=int, help="Filter by DB row id")
    parser.add_argument("--provider", default=None, help="Filter by provider (claude/openai)")
    parser.add_argument("--model",    default=None, help="Filter by model name")
    parser.add_argument("--judge",    default=None, help="Filter by judge deployment name")
    parser.add_argument("--inf-run-id",  nargs="+", default=None, dest="inf_run_id",  metavar="ID", help="Filter by inference run ID(s)")
    parser.add_argument("--eval-run-id", nargs="+", default=None, dest="eval_run_id", metavar="ID", help="Filter by eval run ID(s)")
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=f"Path to runs.db (default: {_DEFAULT_DB})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the run-tracking database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- list sub-command ---------------------------------------------------
    list_p = sub.add_parser("list", help="List matching runs (one row per run)")
    _add_filters(list_p)

    # -- export sub-command -------------------------------------------------
    export_p = sub.add_parser("export", help="Export flat per-(run, question) table")
    _add_filters(export_p)
    export_p.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output CSV path (default: print to stdout)",
    )

    # -- delete sub-command -------------------------------------------------
    delete_p = sub.add_parser("delete", help="Delete matching runs from the DB (dry-run by default)")
    _add_filters(delete_p)
    delete_p.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Actually delete (without this flag the command is a dry-run)",
    )
    delete_p.add_argument(
        "--files",
        action="store_true",
        default=False,
        help="Also delete the inference and eval result files from disk",
    )

    args = parser.parse_args()
    db_path = Path(args.db) if args.db else None

    if args.cmd == "list":
        df = list_runs(
            db_path=db_path,
            provider=args.provider,
            model=args.model,
            judge=args.judge,
            inf_run_id=args.inf_run_id,
            eval_run_id=args.eval_run_id,
        )
        if df.empty:
            print("No runs found.")
        else:
            cols = [
                "id", "inf_run_id", "eval_run_id", "timestamp", "provider", "model", "judge",
                "inference_config_hash", "eval_config_hash",
                "inference_file", "eval_file",
            ]
            cols = [c for c in cols if c in df.columns]
            pd.set_option("display.max_colwidth", 60)
            pd.set_option("display.width", 200)
            print(df[cols].to_string(index=False))

    elif args.cmd == "delete":
        n = delete_runs(
            db_path=db_path,
            provider=args.provider,
            model=args.model,
            judge=args.judge,
            inf_run_id=args.inf_run_id,
            eval_run_id=args.eval_run_id,
            run_id=args.id,
            dry_run=not args.yes,
            delete_files=args.files,
        )
        if n == 0:
            print("No matching runs found.")
        elif not args.yes:
            print(f"\n  {n} run(s) would be deleted. Re-run with --yes to confirm.")
        else:
            print(f"\n  Deleted {n} run(s).")

    elif args.cmd == "export":
        df = query_runs(
            db_path=db_path,
            provider=args.provider,
            model=args.model,
            judge=args.judge,
            inf_run_id=args.inf_run_id,
            eval_run_id=args.eval_run_id,
        )
        if df.empty:
            print("No runs found.")
            return
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Exported {len(df)} rows to {args.output}")
        else:
            print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
