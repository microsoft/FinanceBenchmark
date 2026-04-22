"""
variance_analysis.py — Measure benchmark pipeline variance

Two modes:

  inference  — given multiple answers files for the same model, measure how
               consistently the model answers each question across runs.

  eval       — given multiple eval results files produced by running
               evaluate.py on the same answers file multiple times, measure
               how consistently the judge scores each answer.

Usage:
  uv run scripts/analysis/variance_analysis.py inference \\
      results/answers_gpt-5.2_run1.json results/answers_gpt-5.2_run2.json \\
      [--labels "Run 1" "Run 2"] [--output results/variance_inference_gpt-5.2.json]

  uv run scripts/analysis/variance_analysis.py eval \\
      results/eval_results_run1.json results/eval_results_run2.json \\
      [--labels "Run 1" "Run 2"] [--output results/variance_eval_gpt-5.2.json]
"""

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path


# ── Maths helpers ─────────────────────────────────────────────────────────────

def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    variance = sum((x - m) ** 2 for x in vals) / (n - 1)
    return math.sqrt(variance)


def _rnd(x: float | None, decimals: int = 4) -> float | None:
    if x is None:
        return None
    return round(x, decimals)


def _grand_mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ── File loading ───────────────────────────────────────────────────────────────

def _load_answers(path: str) -> dict[str, dict]:
    """Load an answers file (flat array or envelope format) keyed by question."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "results" in raw:
        items = raw["results"]
    else:
        raise ValueError(f"Unrecognised answers format in {path!r}")

    return {item["question"]: item for item in items if item.get("question")}


def _load_eval_results(path: str) -> dict[str, dict]:
    """Load an eval results file keyed by question.

    Handles:
      - {"summary": ..., "results": [...]}                       (old format)
      - {"metadata": ..., "summary": ..., "results": [...]}      (new format)
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        items = raw["results"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unrecognised eval results format in {path!r}")

    return {item["question"]: item for item in items if item.get("question")}


# ── Common: find questions present in all runs ─────────────────────────────────

def _common_questions(
    runs: list[dict[str, dict]],
) -> tuple[list[str], int]:
    """Return (sorted list of common question keys, count of skipped questions)."""
    if not runs:
        return [], 0
    common = set(runs[0].keys())
    all_qs = set(runs[0].keys())
    for run in runs[1:]:
        common &= set(run.keys())
        all_qs |= set(run.keys())
    skipped = len(all_qs) - len(common)
    return sorted(common), skipped


# ── Inference variance ─────────────────────────────────────────────────────────

def _compute_inference_variance(
    runs: list[dict[str, dict]],
    common_questions: list[str],
) -> tuple[list[dict], dict]:
    """
    Per-question and summary inference variance metrics.

    Returns (per_question_list, summary_dict).
    """
    per_question = []

    # Accumulators for summary
    tool_call_stds:       list[float] = []
    answer_length_stds:   list[float] = []
    inference_time_stds:  list[float] = []
    answered_rates:       list[float] = []

    for question in common_questions:
        entries = [run[question] for run in runs]

        tool_call_counts       = [e.get("tool_call_count", 0) for e in entries]
        successful_tool_calls  = [e.get("successful_tool_calls", 0) for e in entries]
        answer_lengths         = [
            len(e["answer"]) if e.get("answer") else 0 for e in entries
        ]
        answered_flags         = [
            1 if (e.get("answer") and e["answer"]) else 0 for e in entries
        ]
        inference_times        = [e.get("inference_time_secs", 0.0) for e in entries]

        tcc_std  = _std(tool_call_counts)
        al_std   = _std(answer_lengths)
        it_std   = _std(inference_times)
        ans_rate = sum(answered_flags) / len(answered_flags)

        tool_call_stds.append(tcc_std)
        answer_length_stds.append(al_std)
        inference_time_stds.append(it_std)
        answered_rates.append(ans_rate)

        metrics = {
            "tool_call_count": {
                "mean": _rnd(_mean(tool_call_counts)),
                "std":  _rnd(tcc_std),
                "min":  min(tool_call_counts),
                "max":  max(tool_call_counts),
            },
            "successful_tool_calls": {
                "mean": _rnd(_mean(successful_tool_calls)),
                "std":  _rnd(_std(successful_tool_calls)),
            },
            "answer_length_chars": {
                "mean": _rnd(_mean(answer_lengths)),
                "std":  _rnd(al_std),
            },
            "answered_rate": _rnd(ans_rate),
            "inference_time_secs": {
                "mean": _rnd(_mean(inference_times)),
                "std":  _rnd(it_std),
            },
        }

        per_question.append({"question": question, "metrics": metrics})

    summary = {
        "mean_tool_call_count_std": _rnd(_grand_mean(tool_call_stds)),
        "mean_answer_length_std":   _rnd(_grand_mean(answer_length_stds)),
        "mean_inference_time_std":  _rnd(_grand_mean(inference_time_stds)),
        "mean_answered_rate":       _rnd(_grand_mean(answered_rates)),
    }

    return per_question, summary


# ── Eval variance ──────────────────────────────────────────────────────────────

def _get_all_tags(runs: list[dict[str, dict]], common_questions: list[str]) -> list[str]:
    """Collect all tag names that appear in any run for any common question."""
    tags: set[str] = set()
    for question in common_questions:
        for run in runs:
            entry = run[question]
            ts = entry.get("tag_scores") or {}
            tags.update(ts.keys())
    return sorted(tags)


def _compute_eval_variance(
    runs: list[dict[str, dict]],
    common_questions: list[str],
) -> tuple[list[dict], dict]:
    """
    Per-question and summary eval/judge variance metrics.

    Returns (per_question_list, summary_dict).
    """
    all_tags = _get_all_tags(runs, common_questions)

    per_question = []

    # Accumulators
    overall_score_stds: list[float] = []

    # Per-tag accumulators: tag -> list of per-question stds
    tag_score_stds:           dict[str, list[float]] = {t: [] for t in all_tags}
    # Per-tag, per-assertion accumulators: tag -> {assertion_text -> [agreement rates]}
    tag_assertion_agreements: dict[str, dict[str, list[float]]] = {t: {} for t in all_tags}

    for question in common_questions:
        entries = [run[question] for run in runs]

        # ── overall_score ──────────────────────────────────────────────────────
        overall_scores = [e.get("overall_score") for e in entries]
        os_std = _std(overall_scores)
        overall_score_stds.append(os_std)

        # ── per-tag scores ─────────────────────────────────────────────────────
        tag_metrics: dict = {}
        for tag in all_tags:
            scores_across_runs = [
                (e.get("tag_scores") or {}).get(tag) for e in entries
            ]
            valid_scores = [s for s in scores_across_runs if s is not None]

            ts_std  = _std(scores_across_runs)
            ts_mean = _mean(scores_across_runs)
            ts_min  = min(valid_scores) if valid_scores else None
            ts_max  = max(valid_scores) if valid_scores else None

            tag_score_stds[tag].append(ts_std)

            # ── per-assertion agreement for this tag ───────────────────────────
            # assertion_scores in eval results is a nested dict: {tag: {assertion_text: score}}
            assertion_scores_per_run: list[dict[str, int | None]] = []

            for e in entries:
                raw_as = e.get("assertion_scores") or {}
                if isinstance(raw_as, dict):
                    if tag in raw_as and isinstance(raw_as[tag], dict):
                        assertion_scores_per_run.append(raw_as[tag])
                    else:
                        assertion_scores_per_run.append({})
                else:
                    assertion_scores_per_run.append({})

            assertion_texts: set[str] = set()
            for asr in assertion_scores_per_run:
                assertion_texts.update(asr.keys())

            per_assertion: dict = {}
            for atext in sorted(assertion_texts):
                scores_for_assertion = [asr.get(atext) for asr in assertion_scores_per_run]
                n_runs = len(scores_for_assertion)
                counts = Counter(scores_for_assertion)
                most_common_count = counts.most_common(1)[0][1]
                agreement = most_common_count / n_runs
                per_assertion[atext] = _rnd(agreement)

                if tag not in tag_assertion_agreements:
                    tag_assertion_agreements[tag] = {}
                if atext not in tag_assertion_agreements[tag]:
                    tag_assertion_agreements[tag][atext] = []
                tag_assertion_agreements[tag][atext].append(agreement)

            tag_metrics[tag] = {
                "tag_score": {
                    "mean": _rnd(ts_mean),
                    "std":  _rnd(ts_std),
                    "min":  _rnd(ts_min),
                    "max":  _rnd(ts_max),
                },
                "assertion_agreement": per_assertion,
            }

        metrics = {
            "overall_score": {
                "mean": _rnd(_mean(overall_scores)),
                "std":  _rnd(os_std),
            },
            "per_tag": tag_metrics,
        }

        per_question.append({"question": question, "metrics": metrics})

    # ── Summary ────────────────────────────────────────────────────────────────
    per_tag_summary: dict = {}
    for tag in all_tags:
        stds = tag_score_stds[tag]
        mean_std = _rnd(_grand_mean(stds))

        all_agreements: list[float] = []
        for atext, agr_list in tag_assertion_agreements[tag].items():
            all_agreements.extend(agr_list)
        mean_agr = _rnd(_grand_mean(all_agreements))

        per_tag_summary[tag] = {
            "mean_score_std":           mean_std,
            "mean_assertion_agreement": mean_agr,
        }

    summary = {
        "mean_overall_score_std": _rnd(_grand_mean(overall_score_stds)),
        "per_tag": per_tag_summary,
    }

    return per_question, summary


# ── Stdout summary printers ────────────────────────────────────────────────────

def _print_inference_summary(
    n_runs: int,
    n_common: int,
    n_skipped: int,
    summary: dict,
    labels: list[str],
) -> None:
    sep = "-" * 50
    print()
    print(f"  Inference variance summary ({n_runs} runs, {n_common} questions)")
    print(f"  {sep}")
    if n_skipped:
        print(f"  Warning: {n_skipped} question(s) skipped (not present in all runs)")
    print(f"  Mean tool-call-count std:    {summary['mean_tool_call_count_std']}")
    print(f"  Mean answer-length std:      {summary['mean_answer_length_std']}")
    print(f"  Mean inference-time std:     {summary['mean_inference_time_std']}")
    print(f"  Mean answered rate:          {summary['mean_answered_rate']}")
    print()


def _print_eval_summary(
    n_runs: int,
    n_common: int,
    n_skipped: int,
    summary: dict,
    labels: list[str],
) -> None:
    sep = "-" * 45
    print()
    print(f"  Eval variance summary ({n_runs} runs, {n_common} questions)")
    print(f"  {sep}")
    if n_skipped:
        print(f"  Warning: {n_skipped} question(s) skipped (not present in all runs)")

    mean_os_std = summary.get("mean_overall_score_std")
    print(f"  Overall score std (mean):   {mean_os_std}")

    per_tag = summary.get("per_tag", {})
    if per_tag:
        print("  Per-tag score std:")
        max_tag_len = max(len(t) for t in per_tag)
        for tag, tag_s in per_tag.items():
            std_val = tag_s.get("mean_score_std")
            pad = " " * (max_tag_len - len(tag))
            print(f"    {tag}:{pad}  {std_val}")

        print("  Per-tag assertion agreement:")
        for tag, tag_s in per_tag.items():
            agr = tag_s.get("mean_assertion_agreement")
            pad = " " * (max_tag_len - len(tag))
            if agr is not None:
                print(f"    {tag}:{pad}  {agr * 100:.1f}%")
            else:
                print(f"    {tag}:{pad}  n/a")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure inference or eval/judge variance across multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["inference", "eval"],
        help="Type of variance to measure",
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Two or more JSON files (answers files for inference, eval result files for eval)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Display names for each file (must match file count if provided)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON path (default: results/variance_{mode}_{timestamp}.json)",
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("At least 2 files are required.")

    if args.labels and len(args.labels) != len(args.files):
        parser.error(
            f"--labels count ({len(args.labels)}) must match file count ({len(args.files)})."
        )

    labels = args.labels or [f"Run {i + 1}" for i in range(len(args.files))]

    # ── Load files ──────────────────────────────────────────────────────────────
    loader = _load_answers if args.mode == "inference" else _load_eval_results

    runs: list[dict[str, dict]] = []
    for path in args.files:
        try:
            runs.append(loader(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            sys.exit(f"ERROR loading {path!r}: {exc}")

    # ── Find common questions ───────────────────────────────────────────────────
    common_questions, n_skipped = _common_questions(runs)
    n_common = len(common_questions)

    if n_skipped:
        warnings.warn(
            f"{n_skipped} question(s) not present in all runs — skipping them.",
            stacklevel=2,
        )

    if not common_questions:
        sys.exit("ERROR: No questions common to all runs.")

    # ── Compute variance ────────────────────────────────────────────────────────
    if args.mode == "inference":
        per_question, summary = _compute_inference_variance(runs, common_questions)
    else:
        per_question, summary = _compute_eval_variance(runs, common_questions)

    # ── Build output ────────────────────────────────────────────────────────────
    output = {
        "mode":                args.mode,
        "n_runs":              len(runs),
        "files":               args.files,
        "labels":              labels,
        "n_questions_common":  n_common,
        "n_questions_skipped": n_skipped,
        "summary":             summary,
        "per_question":        per_question,
    }

    # ── Write JSON ──────────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"variance_{args.mode}_{ts}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Variance analysis written to: {output_path}")

    # ── Stdout summary ──────────────────────────────────────────────────────────
    if args.mode == "inference":
        _print_inference_summary(len(runs), n_common, n_skipped, summary, labels)
    else:
        _print_eval_summary(len(runs), n_common, n_skipped, summary, labels)


if __name__ == "__main__":
    main()
