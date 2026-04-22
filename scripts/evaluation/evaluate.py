"""
evaluate.py — Finance QA evaluator (per-tag, plugin-aware)

One judge call per tag per question. Plugin-aware routing:
  - finance_qa / groundedness → LLM judge with fetched source content
  - finance_qa / citations    → binary pass/fail: ≥1 unique source URL cited (no LLM)
  - erp_qa / citations        → standard LLM judge (rubric assertions)
  - all others                → standard LLM judge

Inputs:
  --model             model name — resolves to {output_dir}/answers_{model}.json
  --answers           path to answers JSON (overrides --model)
  --eval-file         path to preprocessed eval YAML (default: data/dataset.yaml)
  --output            override output file path
  --threads           number of parallel judge threads (default: 4)
  --judge-model       override config evaluation.judge_deployment (OpenAI model name)
  --no-cache          document that DSPy LM caching is disabled (no additional effect)

Output:
  results/eval_results_{answers_stem}_{eval_slug}.json
"""

import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


import dspy
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

_cfg = yaml.safe_load(open(Path(__file__).parent.parent.parent / "config.yaml", encoding="utf-8"))

EVAL_FILE_DEFAULT    = _cfg["data_prep"]["output"]
OUTPUT_DIR           = Path(_cfg["shared"]["output_dir"])

JUDGE_MODEL            = os.getenv("OPENAI_JUDGE_MODEL", _cfg["evaluation"]["judge_deployment"])
JUDGE_REASONING_EFFORT = _cfg["evaluation"].get("reasoning_effort", "medium")
_CFG_RUN_ID            = _cfg["shared"].get("run_id") or None

TAG_EVALUATOR_INSTRUCTIONS          = _cfg["evaluation"]["tag_evaluator_instructions"].strip()
GROUNDEDNESS_EVALUATOR_INSTRUCTIONS = _cfg["evaluation"]["groundedness_evaluator_instructions"].strip()
BB_ACCURACY_ASSERTIONS              = _cfg["evaluation"].get("bb_accuracy_assertions", [])

# ── Token usage tracking ─────────────────────────────────────────────────────

_history_lock = threading.Lock()


def _extract_usage(entry: dict) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a DSPy LM history entry."""
    u = entry.get("usage")
    if not u:
        resp = entry.get("response")
        if resp:
            u = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
    if not u:
        return 0, 0
    if isinstance(u, dict):
        return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    return getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)


def _lm_call(module, **kwargs) -> tuple[object, dict]:
    """Call a DSPy module and capture token usage from the LM history delta.

    Records history length before the call and reads new entries after.
    Concurrent entries from other threads may occasionally be included —
    acceptable for billing estimates; the summary total is accurate overall.
    """
    lm = dspy.settings.lm
    with _history_lock:
        n_before = len(lm.history)
    pred = module(**kwargs)
    with _history_lock:
        new_entries = list(lm.history[n_before:])
    input_t = output_t = 0
    for entry in new_entries:
        pt, ct = _extract_usage(entry)
        input_t += pt
        output_t += ct
    return pred, {"input_tokens": input_t, "output_tokens": output_t}


# ── DSPy signatures ────────────────────────────────────────────────────────────

class TagAssertionEvaluator(dspy.Signature):
    """Placeholder — overwritten from config.yaml at runtime."""

    query:      str = dspy.InputField(desc="The user's financial query")
    response:   str = dspy.InputField(desc="The generated response to evaluate")
    tag:        str = dspy.InputField(desc="The evaluation dimension being assessed (e.g. accuracy, relevance)")
    assertions: str = dspy.InputField(desc="Newline-separated list of assertions to evaluate for this tag")

    assertion_scores: str = dspy.OutputField(
        desc="JSON array of scores (0.0–1.0, or null), one per assertion in the same order as the assertions input. Use fractional values for partial compliance. Use null for assertions that are not applicable to this query/response."
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the scoring for this tag")


class GroundednessEvaluator(dspy.Signature):
    """Placeholder — overwritten from config.yaml at runtime."""

    query:      str = dspy.InputField(desc="The user's financial query")
    response:   str = dspy.InputField(desc="The generated response to evaluate")
    sources:    str = dspy.InputField(desc="Source content retrieved during inference (web searches and fetched pages)") # TODO: update so that it works for erp qa
    assertions: str = dspy.InputField(desc="Newline-separated groundedness assertions to evaluate")

    assertion_scores: str = dspy.OutputField(
        desc="JSON array of scores (0.0–1.0, or null), one per assertion in the same order as the assertions input. Use fractional values for partial compliance. Use null if the source content is unusable (e.g. blocked/error pages)."
    )
    reasoning: str = dspy.OutputField(desc="Explanation of how well the response is supported by the sources")


tag_assertion_evaluator  = TagAssertionEvaluator.with_instructions(TAG_EVALUATOR_INSTRUCTIONS)
groundedness_evaluator   = GroundednessEvaluator.with_instructions(GROUNDEDNESS_EVALUATOR_INSTRUCTIONS)


class BusinessBriefAccuracyEvaluator(dspy.Signature):
    """Evaluate factual accuracy of a business brief against a ground truth reference brief."""

    query:        str = dspy.InputField(desc="The business brief query (e.g. 'Business Brief of IBM')")
    response:     str = dspy.InputField(desc="The generated business brief to evaluate")
    ground_truth: str = dspy.InputField(desc="Reference business brief containing verified facts")
    assertions:   str = dspy.InputField(desc="Newline-separated list of accuracy assertions to evaluate")

    assertion_scores: str = dspy.OutputField(
        desc="JSON array of scores (0, 1, or null), one per assertion in the same order as the assertions input. Use null only when the assertion cannot be evaluated (e.g. ground truth lacks relevant data)."
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of how the response compares to the ground truth for each assertion"
    )


class BusinessBriefSectionExtractor(dspy.Signature):
    """Extract sections from a business brief response and map them to standard section keys."""

    response: str = dspy.InputField(desc="The business brief response to parse")
    expected_section_keys: str = dspy.InputField(desc="JSON array of expected section keys to look for")
    extracted_sections: str = dspy.OutputField(
        desc='JSON object mapping each section key to its extracted text content, or null if the section is not present. Keys must exactly match the expected_section_keys.'
    )


class BusinessBriefSectionEvaluator(dspy.Signature):
    """Evaluate all quality dimensions for a single section of a business brief."""

    query: str = dspy.InputField(desc="The business brief query (e.g. 'Business Brief of Apple Inc.')")
    section_key: str = dspy.InputField(desc="The section being evaluated (e.g. 'financials')")
    section_content: str = dspy.InputField(desc="The extracted content of this section")
    assertions_by_dimension: str = dspy.InputField(
        desc='JSON object mapping each dimension name to a list of assertion strings to evaluate'
    )
    scores_by_dimension: str = dspy.OutputField(
        desc='JSON object mapping each dimension name to a JSON array of scores (0, 1, or null) — one per assertion in the same order as the input list. Use null for assertions not applicable to this section/query.'
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the scores for each dimension")


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_assertions(tag_entry: dict) -> str:
    """Format assertions as a plain list. Keys in assertion_scores must match a['text'] exactly."""
    return "\n".join(f"- {a['text']}" for a in tag_entry.get("assertions", []))


def parse_assertion_scores(pred, tag_entry: dict) -> dict:
    """Parse assertion_scores from the judge. Expects a JSON array in assertion order;
    zips with assertion texts to produce a {text: score} dict."""
    try:
        raw = json.loads(pred.assertion_scores)
    except Exception:
        return {}
    if isinstance(raw, list):
        assertions = tag_entry.get("assertions", [])
        return {a["text"]: s for a, s in zip(assertions, raw)}
    if isinstance(raw, dict):
        return raw  # fallback for unexpected dict output
    return {}


def compute_tag_score(assertion_scores: dict, tag_entry: dict) -> float | None:
    scores = [
        assertion_scores.get(a["text"])
        for a in tag_entry.get("assertions", [])
        if assertion_scores.get(a["text"]) is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def count_markdown_citations(text: str) -> int:
    """Count unique source URLs embedded as markdown hyperlinks [text](url).

    Deduplicates by URL so the same source cited multiple times counts once.
    """
    urls = re.findall(r'\[(?:[^\]]+)\]\((https?://[^\)]+)\)', text)
    return len(set(urls))


# ── Source extraction ──────────────────────────────────────────────────────────


def get_sources(answer_entry: dict) -> str:
    """Return source content for groundedness evaluation.

    Combines two inference-time source collections:
    - tool_calls outputs: successful tool call outputs captured during inference
      (WebSearch/WebFetch/MCP). Covers all plugins for Claude.
    - sources dict: {url: content} fetched at inference time. Covers OpenAI,
      whose API does not expose page content in tool call outputs.

    No fetching is done at eval time — sources must be captured during inference.
    Returns "" if nothing is available -> evaluation is skipped.
    """
    parts = []

    for tc in answer_entry.get("tool_calls", []):
        if tc.get("success") and tc.get("output"):
            tool_name = tc.get("tool", "tool")
            parts.append(f"[Tool: {tool_name}]\n{tc['output']}")

    for url, snippet in (answer_entry.get("sources") or {}).items():
        if snippet:
            parts.append(f"[Source: {url}]\n{snippet}")

    return "\n\n---\n\n".join(parts)


# ── Business Brief rubric loader ──────────────────────────────────────────────

def load_bb_rubric(rubric_path: str) -> tuple[dict, dict]:
    """Load the business brief rubric JSON and build a nested lookup.

    Returns: (section_rubric, common_rubric)

    section_rubric: {section_key: {dimension: [assertion_texts]}}
        Only category_specific assertions, keyed by (section, dimension).

    common_rubric: {dimension: [assertion_texts]}
        Whole-document assertions evaluated once against the full response.
    """
    with open(rubric_path, encoding="utf-8") as f:
        raw = json.load(f)

    sections   = raw["meta"]["sections"]
    dimensions = raw["meta"]["dimensions"]
    common     = raw.get("common", {})
    cat_spec   = raw.get("category_specific", {})

    section_rubric: dict = {}
    for section in sections:
        section_rubric[section] = {}
        for dim in dimensions:
            key = f"{dim}.{section}"
            if key in cat_spec:
                section_rubric[section][dim] = cat_spec[key]

    common_rubric = {dim: assertions for dim, assertions in common.items() if assertions}

    return section_rubric, common_rubric


# ── Tag evaluation ────────────────────────────────────────────────────────────

def evaluate_tag(
    tag_entry: dict,
    question: str,
    answer: str,
    plugin: str,
    answer_entry: dict,
    judge: dspy.Module,
    groundedness_judge: dspy.Module,
) -> dict:
    """Evaluate a single tag. Returns a result dict with type in {scored, skipped}."""
    tag = tag_entry["tag"]

    # finance_qa citations: binary pass/fail based on unique source count, no LLM
    # Score applies to all assertions in the tag (shared + plugin-specific)
    if tag == "citations" and plugin == "finance_qa":
        has_citation = 1.0 if count_markdown_citations(answer) >= 1 else 0.0
        return {
            "type": "scored",
            "assertion_scores": {a["text"]: has_citation for a in tag_entry.get("assertions", [])},
            "reasoning": "",
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # groundedness: LLM judge with source content (web pages or ERP tool outputs)
    if tag == "groundedness":
        sources = get_sources(answer_entry)
        if not sources:
            return {"type": "skipped", "reason": "no sources available for groundedness"}
        pred, usage = _lm_call(
            groundedness_judge,
            query=question,
            response=answer,
            sources=sources,
            assertions=format_assertions(tag_entry),
        )
        return {
            "type":             "scored",
            "assertion_scores": parse_assertion_scores(pred, tag_entry),
            "reasoning":        getattr(pred, "reasoning", ""),
            "token_usage":      usage,
        }

    # All other tags: standard LLM judge
    pred, usage = _lm_call(
        judge,
        query=question,
        response=answer,
        tag=tag,
        assertions=format_assertions(tag_entry),
    )
    return {
        "type":             "scored",
        "assertion_scores": parse_assertion_scores(pred, tag_entry),
        "reasoning":        getattr(pred, "reasoning", ""),
        "token_usage":      usage,
    }


def evaluate_bb_sections(
    question: str,
    answer: str,
    rubric: dict,  # {section_key: {dimension: [assertion_texts]}} — category_specific only
    section_extractor: dspy.Module,
    section_evaluator: dspy.Module,
) -> dict:
    """Run section extraction and per-section quality evaluation for a business brief.

    Returns:
        assertion_scores: {"{section}.{dim}": {assertion_text: score}}
        section_reasoning: {section_key: reasoning} — only sections with at least one failure
    """
    section_keys = list(rubric.keys())

    bb_token_usage = {"input_tokens": 0, "output_tokens": 0}

    # Step 1: Extract sections
    try:
        ext_pred, usage = _lm_call(
            section_extractor,
            response=answer,
            expected_section_keys=json.dumps(section_keys),
        )
        bb_token_usage["input_tokens"]  += usage["input_tokens"]
        bb_token_usage["output_tokens"] += usage["output_tokens"]
        extracted_sections: dict = json.loads(ext_pred.extracted_sections)
    except Exception:
        extracted_sections = {}

    # Step 2: Evaluate each section
    assertion_scores: dict = {}
    section_reasoning_raw: dict = {}

    for section_key in section_keys:
        section_content = extracted_sections.get(section_key)
        dim_assertions = rubric.get(section_key, {})

        if not section_content:
            for dim, texts in dim_assertions.items():
                assertion_scores[f"{section_key}.{dim}"] = {t: None for t in texts}
            continue

        assertions_by_dim = {dim: texts for dim, texts in dim_assertions.items() if texts}
        if not assertions_by_dim:
            continue

        try:
            eval_pred, usage = _lm_call(
                section_evaluator,
                query=question,
                section_key=section_key,
                section_content=section_content,
                assertions_by_dimension=json.dumps(assertions_by_dim),
            )
            bb_token_usage["input_tokens"]  += usage["input_tokens"]
            bb_token_usage["output_tokens"] += usage["output_tokens"]
            scores_by_dim: dict = json.loads(eval_pred.scores_by_dimension)
            section_reasoning_raw[section_key] = getattr(eval_pred, "reasoning", "")
        except Exception:
            scores_by_dim = {}
            section_reasoning_raw[section_key] = ""

        for dim, texts in assertions_by_dim.items():
            key = f"{section_key}.{dim}"
            raw_scores = scores_by_dim.get(dim, [])
            if isinstance(raw_scores, list):
                assertion_scores[key] = {
                    t: (raw_scores[idx] if idx < len(raw_scores) else None)
                    for idx, t in enumerate(texts)
                }
            else:
                assertion_scores[key] = {t: None for t in texts}

    # Only keep reasoning for sections with at least one failure
    section_reasoning: dict = {
        section_key: reasoning
        for section_key, reasoning in section_reasoning_raw.items()
        if any(
            s == 0
            for ck, scores in assertion_scores.items()
            if ck.startswith(f"{section_key}.")
            for s in scores.values()
        )
    }

    return {
        "assertion_scores":  assertion_scores,
        "section_reasoning": section_reasoning,
        "token_usage":       bb_token_usage,
    }


def evaluate_entry(
    question: str,
    answer: str,
    plugin: str,
    tags: list,
    answer_entry: dict,
    judge: dspy.Module,
    groundedness_judge: dspy.Module,
    section_extractor: dspy.Module | None = None,
    section_evaluator: dspy.Module | None = None,
    bb_rubric: dict | None = None,
) -> dict:
    """Evaluate all tags for one Q&A entry and return the structured result.

    All plugins go through the standard tag evaluation loop.
    For business_brief, section-specific evaluation is run additionally and
    stored in separate output fields (section_assertion_scores, section_reasoning).
    """
    # ── Standard tag evaluation (all plugins) ────────────────────────────────
    all_assertion_scores: dict = {}
    tag_scores:           dict = {}
    tag_reasoning:        dict = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for tag_entry in tags:
        tag    = tag_entry["tag"]
        result = evaluate_tag(
            tag_entry, question, answer, plugin, answer_entry,
            judge, groundedness_judge,
        )

        if result["type"] == "scored":
            scores = result["assertion_scores"]
            all_assertion_scores[tag] = scores
            score = compute_tag_score(scores, tag_entry)
            if score is not None:
                tag_scores[tag] = score
            tag_reasoning[tag] = result.get("reasoning", "")
            u = result.get("token_usage", {})
            total_usage["input_tokens"]  += u.get("input_tokens", 0)
            total_usage["output_tokens"] += u.get("output_tokens", 0)
        # "skipped" tags are omitted from scores silently

    # ── Business Brief: section-specific evaluation ───────────────────────────
    section_assertion_scores: dict = {}
    section_reasoning:        dict = {}

    if plugin == "business_brief" and bb_rubric:
        sec = evaluate_bb_sections(
            question, answer, bb_rubric,
            section_extractor, section_evaluator,
        )
        section_assertion_scores = sec["assertion_scores"]
        section_reasoning        = sec["section_reasoning"]
        u = sec.get("token_usage", {})
        total_usage["input_tokens"]  += u.get("input_tokens", 0)
        total_usage["output_tokens"] += u.get("output_tokens", 0)

    scoreable = list(tag_scores.values())
    overall   = round(sum(scoreable) / len(scoreable), 3) if scoreable else None

    total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
    result: dict = {
        "question":        question,
        "plugin":          plugin,
        "actual_answer":   answer,
        "overall_score":   overall,
        "tag_scores":      tag_scores,
        "assertion_scores": all_assertion_scores,
        "tag_reasoning":   tag_reasoning,
        "token_usage":     total_usage,
        "skipped":         False,
    }
    if plugin == "business_brief":
        result["section_assertion_scores"] = section_assertion_scores
        result["section_reasoning"]        = section_reasoning
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA inference results.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model",   metavar="MODEL", help="Model name — resolves to {output_dir}/answers_{model}.json")
    source.add_argument("--answers", metavar="FILE",  help="Path to answers JSON")
    parser.add_argument("--eval-file",        default=EVAL_FILE_DEFAULT, metavar="FILE")
    parser.add_argument("--output",           default=None,              metavar="FILE")
    parser.add_argument("--threads",          type=int, default=_cfg["evaluation"]["threads"])
    parser.add_argument("--judge-model",      default=None, metavar="NAME",
                        help="Override config evaluation.judge_deployment (OpenAI model name)")
    parser.add_argument("--no-cache",         action="store_true",
                        help="Documents that DSPy LM caching is disabled (no additional effect)")
    args = parser.parse_args()

    # Apply CLI override for judge model
    judge_deployment = args.judge_model or JUDGE_MODEL

    if args.model:
        answers_file = str(OUTPUT_DIR / f"answers_{args.model}.json")
    elif args.answers:
        answers_file = args.answers
    else:
        # Auto-find the latest inference result from the DB
        try:
            from scripts.analysis.runs_db import init_db, _DEFAULT_DB
            db_conn = init_db(_DEFAULT_DB)
            row = db_conn.execute(
                "SELECT inference_file FROM runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            db_conn.close()
            if row:
                candidate = str(Path(__file__).parent.parent.parent / row["inference_file"])
                if Path(candidate).exists():
                    answers_file = candidate
                else:
                    raise FileNotFoundError(
                        f"Latest inference file from DB not found: {candidate}\n"
                        "Specify --answers <file> or --model <model>."
                    )
            else:
                raise FileNotFoundError(
                    "No inference runs found in the DB.\n"
                    "Specify --answers <file> or --model <model>."
                )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Could not auto-find answers file ({e}).\n"
                "Specify --answers <file> or --model <model>."
            ) from e

    # Extract inf_slug from the answers filename
    from scripts.analysis.runs_db import _extract_slug_from_filename, generate_unique_slug
    inf_slug = _extract_slug_from_filename(answers_file, "answers_")

    # Generate eval slug and build output filename
    eval_slug = generate_unique_slug(OUTPUT_DIR, "eval_results_", word_index=1)

    answers_stem = Path(answers_file).stem  # e.g. "answers_gpt-5.2_panda"
    default_output = str(OUTPUT_DIR / f"eval_results_{answers_stem}_{eval_slug}.json")
    output_file = args.output or default_output

    # ── Load answers (new envelope format or old flat list) ──────────────────
    with open(answers_file, encoding="utf-8") as f:
        raw_answers = json.load(f)

    if isinstance(raw_answers, list):
        answers = raw_answers
        inference_metadata = None
    else:
        answers = raw_answers.get("results", [])
        inference_metadata = raw_answers.get("metadata", None)

    # Resume: load already-evaluated results and restore config from existing run.
    # Handles both old format ({"summary":..., "results":[...]}) and new format
    # ({"metadata":..., "summary":..., "results":[...]}) — both have a "results" key.
    eval_config = {
        "judge_deployment":                  judge_deployment,
        "eval_file":                         args.eval_file,
        "tag_evaluator_instructions":        TAG_EVALUATOR_INSTRUCTIONS,
        "groundedness_evaluator_instructions": GROUNDEDNESS_EVALUATOR_INSTRUCTIONS,
    }
    existing: dict = {}
    resume_meta: dict = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_results = existing_data.get("results", []) if isinstance(existing_data, dict) else []
            for item in existing_results:
                if item.get("question") and not item.get("skipped"):
                    existing[item["question"]] = item
            resume_meta = existing_data.get("metadata", {}) if isinstance(existing_data, dict) else {}
            if resume_meta:
                snap = resume_meta.get("config_snapshot") or {}
                for key, value in snap.items():
                    if key in eval_config and value is not None:
                        eval_config[key] = value
                eval_slug = resume_meta.get("eval_run_id") or eval_slug
                inf_slug  = resume_meta.get("inf_run_id")  or inf_slug
            print(f"Resuming: {len(existing)} results already in {output_file}")
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    judge_deployment = eval_config["judge_deployment"]
    args.eval_file   = eval_config["eval_file"]

    print("=" * 80)
    print("Finance QA Evaluation")
    print("=" * 80)
    print(f"  Answers:          {answers_file}")
    print(f"  Eval file:        {args.eval_file}")
    print(f"  Output:           {output_file}")
    print(f"  Threads:          {args.threads}")
    print(f"  Judge deployment: {judge_deployment}")
    if inf_slug:
        print(f"  Inference slug:   {inf_slug}")
    print(f"  Eval slug:        {eval_slug}")

    with open(args.eval_file, encoding="utf-8") as f:
        eval_entries = yaml.safe_load(f)
    eval_index = {e["query"].strip(): e for e in eval_entries}

    print(f"\nInitializing judge LM: {judge_deployment}")
    judge_lm = dspy.LM(
        model=f"openai/{judge_deployment}",
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort=JUDGE_REASONING_EFFORT,
        cache=False,
    )
    dspy.settings.configure(lm=judge_lm)
    judge              = dspy.Predict(tag_assertion_evaluator)
    groundedness_judge = dspy.Predict(groundedness_evaluator)

    # ── Business Brief setup ─────────────────────────────────────────────────
    bb_rubric_path = "data/business_brief/rubric_business_brief_quality.json"
    for ds in _cfg.get("data_prep", {}).get("datasets", []):
        if ds.get("plugin") == "business_brief" and ds.get("bb_rubric"):
            bb_rubric_path = ds["bb_rubric"]
            break
    bb_rubric_path_abs = Path(__file__).parent.parent.parent / bb_rubric_path
    if bb_rubric_path_abs.exists():
        try:
            bb_rubric, _ = load_bb_rubric(str(bb_rubric_path_abs))
        except (KeyError, ValueError) as exc:
            bb_rubric = {}
            print(
                f"  [warn] Failed to load business brief rubric from {bb_rubric_path_abs}: {exc}; "
                "BB section eval will produce empty scores"
            )
    else:
        bb_rubric = {}
        print(f"  [warn] Business brief rubric not found at {bb_rubric_path_abs}; BB section eval will produce empty scores")
    section_extractor = dspy.Predict(BusinessBriefSectionExtractor)
    section_evaluator = dspy.Predict(BusinessBriefSectionEvaluator)

    # Build work list
    to_evaluate:     list = []
    skipped_results: list = []

    for entry in answers:
        question = entry.get("question", "")
        answer   = entry.get("answer")

        if question in existing:
            continue

        if entry.get("error") or not answer:
            skipped_results.append({
                "question": question,
                "skipped":  True,
                "reason":   entry.get("error", "no answer"),
            })
            continue

        eval_entry = eval_index.get(question.strip())
        if not eval_entry:
            skipped_results.append({
                "question": question,
                "skipped":  True,
                "reason":   "question not found in eval file",
            })
            continue

        plugin = entry.get("plugin") or eval_entry.get("plugin", "")
        tags   = eval_entry.get("tags") or []
        to_evaluate.append((question, answer, plugin, tags, entry))

    print(f"\nEvaluating {len(to_evaluate)} answers "
          f"({len(existing)} already done, {len(skipped_results)} skipped)...\n")

    new_results: list = []

    def _worker(item):
        question, answer, plugin, tags, answer_entry = item
        return evaluate_entry(
            question, answer, plugin, tags, answer_entry,
            judge, groundedness_judge,
            section_extractor=section_extractor,
            section_evaluator=section_evaluator,
            bb_rubric=bb_rubric,
        )

    # Build config snapshot and output metadata before the loop — all inputs are known
    config_snapshot = {
        "judge_deployment":                  judge_deployment,
        "eval_file":                         args.eval_file,
        "tag_evaluator_instructions":        TAG_EVALUATOR_INSTRUCTIONS,
        "groundedness_evaluator_instructions": GROUNDEDNESS_EVALUATOR_INSTRUCTIONS,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode()
    ).hexdigest()[:12]

    output_metadata = {
        "judge_deployment":   judge_deployment,
        "eval_run_id":        eval_slug,
        "inf_run_id":         inf_slug,
        "run_timestamp":      datetime.utcnow().isoformat() + "Z",
        "config_hash":        resume_meta.get("config_hash")   if resume_meta else config_hash,
        "eval_file":          args.eval_file,
        "answers_file":       answers_file,
        "inference_metadata": inference_metadata,
    }

    def _compute_summary(results: list) -> dict:
        evaluated_r   = [r for r in results if not r.get("skipped")]
        skipped_r     = [r for r in results if r.get("skipped")]
        overall_sc    = [r["overall_score"] for r in evaluated_r if r.get("overall_score") is not None]
        mean_sc       = sum(overall_sc) / len(overall_sc) if overall_sc else None
        tag_tot: dict = {}
        for r in evaluated_r:
            for tag, rate in r.get("tag_scores", {}).items():
                tag_tot.setdefault(tag, []).append(rate)
        tag_sum = {tag: round(sum(v) / len(v), 3) for tag, v in tag_tot.items()}
        total_input  = sum(r.get("token_usage", {}).get("input_tokens",  0) for r in evaluated_r)
        total_output = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in evaluated_r)
        return {
            "total":             len(results),
            "evaluated":         len(evaluated_r),
            "skipped":           len(skipped_r),
            "overall_score":     round(mean_sc, 3) if mean_sc is not None else None,
            "tag_scores":        tag_sum,
            "total_token_usage": {
                "input_tokens":  total_input,
                "output_tokens": total_output,
                "total_tokens":  total_input + total_output,
            },
        }

    def _flush(current_new_results: list) -> None:
        """Merge completed results with existing and skipped, then flush to disk.

        Preserves the original answers ordering. Called after each future completes
        so the output file always reflects progress to date and resume works correctly.
        All flushes happen in the main thread (as_completed is iterated sequentially),
        so no write lock is required.
        """
        q_to_new  = {r["question"]: r for r in current_new_results}
        q_to_skip = {r["question"]: r for r in skipped_results}
        partial: list = []
        for entry in answers:
            q = entry.get("question", "")
            if q in existing:
                partial.append(existing[q])
            elif q in q_to_new:
                partial.append(q_to_new[q])
            elif q in q_to_skip:
                partial.append(q_to_skip[q])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(
                {"metadata": output_metadata, "summary": _compute_summary(partial), "results": partial},
                fh, indent=2, ensure_ascii=False,
            )

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(_worker, item): item for item in to_evaluate}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                new_results.append(future.result())
            except Exception as exc:
                question = futures[future][0]
                new_results.append({
                    "question": question,
                    "skipped":  True,
                    "reason":   f"evaluation error: {exc}",
                })
            _flush(new_results)

    # Merge results preserving answers order
    question_to_new  = {r["question"]: r for r in new_results}
    question_to_skip = {r["question"]: r for r in skipped_results}
    all_results = []
    for entry in answers:
        q = entry.get("question", "")
        if q in existing:
            all_results.append(existing[q])
        elif q in question_to_new:
            all_results.append(question_to_new[q])
        elif q in question_to_skip:
            all_results.append(question_to_skip[q])

    # Compute final summary
    evaluated      = [r for r in all_results if not r.get("skipped")]
    skipped        = [r for r in all_results if r.get("skipped")]
    overall_scores = [r["overall_score"] for r in evaluated if r.get("overall_score") is not None]
    mean_score     = sum(overall_scores) / len(overall_scores) if overall_scores else None

    tag_totals: dict = {}
    for r in evaluated:
        for tag, rate in r.get("tag_scores", {}).items():
            tag_totals.setdefault(tag, []).append(rate)
    tag_summary = {tag: round(sum(v) / len(v), 3) for tag, v in tag_totals.items()}

    summary = {
        "total":         len(all_results),
        "evaluated":     len(evaluated),
        "skipped":       len(skipped),
        "overall_score": round(mean_score, 3) if mean_score is not None else None,
        "tag_scores":    tag_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": output_metadata, "summary": summary, "results": all_results},
            f, indent=2, ensure_ascii=False,
        )

    # Register the completed run in runs.db
    try:
        from scripts.analysis.runs_db import init_db, register_run, _DEFAULT_DB
        db_conn = init_db(_DEFAULT_DB)
        row_id = register_run(db_conn, answers_file, output_file, inf_run_id=inf_slug, eval_run_id=eval_slug)
        db_conn.close()
        print(f"  [runs.db] Registered run id={row_id}: {Path(answers_file).name} + {Path(output_file).name}")
    except Exception as exc:
        print(f"  [warn] Could not register run in runs.db: {exc}")

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\n  Evaluated: {summary['evaluated']}/{summary['total']} questions")
    if summary["overall_score"] is not None:
        print(f"  Overall score: {summary['overall_score']:.3f}")
    if tag_summary:
        print("\n  Per-tag pass rates:")
        for tag, rate in tag_summary.items():
            print(f"    {tag}: {rate:.1%}")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
