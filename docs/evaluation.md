# Evaluation

`scripts/evaluation/evaluate.py` scores inference results against `data/dataset.yaml` using an LLM judge via DSPy. The judge is configured in `config.yaml` (`evaluation.judge_deployment` for the model name, `evaluation.judge_client` for the client). The model can be overridden at runtime with `--judge-model`.

## Structure

Each question has one or more **tags** (e.g. `accuracy`, `relevance`, `groundedness`, `citations`, `structure`, `depth`). Each tag has a list of assertions. The evaluator runs **one judge call per tag per question** and scores each assertion on a continuous **0.0–1.0 scale**.

Fractional values (0.25, 0.5, 0.75) are used for partial compliance — assertions are not binary pass/fail. A tag's score is the mean of its assertion scores (excluding `null` scores). The overall score is the mean across all scoreable tags.

The judge uses `dspy.Predict` (not ChainOfThought). The DSPy signatures include an explicit `reasoning` output field, making a CoT wrapper redundant.

## Plugin-aware routing

The evaluation behaviour differs by plugin and tag:

| Plugin | Tag | Method |
|---|---|---|
| all plugins | `groundedness` | LLM judge with source content captured at inference time |
| all plugins | `accuracy` | LLM judge with accuracy-specific partial-credit instructions |
| all plugins | everything else | Standard LLM judge (`TagAssertionEvaluator`) |

## Groundedness

Source content for groundedness evaluation is captured **at inference time only** — no fetching occurs during evaluation.

Two source collections are combined:

- **Tool call outputs**: successful `WebFetch` and MCP tool call outputs embedded in the inference result. `WebSearch` outputs are excluded — they contain URL/title citation lists rather than page content, so including them would score against citation metadata rather than source content. This is the primary source for Claude inference.
- **`sources` dict**: a `{url: content}` mapping embedded in the inference result at inference time. This covers OpenAI inference, whose API does not expose page content in tool call outputs.

If no source content is available after combining both collections, the groundedness tag is **skipped** and omitted from scoring.

The judge uses fractional scores (0.0–1.0) and `null` for assertions where the source content is unusable (e.g. blocked or error pages). `null` scores are excluded from the tag mean.

## Business Brief evaluation

For `business_brief` questions, the standard tag pipeline always runs (accuracy, citations, clarity, depth, groundedness, recency, relevance, structure).

Section-by-section evaluation is **opt-in** via the `--sections` CLI flag (default: off). When enabled:

1. **Standard tag evaluation** — same tag loop as all other plugins (relevance, structure, depth, etc.).
2. **Section-specific evaluation** — the response is first parsed into sections via `BusinessBriefSectionExtractor`, then each section is scored across quality dimensions via `BusinessBriefSectionEvaluator`. Results are stored in `section_assertion_scores` and `section_reasoning` fields on the result entry.

The section rubric is loaded from the YAML file configured under `data_prep.datasets[plugin=business_brief].sections_rubric` in `config.yaml`. The YAML format is `{section_key: {dimension: [assertion_texts]}}`. Without `--sections`, section extraction and evaluation are skipped entirely.

## Token usage tracking

Token usage is tracked per result entry and in the run summary.

Each evaluated result includes:

```json
"token_usage": {
  "input_tokens": 1234,
  "output_tokens": 456,
  "total_tokens": 1690
}
```

Token counts are summed across all tag judge calls for that question (including business brief section extraction and evaluation). The summary includes `total_token_usage` with the same three fields aggregated across all evaluated questions.

## Output

Results are written to `results/eval_results_{answers_stem}_{eval_slug}.json`. The output uses a metadata envelope:

```json
{
  "metadata": {
    "judge_deployment": "gpt-4.1",
    "eval_run_id": "falcon",
    "inf_run_id": "panda",
    "run_timestamp": "2026-04-21T10:00:00Z",
    "config_hash": "abc123def456",
    "eval_file": "data/dataset.yaml",
    "answers_file": "results/answers_gpt-5.2_panda.json",
    "inference_metadata": { ... }
  },
  "summary": {
    "total": 100,
    "evaluated": 95,
    "skipped": 5,
    "overall_score": 0.812,
    "tag_scores": { "accuracy": 0.9, "relevance": 0.85, ... },
    "total_token_usage": {
      "input_tokens": 120000,
      "output_tokens": 45000,
      "total_tokens": 165000
    }
  },
  "results": [
    {
      "question": "...",
      "plugin": "finance_qa",
      "overall_score": 0.75,
      "tag_scores": { "accuracy": 1.0, "groundedness": 0.5 },
      "assertion_scores": { "accuracy": { "<assertion text>": 0.75 }, ... },
      "tag_reasoning": { "accuracy": "...", ... },
      "token_usage": { "input_tokens": 1234, "output_tokens": 456, "total_tokens": 1690 },
      "skipped": false
    }
  ]
}
```

## Resume behaviour

If the `--output` file already exists, already-evaluated questions are preserved and not re-evaluated. Only entries with `skipped: true` are retried on the next run — successfully evaluated entries (those without `skipped: true`) are treated as done. The output file is flushed to disk after each completed future, so progress is preserved even if the run is interrupted.

## Running

```bash
# Auto-find the latest inference result from runs.db
uv run scripts/evaluation/evaluate.py

# Evaluate a specific model's output
uv run scripts/evaluation/evaluate.py --model gpt-5.2

# Evaluate an explicit answers file
uv run scripts/evaluation/evaluate.py --answers results/my_answers.json --output results/my_eval.json

# Override the judge model
uv run scripts/evaluation/evaluate.py --judge-model gpt-4.1

# Evaluate only specific tags or plugins
uv run scripts/evaluation/evaluate.py --tags accuracy --tags groundedness
uv run scripts/evaluation/evaluate.py --plugins erp_qa --plugins finance_qa
uv run scripts/evaluation/evaluate.py --plugins bb      # alias for business_brief; fqa and eqa also work

# Evaluate only business brief questions with section-by-section evaluation enabled
uv run scripts/evaluation/evaluate.py --plugins bb --sections

# Control parallelism (default: from config.yaml evaluation.threads)
uv run scripts/evaluation/evaluate.py --threads 8
```
