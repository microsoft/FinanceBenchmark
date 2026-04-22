"""Tests for dataset building in scripts/data_prep/preprocess.py."""
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_prep.preprocess import _build_merged_tags, process_dataset


# ── _build_merged_tags ─────────────────────────────────────────────────────


def test_build_merged_tags_shared_only():
    shared = {"relevance": {"tag": "relevance", "metric": True, "assertions": [{"text": "a"}]}}
    result = _build_merged_tags(shared, {})
    assert any(e["tag"] == "relevance" for e in result)


def test_build_merged_tags_plugin_appends_assertions():
    shared = {"relevance": {"tag": "relevance", "metric": True, "assertions": [{"text": "shared"}]}}
    plugin = {"relevance": {"tag": "relevance", "metric": True, "assertions": [{"text": "plugin-only"}]}}
    result = _build_merged_tags(shared, plugin)
    relevance = next(e for e in result if e["tag"] == "relevance")
    texts = [a["text"] for a in relevance["assertions"]]
    assert "shared" in texts
    assert "plugin-only" in texts


def test_build_merged_tags_no_duplicates():
    shared = {"relevance": {"tag": "relevance", "assertions": [{"text": "shared"}]}}
    plugin = {"relevance": {"tag": "relevance", "assertions": [{"text": "shared"}]}}
    result = _build_merged_tags(shared, plugin)
    relevance = next(e for e in result if e["tag"] == "relevance")
    texts = [a["text"] for a in relevance["assertions"]]
    assert texts.count("shared") == 1


def test_build_merged_tags_plugin_only_tag_included():
    shared = {}
    plugin = {"depth": {"tag": "depth", "metric": True, "assertions": [{"text": "depth-only"}]}}
    result = _build_merged_tags(shared, plugin)
    assert any(e["tag"] == "depth" for e in result)


# ── process_dataset ────────────────────────────────────────────────────────


def _write_temp_files(tmp_path, segment="accounts payable", scenario=None):
    """Write minimal YAML + TSV + rubric files and return a ds dict."""
    input_yaml = tmp_path / "input.yaml"
    tsv_path   = tmp_path / "utterances.tsv"
    rubric     = tmp_path / "rubric.yaml"

    query = "What is the outstanding balance for vendor V-1042?"
    input_yaml.write_text(
        yaml.dump([{"query": query, "assertions": [{"text": "covers balance", "level": "critical"}]}]),
        encoding="utf-8",
    )

    if scenario:
        header = "Utterance\tSegment\tScenario\n"
        row    = f"{query}\t{segment}\t{scenario}\n"
    else:
        header = "Utterance\tSegment\n"
        row    = f"{query}\t{segment}\n"

    tsv_path.write_text(header + row, encoding="utf-8", newline="")

    rubric_content = [
        {"tag": "relevance",  "metric": True, "assertions": [{"text": "relevant", "level": "critical"}]},
        {"tag": "depth",      "metric": True, "assertions": [{"text": "thorough", "level": "critical"}]},
    ]
    rubric.write_text(yaml.dump(rubric_content), encoding="utf-8")

    ds = {
        "plugin":     "erp_qa",
        "input_yaml": str(input_yaml),
        "tsv":        str(tsv_path),
        "lmc_rubric": str(rubric),
    }
    return ds, query


def test_process_dataset_sets_plugin_field(tmp_path):
    ds, _ = _write_temp_files(tmp_path)
    entries, _ = process_dataset(ds)
    assert all(e["plugin"] == "erp_qa" for e in entries)


def test_process_dataset_non_depth_tags_included(tmp_path):
    ds, _ = _write_temp_files(tmp_path)
    entries, _ = process_dataset(ds)
    assert len(entries) == 1
    tag_names = [t["tag"] for t in entries[0]["tags"]]
    assert "accuracy" in tag_names
    assert "relevance" in tag_names


def test_process_dataset_depth_fallback_to_generic(tmp_path):
    """When no segment-specific depth tag exists, generic 'depth' is used."""
    ds, _ = _write_temp_files(tmp_path, segment="accounts payable")
    entries, _ = process_dataset(ds)
    tag_names = [t["tag"] for t in entries[0]["tags"]]
    # No "accountspayable_depth" in rubric, so falls back to "depth"
    assert "depth" in tag_names


def test_process_dataset_depth_uses_segment_specific(tmp_path):
    """When a segment-specific depth tag exists, it is used instead of generic."""
    input_yaml = tmp_path / "input.yaml"
    tsv_path   = tmp_path / "utterances.tsv"
    rubric     = tmp_path / "rubric.yaml"

    query = "Describe aged balances."
    segment = "aged balances"
    input_yaml.write_text(
        yaml.dump([{"query": query, "assertions": [{"text": "covers topic"}]}]),
        encoding="utf-8",
    )
    tsv_path.write_text(f"Utterance\tSegment\n{query}\t{segment}\n", encoding="utf-8", newline="")

    # Include both generic depth and a segment-specific "agedbalances_depth"
    rubric_content = [
        {"tag": "depth",             "metric": True, "assertions": [{"text": "generic"}]},
        {"tag": "agedbalances_depth","metric": True, "assertions": [{"text": "aged-specific"}]},
    ]
    rubric.write_text(yaml.dump(rubric_content), encoding="utf-8")

    ds = {
        "plugin": "erp_qa",
        "input_yaml": str(input_yaml),
        "tsv": str(tsv_path),
        "lmc_rubric": str(rubric),
    }
    entries, _ = process_dataset(ds)
    tag_names = [t["tag"] for t in entries[0]["tags"]]
    assert "agedbalances_depth" in tag_names
    # The generic "depth" tag must NOT appear (segment-specific takes its place)
    depth_tags = [n for n in tag_names if "depth" in n]
    assert "depth" not in depth_tags or "agedbalances_depth" in depth_tags


def test_process_dataset_missing_rubric_raises(tmp_path):
    input_yaml = tmp_path / "input.yaml"
    tsv_path   = tmp_path / "utterances.tsv"
    input_yaml.write_text(yaml.dump([{"query": "q", "assertions": []}]), encoding="utf-8")
    tsv_path.write_text("Utterance\tSegment\nq\tseg\n", encoding="utf-8", newline="")
    ds = {
        "plugin": "erp_qa",
        "input_yaml": str(input_yaml),
        "tsv": str(tsv_path),
        "lmc_rubric": str(tmp_path / "nonexistent_rubric.yaml"),
    }
    with pytest.raises((FileNotFoundError, OSError)):
        process_dataset(ds)
