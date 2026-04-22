import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference.dataloader import load_questions

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "dataset_fixture.yaml"


def test_load_returns_list():
    questions = load_questions(_FIXTURE)
    assert isinstance(questions, list)
    assert len(questions) > 0


def test_query_renamed_to_question():
    questions = load_questions(_FIXTURE)
    for q in questions:
        assert "question" in q
        assert "query" not in q


def test_other_fields_preserved():
    questions = load_questions(_FIXTURE)
    for q in questions:
        assert "plugin" in q
        assert "segment" in q
        assert "tags" in q


def test_shuffle_returns_same_entries():
    original = load_questions(_FIXTURE, shuffle=False)
    shuffled = load_questions(_FIXTURE, shuffle=True)
    assert len(original) == len(shuffled)
    orig_questions = sorted(q["question"] for q in original)
    shuf_questions = sorted(q["question"] for q in shuffled)
    assert orig_questions == shuf_questions


def test_shuffle_produces_different_order_sometimes():
    # Run shuffle many times; assert at least one result differs from original order
    original = load_questions(_FIXTURE, shuffle=False)
    original_order = [q["question"] for q in original]
    found_different = False
    for _ in range(20):
        shuffled = load_questions(_FIXTURE, shuffle=True)
        if [q["question"] for q in shuffled] != original_order:
            found_different = True
            break
    assert found_different, "shuffle=True never produced a different order over 20 tries"


def test_nonexistent_file_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        load_questions("/nonexistent/path/dataset.yaml")
