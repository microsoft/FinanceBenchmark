"""
dataloader.py

Unified loader for the benchmark dataset (YAML format).

Each entry in the YAML must have at minimum:
  - query:   the question text
  - segment: question category

Returns a normalized list[dict] with a "question" key so the orchestrator
and all inference scripts work uniformly.
"""

import yaml
from pathlib import Path
import random


def load_questions(path: str | Path, shuffle: bool = False) -> list[dict]:
    """
    Load benchmark questions from a YAML file.

    Args:
        path: Path to the YAML file.
        shuffle: Whether to shuffle the questions.

    Returns:
        List of dicts with at minimum {"question": str, "segment": str}.
        All other fields from the YAML entry are preserved as-is.
    """
    with open(path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    questions = []
    for entry in entries:
        row = {k: v for k, v in entry.items() if k != "query"}
        row["question"] = entry["query"]
        questions.append(row)

    if shuffle:
        random.shuffle(questions)

    return questions
