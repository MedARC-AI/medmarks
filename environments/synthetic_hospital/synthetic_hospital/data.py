"""Frozen public release data; no DB, credentials, downloads, or split generation."""

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")
TASKS = ("patient_diagnosis", "summarization", "specialty_summarization", "retrieval", "imaging")
ALIASES = {"val": "validation"}
SPLITS = ("train", "train_fit", "validation", "public", "heldout")


def canonical_split(split):
    split = ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError(
            f"Unknown split {split!r}; choose {SPLITS}. Use public for paper scores or heldout for generalization."
        )
    return split


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ASSETS / "manifest.json").read_text())


@lru_cache(maxsize=7)
def asset(name):
    data = (ASSETS / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest()["files"][name]:
        raise ValueError(f"Frozen asset checksum mismatch: {name}")
    return json.loads(gzip.decompress(data))


def load_rows(task="patient_diagnosis", split="public"):
    if task not in (*TASKS, "all"):
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS} or all")
    rows = asset(f"{canonical_split(split)}.json.gz")
    # Return detached records so callers cannot mutate the process-wide asset cache.
    return [json.loads(json.dumps(row)) for row in rows if task == "all" or row["task"] == task]


def make_dataset(task, split):
    from datasets import Dataset

    rows = load_rows(task, split)
    # JSON avoids Arrow coercing heterogeneous scoring references into a union
    # struct with null fields, which changes the original scorer's get() semantics.
    for row in rows:
        # Verifiers 0.1.14 reserves `task` for structured task payloads.
        # Our benchmark variant remains in info.task_name, not a routing column.
        del row["task"]
        row["info"] = json.dumps(row["info"], ensure_ascii=False)
    return Dataset.from_list(rows)
