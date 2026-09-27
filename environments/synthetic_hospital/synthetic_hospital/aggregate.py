"""Recompute exact benchmark metrics from saved Verifiers results.

Corpus-level metrics are not generally means of rollout rewards. Each rollout
replicate must contain each requested example exactly once; partial results need
an explicit --allow-partial flag and are labelled incomplete.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import canonical_split, load_rows
from .scoring import answer_text, parse, score_predictions


def aggregate(records, task="all", split="public", allow_partial=False):
    expected = {r["example_id"]: r for r in load_rows(task, split)}
    by_id = defaultdict(list)
    for record in records:
        identifier = record.get("example_id", record.get("gt_id"))
        if identifier is None:
            raise ValueError("Every result must have example_id or gt_id")
        identifier = int(identifier)
        if identifier not in expected:
            raise ValueError(f"Unknown example {identifier} for {task}/{split}")
        # Resolve references from the frozen dataset, never from model output or
        # editable result-file info. Completions, including empty ones, are required.
        if "completion" not in record and "raw_output" not in record:
            raise ValueError(f"Missing completion for {identifier}")
        by_id[identifier].append(record)
    if not by_id:
        raise ValueError("No results")
    counts = {len(v) for v in by_id.values()}
    if len(counts) != 1:
        raise ValueError("Unequal rollout counts: cannot aggregate incomplete replicates")
    missing = sorted(set(expected) - set(by_id))
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} expected examples; use --allow-partial only for smoke tests")
    reports = []
    for replicate in range(next(iter(counts))):
        groups = defaultdict(list)
        for identifier in sorted(by_id):
            row = expected[identifier]
            result = by_id[identifier][replicate]
            text = answer_text(result.get("completion", result.get("raw_output", "")))
            groups[row["task"]].append((parse(row["task"], text), row["info"]))
        reports.append(
            {
                name: score_predictions(name, [p for p, _ in items], [i for _, i in items])
                for name, items in groups.items()
            }
        )
    return {
        "split": canonical_split(split),
        "complete": not missing,
        "expected_examples": len(expected),
        "scored_examples": len(by_id),
        "missing_example_ids": missing,
        "rollouts_per_example": next(iter(counts)),
        "replicates": reports,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--task", default="all")
    p.add_argument("--split", default="public")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with args.results.open() as f:
        report = aggregate([json.loads(line) for line in f if line.strip()], args.task, args.split, args.allow_partial)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
