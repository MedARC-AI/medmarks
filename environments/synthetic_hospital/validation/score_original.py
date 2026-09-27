"""Independent scoring with the original checkout, live DB loaders, and PostgreSQL JSONB."""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--run-root", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--variant", action="append")
args = ap.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from eval.chart_neutral import neutral_sets
from eval.config import get_pg_connection
from eval.imaging_concepts import ConceptExtractor
from eval.scoring import compute_all_metrics
from eval.tasks import (
    TASK_FORMATTERS,
    TASK_LOADERS,
    TASK_PARSERS,
)

variants = {
    "patient-diagnosis": ("patient_diagnosis", "cot", None),
    "summarization": ("context_summarization", "structured", None),
    "specialty-summarization": ("context_summarization", "zero_shot", "specialty"),
    "retrieval": ("evidence_retrieval", "zero_shot", None),
    "imaging": ("imaging_indication", "few_shot", None),
}
report = {}
with get_pg_connection() as conn:
    neutrals = neutral_sets(conn)
    extractor = ConceptExtractor.from_db(conn)
    for variant, (task, strategy, granularity) in variants.items():
        if args.variant and variant not in args.variant:
            continue
        files = list(args.run_root.glob(f"*/synthetic-hospital/{variant}/results.jsonl"))
        assert len(files) == 1, files
        records = sorted(map(json.loads, files[0].read_text().splitlines()), key=lambda r: r["example_id"])
        inputs = {x.gt_id: x for x in TASK_LOADERS[task](conn, split="public", granularity=granularity)}
        assert len(records) == len(inputs) and {r["example_id"] for r in records} == set(inputs)
        predictions = []
        references = []
        ehr = []
        parse_failures = 0
        errors = 0
        for row in records:
            inp = inputs[row["example_id"]]
            system, user = TASK_FORMATTERS[task](inp, strategy)
            assert row["prompt"] == [{"role": "system", "content": system}, {"role": "user", "content": user}], (
                variant,
                row["example_id"],
                "prompt differs",
            )
            final = [m for m in row["completion"] if m.get("role") == "assistant"]
            text = (final[-1].get("content") or "") if final else ""
            assert isinstance(text, str)
            pred = TASK_PARSERS[task](text)
            # Match the original runner's actual persistence boundary, independently of the port.
            with conn.cursor() as cur:
                cur.execute("SELECT %s::jsonb", (json.dumps(pred),))
                pred = cur.fetchone()[0]
            predictions.append(pred)
            parse_failures += bool(pred.get("parse_error")) or pred.get("_parse_tier") == "unparseable"
            errors += bool(row.get("error"))
            gt = dict(inp.ground_truth)
            if task == "patient_diagnosis":
                gt["_neutral_categories"] = sorted(neutrals[inp.patient_id])
            if task == "evidence_retrieval":
                gt["_judgments"] = inp.judgments
            references.append(gt)
            if task == "context_summarization":
                ehr.append(inp.ehr_text)
        kwargs = {}
        if ehr:
            kwargs["ehr_texts"] = ehr
        if task == "imaging_indication":
            kwargs["concept_extractor"] = extractor
        metrics = compute_all_metrics(task, predictions, references, **kwargs)
        metrics.update(n_predictions=len(predictions), parse_failures=parse_failures)
        report[variant] = {"metrics": metrics, "exact_prompts": len(records), "api_errors": errors}
        print(variant, len(records), "original scorer complete", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
