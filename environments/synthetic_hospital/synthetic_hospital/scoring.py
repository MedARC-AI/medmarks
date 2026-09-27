"""Shared rollout reward and exact corpus-level evaluation boundary."""

import json
from functools import lru_cache

import verifiers as vf

from ._vendor import imaging, patient_diagnosis, retrieval, summarization
from ._vendor.imaging_concepts import ConceptExtractor
from ._vendor.scoring import compute_all_metrics
from .data import asset

SOURCE_TASK = {
    "patient_diagnosis": "patient_diagnosis",
    "summarization": "context_summarization",
    "specialty_summarization": "context_summarization",
    "retrieval": "evidence_retrieval",
    "imaging": "imaging_indication",
}
PRIMARY = {
    "patient_diagnosis": "weighted_problem_list_f1_neutral",
    "summarization": "clinical_f1",
    "specialty_summarization": "conditioned_f1",
    "retrieval": "precision_5",
    "imaging": "clinical_question_concept_f1",
}
PARSERS = {
    "patient_diagnosis": patient_diagnosis.parse_output,
    "summarization": summarization.parse_output,
    "specialty_summarization": summarization.parse_output,
    "retrieval": retrieval.parse_output,
    "imaging": imaging.parse_output,
}
PARSER = vf.Parser()


def decode_info(info):
    return json.loads(info) if isinstance(info, str) else info


def answer_text(completion):
    # Provider reasoning fields/blocks are not included in the scored final answer.
    return PARSER.parse_answer(completion) or ""


def parse(task, text):
    try:
        return PARSERS[task](text)
    except (AttributeError, TypeError, ValueError, KeyError, OverflowError) as exc:
        # Valid JSON with the wrong shape must receive a failed response, not
        # abort the evaluation or disappear from the denominator.
        pred = PARSERS[task]("")
        pred["parse_error"] = f"Invalid response shape: {type(exc).__name__}"
        return pred


@lru_cache(maxsize=1)
def concept_extractor():
    return ConceptExtractor(asset("imaging_inventory.json.gz"))


def _database_json_order(value):
    """Match PostgreSQL JSONB key order used by the original persisted predictions.

    Structured summary fallbacks stringify nested objects, so object iteration
    order can affect sentence boundaries and secondary text metrics.
    """
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: (len(key.encode("utf-8")), key.encode("utf-8")))
        return {key: _database_json_order(value[key]) for key in keys}
    if isinstance(value, list):
        return [_database_json_order(item) for item in value]
    return value


def score_predictions(task, predictions, infos):
    if len(predictions) != len(infos) or not predictions:
        raise ValueError("Provide equally sized, nonempty predictions and references")
    infos = [decode_info(i) for i in infos]
    if any(i["task_name"] != task for i in infos):
        raise ValueError("Cannot mix tasks in one aggregate")
    predictions = [_database_json_order(p) for p in predictions]
    kwargs = {}
    if task in ("summarization", "specialty_summarization"):
        kwargs["ehr_texts"] = [i["ehr_text"] for i in infos]
    if task == "imaging":
        kwargs["concept_extractor"] = concept_extractor()
    references = [i["ground_truth"] for i in infos]
    try:
        result = compute_all_metrics(SOURCE_TASK[task], predictions, references, **kwargs)
    except (AttributeError, TypeError, ValueError, KeyError, OverflowError):
        # Some malformed field types are harmless when their reference list is
        # empty. Preserve those original semantics; repair only examples that
        # actually raise in the original scorer, retaining them in the denominator.
        predictions = list(predictions)
        for index, (prediction, reference) in enumerate(zip(predictions, references)):
            one_kwargs = dict(kwargs)
            if "ehr_texts" in one_kwargs:
                one_kwargs["ehr_texts"] = [infos[index]["ehr_text"]]
            try:
                compute_all_metrics(SOURCE_TASK[task], [prediction], [reference], **one_kwargs)
            except (AttributeError, TypeError, ValueError, KeyError, OverflowError) as exc:
                predictions[index] = dict(
                    PARSERS[task](""), parse_error=f"Invalid response fields: {type(exc).__name__}"
                )
        result = compute_all_metrics(SOURCE_TASK[task], predictions, references, **kwargs)
    result["n_predictions"] = len(predictions)
    result["parse_failures"] = sum(
        bool(p.get("parse_error")) or p.get("_parse_tier") == "unparseable" for p in predictions
    )
    return result


def score_completion(completion, info):
    info = decode_info(info)
    task = info["task_name"]
    prediction = parse(task, answer_text(completion))
    metrics = score_predictions(task, [prediction], [info])
    # Absent specialties train abstention; their reward is never folded into
    # the paper's involved-only conditioned F1 in aggregate scoring.
    reward = metrics.get(PRIMARY[task], metrics.get("abstention_accuracy", 0.0))
    return {"prediction": prediction, "metrics": metrics, "reward": float(reward)}
