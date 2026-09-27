import json
import math

import pytest
import verifiers as vf
from synthetic_hospital import load_environment
from synthetic_hospital.aggregate import aggregate
from synthetic_hospital.data import TASKS, asset, canonical_split, load_rows, manifest
from synthetic_hospital.scoring import answer_text, score_completion, score_predictions


def test_split_membership_and_counts():
    m = manifest()
    patients = [set(m["patient_ids"][s]) for s in ["train_fit", "validation", "public", "heldout"]]
    assert [len(x) for x in patients] == [720, 80, 200, 268]
    assert all(not a & b for i, a in enumerate(patients) for b in patients[i + 1 :])
    assert [sum(m["counts"][s].values()) for s in ["train_fit", "validation", "public", "heldout"]] == [
        6871,
        748,
        1859,
        2536,
    ]
    for split in m["counts"]:
        rows = asset(split + ".json.gz")
        assert len({r["example_id"] for r in rows}) == len(rows)
        for task in TASKS:
            assert sum(r["task"] == task for r in rows) == m["counts"][split][task]
        assert {r["info"]["patient_id"] for r in rows} == set(m["patient_ids"][split])


@pytest.mark.parametrize("task", TASKS)
def test_loader_training_and_validation(task):
    env = load_environment(task=task)
    train = env.get_dataset()
    val = env.get_eval_dataset()
    assert len(train) == manifest()["counts"]["train_fit"][task]
    assert len(val) == manifest()["counts"]["public"][task]
    assert not set(train["example_id"]) & set(val["example_id"])
    assert json.loads(train[0]["info"])["split"] == "train_fit"
    assert json.loads(val[0]["info"])["split"] == "public"
    assert train[0]["answer"] == ""


def test_aliases_and_invalid_options():
    assert canonical_split("public") == "public"
    assert canonical_split("heldout") == "heldout"
    with pytest.raises(ValueError):
        load_environment(split="development")
    with pytest.raises(ValueError):
        load_environment(task="unknown")
    with pytest.raises(ValueError):
        load_environment(system_prompt="replace the benchmark")


def test_reasoning_is_not_scored():
    assert answer_text([{"role": "assistant", "content": "final", "reasoning": "secret"}]) == "final"
    assert (
        answer_text(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "final"}],
                }
            ]
        )
        == "final"
    )


@pytest.mark.parametrize("task", TASKS)
def test_bad_json_shapes_and_empty_completions(task):
    info = load_rows(task)[0]["info"]
    for text in [
        "[]",
        "null",
        "",
        '{"rankings": [null]}',
        '{"summary": ["bad shape"]}',
        '{"clinical_question": ["bad shape"]}',
        '{"differential": null}',
    ]:
        result = score_completion(text, info)
        assert math.isfinite(result["reward"]) and 0 <= result["reward"] <= 1


@pytest.mark.asyncio
async def test_real_verifiers_rubric_keeps_reference_out_of_prompt():
    env = load_environment(task="patient_diagnosis")
    row = env.get_eval_dataset()[0]
    info = json.loads(row["info"])
    # Perfect reference-format output gives full chart-neutral recall/precision.
    completion = [{"role": "assistant", "content": json.dumps(info["ground_truth"])}]
    state = {"prompt": row["prompt"], "completion": completion, "answer": "", "info": info}
    await env.rubric.score_rollout(state)
    assert 0 <= state["reward"] <= 1
    assert state["metrics"]["task_reward"] == state["reward"]
    assert state["metrics"]["parse_failures"] == 0
    assert all(set(message) == {"role", "content"} for message in row["prompt"])


def test_neutral_diagnosis_rule():
    info = {
        "task_name": "patient_diagnosis",
        "ground_truth": {
            "active_diagnoses": [{"icd10": "J189", "acuity": "acute"}],
            "chronic_conditions": [],
            "_neutral_categories": ["I10"],
        },
    }
    pred = {
        "active_diagnoses": [{"icd10": "J189", "acuity": "acute"}, {"icd10": "I10", "acuity": "chronic"}],
        "chronic_conditions": [],
    }
    m = score_predictions("patient_diagnosis", [pred], [info])
    assert m["weighted_problem_list_f1_neutral"] == 1
    assert m["weighted_problem_list_f1"] < 1


def test_aggregate_rejects_missing_foreign_and_uneven_records():
    row = load_rows("retrieval")[0]
    result = {"example_id": row["example_id"], "completion": ""}
    with pytest.raises(ValueError, match="Missing"):
        aggregate([result], task="retrieval")
    with pytest.raises(ValueError, match="Unknown"):
        aggregate([dict(result, example_id=-1)], task="retrieval")
    report = aggregate([result], task="retrieval", allow_partial=True)
    assert report["complete"] is False
    assert report["replicates"][0]["retrieval"]["precision_5"] == 0
    other = load_rows("retrieval")[1]
    with pytest.raises(ValueError, match="Unequal"):
        aggregate([result, result, dict(result, example_id=other["example_id"])], task="retrieval", allow_partial=True)


def test_specialty_absence_and_aggregate_is_not_mean_reward():
    rows = load_rows("specialty_summarization")
    # Include all cases to test the original denominators, including empty critical sets.
    infos = [r["info"] for r in rows]
    predictions = [{"summary": "No relevant specialty findings."} for _ in rows]
    report = score_predictions("specialty_summarization", predictions, infos)
    assert report["n_involved"] == 583 and report["n_absent"] == 400
    assert "abstention_accuracy" in report and "conditioned_f1" in report
    absent = next(i for i in infos if score_predictions("specialty_summarization", [{"summary": ""}], [i])["n_absent"])
    result = score_completion("No relevant specialty findings.", absent)
    assert result["reward"] == result["metrics"]["abstention_accuracy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS)
async def test_end_to_end_mocked_api_and_saved_result(task, tmp_path):
    """Exercise actual Verifiers generation, serialization and aggregate replay."""
    import httpx
    from openai import AsyncOpenAI
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

    env = vf.load_environment("synthetic-hospital", task=task)
    row = env.get_eval_dataset()[0]
    final = {
        "patient_diagnosis": '{"active_diagnoses": [], "chronic_conditions": []}',
        "summarization": '{"summary": "No relevant specialty findings."}',
        "specialty_summarization": '{"summary": "No relevant specialty findings."}',
        "retrieval": '{"rankings": []}',
        "imaging": '{"clinical_question": "Evaluate abdominal pain", "pre_read_summary": ""}',
    }[task]
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["messages"] == row["prompt"]
        assert "info" not in payload and "answer" not in payload
        return httpx.Response(
            200,
            json={
                "id": "mock-sh",
                "object": "chat.completion",
                "created": 1,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final,
                            "reasoning_content": "not part of the answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            },
        )

    sdk = AsyncOpenAI(
        api_key="unused",
        base_url="http://mock/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    client = OpenAIChatCompletionsClient(sdk)
    try:
        results = await env.evaluate(
            client=client, model="mock", num_examples=1, save_results=True, results_path=tmp_path / task
        )
    finally:
        await client.close()
    assert len(requests) == 1
    assert len(results["outputs"]) == 1
    record = json.loads((tmp_path / task / "results.jsonl").read_text().strip())
    assert record["example_id"] == row["example_id"]
    report = aggregate([record], task=task, allow_partial=True)
    assert report["replicates"][0][task]["parse_failures"] == 0
    assert report["scored_examples"] == 1


def test_specialty_harmonic_mean_must_be_computed_after_aggregation():
    from copy import deepcopy

    base = {
        "task_name": "specialty_summarization",
        "ehr_text": "hypertension diabetes",
        "ground_truth": {
            "variant": "specialty_conditioned",
            "involvement": "high",
            "tiers": {
                "primary": [{"display_name": "hypertension", "importance": "critical"}],
                "relevant": [],
                "excluded_sample": [{"display_name": "diabetes"}],
            },
        },
    }
    infos = [base, deepcopy(base)]
    preds = [{"summary": "hypertension"}, {"summary": "unrelated words"}]
    aggregate_f1 = score_predictions("specialty_summarization", preds, infos)["conditioned_f1"]
    mean_reward = (
        sum(score_predictions("specialty_summarization", [p], [i])["conditioned_f1"] for p, i in zip(preds, infos)) / 2
    )
    assert aggregate_f1 == pytest.approx(2 / 3)
    assert mean_reward == pytest.approx(1 / 2)


def test_derived_validation_is_from_training_only():
    import hashlib

    m = manifest()["patient_ids"]
    ranked = sorted(m["train"], key=lambda pid: hashlib.sha256(f"20260927:{pid}".encode()).hexdigest())
    assert set(m["validation"]) == set(ranked[:80])
    assert set(m["train_fit"]) | set(m["validation"]) == set(m["train"])
    assert not set(m["train_fit"]) & set(m["validation"])
    env = load_environment(split="validation")
    assert len(env.get_dataset()) == 720
    assert len(env.get_eval_dataset()) == 80
    with pytest.raises(ValueError, match="includes validation"):
        load_environment(split="validation", training_split="train")
    with pytest.raises(ValueError):
        load_environment(split="test")
    with pytest.raises(ValueError):
        load_environment(training_split="public")
    full = load_environment(training_split="train")
    assert len(full.get_dataset()) == 800
    assert len(full.get_eval_dataset()) == 200


def test_original_public_membership_is_preserved():
    rows = load_rows("all", "public")
    assert len(rows) == 1859
    assert all(r["info"]["source_split"] == "public" and r["info"]["split"] == "public" for r in rows)
    assert all(r["info"]["source_split"] == "train" for r in load_rows("all", "validation"))


def test_structured_summary_scoring_matches_persisted_jsonb_order():
    from synthetic_hospital._vendor.scoring import compute_all_metrics
    from synthetic_hospital.scoring import score_predictions

    # Original PostgreSQL JSONB sorts keys by UTF-8 byte length then bytes,
    # recursively. This matters when a structured summary is rendered as text.
    raw = {
        "summary": {
            "longer": "Known common words are here.",
            "a": "Known common words are here. Unknown long sentence with many foreign tokens.",
        }
    }
    persisted = {"summary": {"a": raw["summary"]["a"], "longer": raw["summary"]["longer"]}}
    info = {
        "task_name": "summarization",
        "ground_truth": {"must_include_findings": []},
        "ehr_text": "Known common words are here",
    }
    expected = compute_all_metrics(
        "context_summarization", [persisted], [info["ground_truth"]], ehr_texts=[info["ehr_text"]]
    )
    actual = score_predictions("summarization", [raw], [info])
    unnormalized = compute_all_metrics(
        "context_summarization", [raw], [info["ground_truth"]], ehr_texts=[info["ehr_text"]]
    )
    assert unnormalized["hallucination_rate"] != expected["hallucination_rate"]
    for name, value in expected.items():
        assert actual[name] == value


def test_diagnosis_f1_requires_corpus_aggregation():
    gt = {"active_diagnoses": [], "chronic_conditions": [{"icd10": "E11"}, {"icd10": "F32"}]}
    info = {"task_name": "patient_diagnosis", "ground_truth": gt}
    predictions = [
        {"chronic_conditions": [{"icd10": c} for c in ("E11", "F32", "G40", "J45")]},
        {"chronic_conditions": [{"icd10": "E11"}]},
    ]
    corpus = score_predictions("patient_diagnosis", predictions, [info, info])
    individual = [score_predictions("patient_diagnosis", [p], [info]) for p in predictions]
    key = "weighted_problem_list_f1_neutral"
    assert corpus[key] == pytest.approx(0.75)
    assert sum(m[key] for m in individual) / 2 == pytest.approx(2 / 3)
