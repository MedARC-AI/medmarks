"""Single-turn Prime Intellect / Medmarks loader."""

import verifiers as vf

from .data import TASKS, canonical_split, make_dataset
from .scoring import score_completion


def load_environment(
    task: str = "patient_diagnosis", split: str = "public", training_split: str = "train_fit", **kwargs
) -> vf.SingleTurnEnv:
    """Load Synthetic Hospital with a training dataset and selected evaluation split.

    Args:
        task: patient_diagnosis, summarization, specialty_summarization, retrieval,
            imaging, or all (mixed training; report evaluation tasks separately).
        split: Evaluation split: public (200 patients; paper benchmark, default),
            heldout (268; generalization), validation (80 reserved training patients),
            train_fit (720), or train (the original full 800-patient training pool).
        training_split: Training dataset: train_fit (default, excludes validation)
            or train (all original training patients; overlaps derived validation).
    """
    if task not in (*TASKS, "all"):
        raise ValueError(f"Unknown task: {task}")
    split = canonical_split(split)
    if training_split not in ("train_fit", "train"):
        raise ValueError("training_split must be train_fit or train")
    if split == "validation" and training_split == "train":
        raise ValueError("Use training_split=train_fit when evaluating validation; train includes validation patients")
    if any(k in kwargs for k in ("dataset", "eval_dataset", "system_prompt", "rubric", "parser")):
        raise ValueError("Dataset, prompts, parser and rubric are fixed by the benchmark")

    def cached(completion, info, state):
        if "_synthetic_hospital_score" not in state:
            state["_synthetic_hospital_score"] = score_completion(completion, info)
        return state["_synthetic_hospital_score"]

    async def task_reward(completion, info, state, **_):
        return cached(completion, info, state)["reward"]

    rubric = vf.Rubric(funcs=[task_reward], weights=[1.0])

    # Diagnostics have zero weight. Exact corpus metrics use synthetic-hospital-score.
    def metric_func(name):
        async def metric(completion, info, state, **_):
            return float(cached(completion, info, state)["metrics"].get(name, 0.0))

        metric.__name__ = name
        return metric

    for name in [
        "parse_failures",
        "weighted_problem_list_f1_neutral",
        "clinical_f1",
        "conditioned_f1",
        "abstention_accuracy",
        "leakage_rate",
        "precision_5",
        "ndcg_10",
        "clinical_question_concept_f1",
    ]:
        rubric.add_metric(metric_func(name))
    return vf.SingleTurnEnv(
        dataset=lambda: make_dataset(task, training_split),
        eval_dataset=lambda: make_dataset(task, split),
        rubric=rubric,
        **kwargs,
    )
