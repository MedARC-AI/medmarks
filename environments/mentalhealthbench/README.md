# MentalHealthBench

## Overview
- **Environment ID**: `mentalhealthbench`
- **Short description**: MentalHealthBench from OpenAI
([Malik et al., 2026](https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf),
[announcement](https://openai.com/index/introducing-mentalhealthbench/))

## Dataset
- **Source**: official OpenAI release `https://cdn.openai.com/ctf-cdn/OAI_MentalHealthBench.zip` (MIT license),
  downloaded and cached via `datasets`. The jsonl is checked against a pinned SHA-256 so upstream changes fail loudly.
- **Size**: 1,215 synthetic multi-turn conversations, 5,262 expert-authored rubric criteria (2–15 per task).
- **Metadata kept in `info`**: `id`, `acuity`, `user_profile`, `language`, `has_prior_context`, and per-criterion
  `criteria`, `points_list`, `axes` (one of 10 behavior axes).
- `canary_string` is dropped and never reaches the model or the grader. The dataset authors ask that examples are not
  published in plain text; keep tests and docs on synthetic data.

## Task
- **Type**: Single-turn completion of a multi-turn conversation prefix. Messages are sent as-is, including the
  dataset's own system message (age notice / prior user context) on 311 tasks; no extra system prompt is added.
- **Rubric**: each criterion is graded independently by an LLM judge with the paper's grader prompt (Appendix B.2),
  which returns `{"explanation", "satisfaction": 0|1}`. The judge sees the criterion text only, not its points.

## Scoring
Per response (paper Sec. 2.3.1), with all terms normalized by the task's total positive points:

| Name | Definition |
| --- | --- |
| `reward` | Task-clipped score `max(signed_score, 0)` — the headline metric |
| `signed_score` | `Σ points of met criteria / Σ positive points` (unclipped, can be negative) |
| `positive_points_earned` | Met positive points / total positive points |
| `normalized_penalty_burden` | Met negative points (absolute) / total positive points |

An empty model response scores 0 without being graded (paper B.1). With several judges, each criterion's
satisfaction is averaged across judges.

## Quickstart

```bash
medarc-eval mentalhealthbench -m gpt-4.1-mini -n 5 -r 1
```

`-r` (rollouts per example) defaults to 3 in `medarc-eval`; set it explicitly. The paper samples 4 responses per
task and averages the clipped scores within each task, then across tasks. With an equal number of rollouts per task
this equals the plain mean of `reward`, so the paper setting is just `-r 4` (4x the cost of `-r 1`). Extra samples do
not change the expected score, only its noise; the paper does not report a 1-vs-4 comparison.

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str \| list[str] | `"openai/gpt-5-mini"` | Judge model(s). |
| `judge_base_url` | str \| list[str] \| None | `None` | Judge endpoint base URL(s). |
| `judge_api_key` | str \| list[str] \| None | `None` | Judge API key(s). |
| `judge_timeout` | int \| None | `300` | Per-call judge timeout (s). |
| `max_parallel_judges` | int \| None | `3` | Concurrent criterion judgments per rollout. |
| `max_judge_retries` | int | `3` | Retries when a judge call fails or returns no valid 0/1 `satisfaction`; still invalid → not met. |

## Differences from the paper
OpenAI published the dataset, grader prompt and formulas but no evaluation code, so these choices are ours:

- **Conversation layout for the grader**: the paper does not specify how `{TASK_DEFINITION}` is rendered; we use
  `Role: content` blocks separated by blank lines.
- **Judge**: the paper uses GPT-5.6 Sol at high reasoning effort. The default here is `openai/gpt-5-mini`, and the
  shared judge defaults (`medarc_verifiers/utils/judge_helpers.py`) run GPT-5-family judges at `reasoning_effort="low"`
  and have no `gpt-5.6` entry yet, so scores are not directly comparable to the paper.
- **Breakdowns** by axis, acuity, user profile and language are not computed in the env; the needed fields are in
  `info` and per-criterion judgments in `info.judge_feedback`.
