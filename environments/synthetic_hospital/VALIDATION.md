# Fresh-generation validation

Tested on 2026-09-27–28 from a fresh Medmarks clone and Python 3.12 virtual
environment. Base commit: `154710b74e01fb203b5ccf47312dc73d27cd04b4`.
Medmarks 0.2.0, Verifiers 0.1.14, Synthetic Hospital environment 0.2.0, vLLM 0.20.0.

## Workflow

Followed the repository setup: `uv venv --python 3.12`, `uv sync`, then
`uv run vf-install synthetic-hospital`. Both models ran the documented
`medarc-eval bench --config configs/synthetic-hospital.toml` commands from the
environment README, with the default Verifiers environment server enabled.
Each model used a fresh output directory and a newly launched vLLM server.
Cached checkpoint weights were reused; model completions were generated afresh.

- Two fresh smoke examples per task, then all 1,859 public cases per model.
- 1,869 successful live generation calls per model: 10 smoke + 1,859 full.
- Zero API errors; all expected IDs covered exactly once in each full run.
- All 1,859 prompt messages matched the original formatter outputs per model.
- All 1,859 input-token counts matched the previous run per model.
- The bundled corpus scorer matched all 60 returned metric/counter fields from
  an independent original-harness computation, including nested diagnosis metrics.
  Maximum numeric difference: **0 for both models**.
- 27 environment tests pass. Authored Python code passes Ruff.
- Standard `medarc-eval process` was also exercised on the new environment outputs.

## Scores versus the previous local vLLM runs

Fresh generation changes answers, so the model scores are not identical to the
previous runs. Scorer parity compares both implementations on the same new
answers; that comparison is exact. These are public-split corpus scores.

| Metric | Gemma previous | Gemma fresh | Kimi previous | Kimi fresh |
|---|---:|---:|---:|---:|
| Patient dx | 0.275202 | 0.277329 | 0.700438 | 0.691953 |
| Summarization | 0.383173 | 0.378903 | 0.454662 | 0.449322 |
| Specialty summarization | 0.379644 | 0.402653 | 0.576402 | 0.579872 |
| Retrieval P@5 | 0.889000 | 0.886000 | 0.851000 | 0.849000 |
| Retrieval NDCG@10 | 0.733212 | 0.729511 | 0.684694 | 0.670405 |
| Imaging | 0.411379 | 0.412286 | 0.498870 | 0.504797 |

Full precision metrics, sampling settings, checkpoint revisions and hardware:
[Gemma](validation/gemma-fresh.json), [Kimi](validation/kimi-fresh.json).

Diagnosis and specialty F1 require corpus aggregation; mean rollout rewards
are training diagnostics. Use `synthetic-hospital-score` as documented in the
[README](README.md). Summarization `clinical_f1` is finding recall.
Provider latency and cost are not part of scorer parity.

## Independent reference scoring

The audit imports the original Synthetic Hospital prompt formatters, parsers and
scorers, loads references from its database, and performs a real PostgreSQL JSONB
round trip for parsed predictions. It does not use the port’s frozen references
or scoring implementation. This checks the portable scorer’s JSONB key-order
compatibility for structured summaries, along with every other returned metric.

```bash
# Run with the original checkout’s dependencies and loaded benchmark database.
DATABASE_URL=postgresql://... /path/to/synthetic_hospital/.venv/bin/python \
  environments/synthetic_hospital/validation/score_original.py \
  --source /path/to/synthetic_hospital \
  --run-root outputs/synthetic-hospital/gemma --output original-scores.json
```

The helper requires full public coverage, verifies every prompt, and reports API
errors separately. Compare its metrics with `synthetic-hospital-score` on the
same completions. Raw model answers and local cluster logs are kept outside git.
