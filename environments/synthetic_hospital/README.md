# Synthetic Hospital for Medmarks / Prime Intellect

Five single-turn clinical tasks from Synthetic Hospital v1.3, using its frozen
prompts, parsers and deterministic scorers. No judge model, database, simulator,
GPU, API key or network connection is needed to load data or score completions.
Model inference uses the endpoint configured in Verifiers or Medmarks.

## Install and run

From a fresh Medmarks checkout, follow the repository setup and install this
local environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv run vf-install synthetic-hospital
```

Run only Synthetic Hospital against an OpenAI-compatible endpoint:

```bash
export VLLM_API_KEY=local-key
uv run medarc-eval bench --config configs/synthetic-hospital.toml \
  --model google/gemma-3-27b-it --provider local \
  --api-base-url http://HOST:8000/v1 --api-key-var VLLM_API_KEY \
  --max-concurrent 16 --timeout 600 --output-dir outputs/synthetic-hospital/gemma
```

The config runs all 1,859 public cases across five task variants, one rollout per
case, with temperature 0 and a 16,384-token output budget. It selects no other
Medmarks benchmarks. Use a new output directory for fresh generation: Medmarks
resumes existing results automatically.

For Kimi K2.5 thinking, use its endpoint and override sampling:

```bash
uv run medarc-eval bench --config configs/synthetic-hospital.toml \
  --model moonshotai/Kimi-K2.5 --provider local \
  --api-base-url http://HOST:8000/v1 --api-key-var VLLM_API_KEY \
  --max-concurrent 32 --timeout 1800 --output-dir outputs/synthetic-hospital/kimi \
  --sampling-args '{"temperature":1.0,"top_p":0.95,"seed":42,"extra_body":{"chat_template_kwargs":{"thinking":true}}}'
```

To smoke-test one task first, use `uv run medarc-eval synthetic-hospital
--task patient_diagnosis --split public -m MODEL -n 5` with the same endpoint
options. Publishing to the Hub is not required for local use.

```python
from synthetic_hospital import load_environment

env = load_environment(task="specialty_summarization")  # public evaluation
train = env.get_dataset()       # train_fit: 720 patients
public = env.get_eval_dataset()
validation = load_environment(task="specialty_summarization", split="validation").get_eval_dataset()
heldout = load_environment(task="specialty_summarization", split="heldout").get_eval_dataset()
```

`task="all"` enables mixed-task training. Evaluate/report tasks separately; an
average over all 1,859 cases would overweight specialty summarization.

## Frozen splits

The original release names and patient membership are preserved. `public` is the
200-patient benchmark used for the paper's leaderboard and the default evaluation
split. `heldout` is the separate generalization partition. The paper defines no
validation partition. There is deliberately no ambiguous `test` alias.

For tuning, this port reserves 80 patients from the original `train` pool as
`validation` (also `val`). The remaining 720 form `train_fit`, the default training
dataset. Selection uses the lowest SHA-256 hashes of `20260927:patient_id`; all
tasks for a patient stay together. These derived partitions are disjoint from each
other, `public`, and `heldout`.

| Task | train (800) | train_fit (720) | validation (80) | public (200) | heldout (268) |
|---|---:|---:|---:|---:|---:|
| patient_diagnosis | 800 | 720 | 80 | 200 | 268 |
| summarization | 800 | 720 | 80 | 200 | 268 |
| specialty_summarization | 4,037 | 3,640 | 397 | 983 | 1,325 |
| retrieval | 800 | 720 | 80 | 200 | 268 |
| imaging | 1,182 | 1,071 | 111 | 276 | 407 |
| Total | 7,619 | 6,871 | 748 | 1,859 | 2,536 |

`training_split="train"` explicitly selects the full original training pool; it
cannot be combined with `split="validation"` because those patients overlap.
`split` selects evaluation data independently of the training dataset.
The released heldout labels are packaged here, not served by a secret evaluation
service. Preserve that partition when training. Prompts retain the release's
frozen few-shot demonstrations and ontology hints verbatim; this port does not
introduce a newly decontaminated prompting protocol.

## Rewards and exact evaluation scores

Training rewards are per-example scalars in [0,1]: chart-neutral severity-weighted
F1 for diagnosis, finding recall for summarization, conditioned F1 for involved
specialties / abstention accuracy for absent specialties, P@5 for retrieval, and
concept F1 for imaging. Additional rubric metrics have zero reward weight.

**The average reward is not the paper's aggregate score.** Diagnosis F1 is the
harmonic mean of aggregate severity-weighted recall and chart-neutral precision.
The original diagnosis scorer excludes empty diagnosis lists from those means
and reports them through `n_tier_c`; rollout rewards for those cases are zero.
Specialty
conditioned F1 is a harmonic mean of aggregate critical-finding recall and
one minus aggregate leakage, over involved cases only. Absent cases are reported
separately. Use the exact aggregate scorer on Verifiers `results.jsonl`:

```bash
synthetic-hospital-score /path/to/results.jsonl \
  --task specialty_summarization --split public --output specialty-scores.json
```

For all five tasks from the Gemma command above, combine only that model's task
files and score the complete public suite:

```bash
cat outputs/synthetic-hospital/gemma/*/synthetic-hospital/*/results.jsonl \
  > outputs/synthetic-hospital/gemma/all-tasks.jsonl
uv run synthetic-hospital-score outputs/synthetic-hospital/gemma/all-tasks.jsonl \
  --task all --split public --output outputs/synthetic-hospital/gemma/scores.json
```

Use the corresponding Kimi directory for its scores. Do not combine different
models into one file. The standard `medarc-eval process` workflow still works,
but mean rollout rewards are training diagnostics, not the headline diagnosis or specialty F1.

It requires full coverage by default, rejects unknown IDs and unequal replicate
counts, and uses trusted packaged references. Use `--allow-partial` for smoke
runs only. Multiple rollouts are scored as separate complete replicates (in
per-example result order), never collapsed to a single selected answer. A single
file can contain all tasks with `--task all`. Scores include every deterministic
metric from the original harness. Hardware latency and API cost are not clinical
metrics and are not reproduced by this scorer.

Only final assistant content is scored, not separate reasoning fields. The
original permissive parsing fallbacks are preserved. Invalid JSON shapes are
reported as parse failures instead of aborting/dropping examples. Malformed output
may still receive partial credit under the original imaging fallback, as before.
The summarization metric named `clinical_f1` is finding recall, not conventional F1.
Predictions are normalized to PostgreSQL JSONB object-key order before scoring,
matching the original harness's storage round trip. This matters for structured
summary objects that the original scorer converts to text, including secondary
hallucination metrics; it requires no database at runtime.

## Data and provenance

`synthetic_hospital/assets/manifest.json` records split membership, counts, source
hashes and asset SHA-256 checksums (verified on load). Packaged gzip files contain
preformatted messages and private-to-the-scorer references, an imaging concept
inventory, and the CMS 2025 ICD-10 code/header index. References reside in dataset
`info`; the environment sends only `prompt` to the model. This is a single-turn
evaluation environment, not an isolated sandbox for untrusted tool agents.

`_vendor/` is a generated snapshot of the original MIT-licensed deterministic
scorers and parser definitions, with import namespaces rewritten. Source hashes
make updates auditable. The source export tool is `scripts/export_medmarks.py` in the upstream Synthetic
Hospital checkout; rebuilding requires its loaded database and ICD dictionary.
No export or database is needed to install or evaluate this Medmarks package.
Frozen prompt generation retains CoT diagnosis, structured summaries, zero-shot
specialty/retrieval, and few-shot imaging. A wheel works outside the source checkout.

Run the environment tests from Medmarks:

```bash
uv run pytest environments/synthetic_hospital/tests
```

See [VALIDATION.md](VALIDATION.md) for validation methodology and results. This
package implements five single-turn tasks; interactive simulator episodes are
not part of this version.
