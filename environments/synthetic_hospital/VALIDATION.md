# Validation record

Tested against Medmarks commit `154710b74e01fb203b5ccf47312dc73d27cd04b4`
and Verifiers 0.1.14, Python 3.12, on 2026-09-27.

Package version 0.2.0 preserves the original `train`, `public`, and `heldout`
partitions. An additional deterministic 80-patient validation subset is reserved
from `train`; the other 720 patients form the default training dataset `train_fit`.

The full saved Gemma 3 27B and Kimi K2.5 thinking public runs were replayed:

- 1,859 examples per model, all five tasks.
- All 55 reproducible aggregate metrics per model matched within 1e-12,
  including nested diagnosis metrics. Provider latency, token totals and
  latency-adjusted retrieval are excluded from scorer parity.
- All 1,859 parsed predictions per model matched the original parser outputs.
- All 1,859 Kimi request prompts matched the packaged prompts byte-for-byte.
  Gemma has saved final answers but no full request traces, so no equivalent
  request-byte comparison is claimed for Gemma.

| Primary metric | Gemma | Kimi |
|---|---:|---:|
| Diagnosis weighted chart-neutral F1 | 0.27520212336231814 | 0.7004381160820069 |
| Summarization finding recall | 0.383172643554416 | 0.454662003330347 |
| Specialty conditioned F1 | 0.37964391325846036 | 0.5764015811612511 |
| Retrieval P@5 | 0.889 | 0.851 |
| Retrieval NDCG@10 | 0.733211689470937 | 0.6846936386165282 |
| Imaging concept F1 | 0.41137853135956925 | 0.49887004640317895 |

The full CLI replay script is `scripts/replay_medmarks.py`; it runs all five tasks
through `medarc-eval bench` against a local OpenAI-compatible response server,
saves Medmarks results, and applies `synthetic-hospital-score` aggregation. Its
machine-readable report is `comparison.json` in the requested output directory.
The additional `scripts/verify_medmarks_parity.py` checks scorers directly. Saved model runs are local
artifacts and are not required to install the environment or run its unit tests.

The Medmarks environment-specific CLI detects `--task`, `--split`, and
`--training-split`. The five-variant TOML evaluates `public` with separate output
identities. This is saved-response replay, not new stochastic model inference.

The full CLI check exposed a secondary specialty hallucination-rate discrepancy:
the original runner stores parsed predictions in PostgreSQL JSONB before scoring,
which reorders nested object keys. Structured summary fallback rendering depends
on that order. The portable scorer now reproduces JSONB ordering without a database;
all 3,718 parsed predictions match their persisted serialization exactly. A regression
test covers the resulting text-metric difference.

The package test suite has 26 tests, including five actual Verifiers generation
and saved-result round trips using a mocked OpenAI-compatible transport. They
check payload isolation, original/derived split loading, patient disjointness and deterministic selection, malformed
outputs, chart-neutral scoring, coverage enforcement, and the specialty aggregate
harmonic-mean distinction. The existing repository evaluation suite has 33 passing
tests. No new paid model inference was required for these checks.
