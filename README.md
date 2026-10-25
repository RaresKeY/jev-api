# Decision Lab

A Python evaluation harness for typed decisions from Jev, native llama.cpp
endpoints and local specialist models. It compares model answers with frozen
fixtures and original public benchmark labels, retaining validation failures
and reporting per-field accuracy, exact-case accuracy and latency.

The synthetic workstation suite covers reply intent, tool selection, model
routing, authorization recommendations, completion and GUI alignment. The lab
prints and scores recommendations; it does not execute the proposed actions.

The [model testing report](docs/model-testing-report.md) describes the recorded
15 October 2026 comparison of Jev, Clef, Mica, Kev and other local candidates.
It separates benchmark accuracy from agreement with saved decisions and lists
the limits of the small, correlated cohorts.

## Start offline

Python 3.11 or newer is required. The portable tests need only the standard
library; optional model, benchmark and private-archive integrations skip when
their prerequisites are unavailable.

```bash
python -m unittest discover -s tests -v
python -m unittest -v test_local_llama
```

These checks make no provider calls and run no models. HTTP contract tests use
a loopback fixture server. To inspect the synthetic requests without inference:

```bash
python scripts/demo.py --dump-payloads
```

The included [Odysseus suite](benchmarks/odysseus-v2.json) has 64 scored cases:
12 original holdout cases and 52 authored scenarios. Development examples remain
separate. [Comparison protocol](docs/comparison-flow-20261015.md) explains the
label, scoring and experiment boundaries.

## Run decisions

With a compatible llama.cpp chat server already running:

```bash
python scripts/demo.py --llama-url http://127.0.0.1:8080 --interactive
```

The client validates answer keys, choices, finite numeric values and score
ranges. Invalid or incomplete responses raise errors. Native `/v1/systemone`
decision probabilities use a separate adapter from chat completion.

For an explicit Jev API demo, provide `JEV_API` or `TYPESAFE_API_KEY` in the
environment or a local `.env`, then run:

```bash
python scripts/demo.py --backend jev
```

This command sends fixture inputs to the provider and incurs API usage. It does
not replay saved answers. `.env.example` documents the supported variables;
`.env` is read as data and never executed. `LLAMA_BASE_URL`, `LLAMA_MODEL` and
optional `LLAMA_API_KEY` configure the chat-server client. Credential-bearing
requests require HTTPS except for loopback HTTP, and the included HTTP helper
rejects redirects.

Local specialist demos select `kev`, `laya`, `verdict`, `nli-small` or
`nli-large` with `--backend`. They need the appropriate downloaded weights and
optional packages. No model or runtime dependency is installed automatically.

## Evaluation and reproduction

Every script identifies its standalone dependencies in its module docstring.
Read the [dependency guide](docs/standalone-dependencies.md) before running an
experiment. Relevant constraints include:

- Model weights, CUDA images and prepared native libraries are external inputs.
- Managed GPU runs require the external shared manager, Podman and NVIDIA tools.
- Historical report regeneration requires the named local result directories.
- Workspace campaign/replay helpers also need unbundled sibling projects,
  original accounting state and, for some cohorts, private inputs.

The pinned public corpus is included with source hashes and scoring provenance.
See the [benchmark inventory](docs/benchmark-inventory.md),
[public benchmark preparation](docs/public-benchmark-preparation-20261015.md)
and [configuration notes](docs/config-research.md). Hardware, engine revisions,
quantization and scoring adaptations remain explicit in the historical report.

Paid comparison runners require their explicit execution controls. They preserve
existing attempt records, reserve input/output usage before dispatch and do not
retry automatically. Changed code requires a fresh reviewed plan; a new output
directory does not reset an existing campaign. The dated results are historical
measurements, not a statement of current provider pricing or model availability.

## Repository layout

| Path | Contents |
|---|---|
| `decision_lab/` | Adapters, contracts, evaluation, scoring and campaign helpers. |
| `scripts/` | Acquisition, evaluation, replay and reporting entry points. |
| `benchmarks/` | Synthetic fixtures, pinned public data and provenance. |
| `configs/` | Saved model settings and historical runtime profiles. |
| `docs/` | Methodology, aggregate reports and dependency guidance. |
| `specs/` | Current implementation contracts, indexed by [the specs map](specs/_readme.md). |
| `vendored/` | Preserved upstream code and notices. |
| `legacy/` | Earlier heuristic demo retained for reference. |

Credentials, model weights, root `results/`, private `.work/`, runtime `.local/`
and machine-specific preparation records remain local and ignored. Public
benchmark evidence nested below `benchmarks/` is retained. Git ignore rules do
not sanitize a copy of the entire working folder; the committed source tree
omits those local artifacts.

No project license is assigned. Preserve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and the upstream licenses accompanying included code and data.
