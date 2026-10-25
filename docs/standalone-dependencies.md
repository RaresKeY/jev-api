# Standalone and workspace dependencies

This repository contains the evaluation source, synthetic Odysseus fixtures,
public upstream data and preserved third-party notices. It does not ship API
keys, model weights, runtime images, experiment results, private replay inputs,
external checkouts or sibling projects. No project license has been assigned.

Use Python 3.11 or newer (asset verification uses `hashlib.file_digest`). The portable tests use the standard library:

```bash
python -m unittest discover -s tests -v
python -m unittest -v test_local_llama
```

Missing optional libraries and the absent workspace campaign integration are
reported as test skips. Private-cohort tests require the original frozen source,
private archive and explicit `JEV_TEST_PRIVATE_REPLAY=1`; they skip by default.
This does not certify a model, GPU, provider or historical replay.
Some HTTP contract tests bind only a loopback fixture server.

## Entry points

Each script's module docstring identifies its standalone dependencies. Only
argument-parsing entry points support `--help`; archive helpers without
an argument parser run when invoked. Importing a script does not authorize
inference or provision its dependencies.

| Scripts | Dependencies not supplied in a source checkout |
|---|---|
| `build_odysseus_suite.py` | None beyond Python and the included original fixtures. |
| `demo.py`, `smoke.py` | An explicitly selected running compatible server, local downloaded specialist weights and optional Python runtime, or a Jev API key for the explicit Jev backend. |
| `start_local.py`, `evaluate.py`, `retest.py`, `general_benchmarks.py`, `upstream_benchmarks.py`, `test_kev_cpp.py` | Relevant model weights; local runs launched here also need executable `~/scripts/llm_manager.sh`, Podman, the specified CUDA image and `nvidia-smi`. Specialist adapters need the optional packages. Reports/comparisons may also read ignored previous results. |
| `download_decision_models.py`, `download_kev_gguf.py`, `download_new_decisions.py`, `fetch_benchmarks.py`, `fetch_general_benchmarks.py`, `fetch_ifeval.py`, `restore_benchmarks.py` | Network access to the declared public sources, normal TLS trust and sufficient local storage. Downloads do not run inference. |
| `prepare_public_sources.py` | Git and network for acquisition; its later stages require generated local acquisition/static-review metadata and original checkouts. These stage receipts are not all shipped. |
| `public_trials.py` | Corpus extraction reads retained source metadata. Token estimation can omit unavailable historical timings. Frozen campaign planning/execution additionally requires local results, runtime identities, the existing campaign ledger and sibling `../jev-pipelines`; Mica adds the managed runner below. |
| `jev_retest.py` | An explicit Jev API key for `run`; ignored matching local evidence for a historical `report`. |
| `report.py`, `benchmark_report.py`, `benchmark_diagnostics.py` | Local results at the named/default paths; derived diagnostics also use the included original benchmark data. |
| `report_new_decisions.py` | Ignored comparison results, sibling `../jev-pipelines`, the original campaign ledger/authorization and Podman for cleanup checks. |
| `report_public_preparation.py`, `verify_public_preparation.py` | Local preparation plans, acquisition/cleanup receipts, forecast, original source bindings and the existing campaign ledger. These are archive checks, not fresh-checkout validation. |
| `prepare_mica_runtime.py`, `smoke_mica.py` | Verified Mica weights/tokenizer/codebook, the pinned llama.cpp image/libraries, generated `.local/mica-abi.json`, and required Python packages. Preparation uses Podman and the external managed runner. |
| `compare_new_decisions.py` | Ignored 228-case upstream cohort and weight/runtime evidence. Native runs need the shared manager; Mica needs `../tooling/godot-podman/bin/godot-podman` and the retained asset-pipeline image; Jev needs sibling `../jev-pipelines` and the original ledger. |
| `replay_workspace.py`, `replay_kev_capacity.py`, `replay_kev_capacity4096.py` | Private frozen `.work/` cohort, ignored prior results, model weights and external runtimes. Preparing a cohort also needs the original sibling projects. |
| `analyze_workspace_replay.py`, `verify_workspace_replay.py` | Private frozen cohort, ignored results, original campaign ledger and retained policy/source evidence; the final verification also needs sibling `../jev-pipelines` and Podman. |

`requirements-evaluation.txt` and `requirements-benchmarks.txt` record optional
packages used by historical runs. They are not a claim that those packages or a
compatible CUDA build are installed. The Mica image also supplies its tokenizer,
Jinja and native-library dependencies; neither requirements file provisions that
image. The scripts never install these dependencies automatically.

## Local artifacts and integrity

Root `results/`, `.work/`, `.local/`, `models/` and `.env` remain local and ignored.
Generated `benchmarks/public-20261015` acquisition/cleanup receipts, plans,
forecasts and execution-verification receipts are also ignored: they contain
machine locations, local accounting state and implementation bindings. Included
public source bytes, corpus hashes, scorers and upstream notices are preserved.
The preparation-report generator writes to ignored `results/`; it cannot
overwrite the curated public methodology summary with local accounting details.
The nested `retained-evidence/.../results/` contains upstream public evidence,
not this workstation's results, and remains included.

Links to excluded local artifacts are shown as code paths in public documents.
Their absence is expected. Historical receipts bind original source hashes;
sanitization changes source hashes, so historical execution plans cannot be
reused as approval for changed code. Frozen originals remain local. Create and
review a fresh plan for any future execution; never reset existing accounting.

Local-only `.env` is still sensitive in a ZIP or folder copy: Git ignore rules
define the committed tree, not a sanitized copy of the whole working directory.

The included HTTP helper enforces the credential boundaries described in
`specs/publication.md`. Workspace campaign integrations import their transport
from the unbundled `jev-pipelines` project; review that dependency separately
before using its live transport.
