# Upstream benchmark contracts

## Ownership and originals

`benchmarks/upstream/` holds unmodified original datasets, model cards, scoring
sources and upstream manifests. `manifest.json` and `*-manifest.json` pin source
revisions, URLs, sizes, Git blob/LFS identities and verified local SHA-256.
`scripts/fetch_benchmarks.py`, `fetch_general_benchmarks.py` and `fetch_ifeval.py`
perform acquisition without executing downloaded programs.
`restore_benchmarks.py` restores saved manifest revisions and safely extracts the
verified tokenizer corpus without selecting new upstream heads. Existing originals
are verified, not replaced silently when upstream revisions change.

`decision_lab/benchmarks.py` owns source adapters. Each case retains its original
row, source file hash and row index. Model input is exclusively state/questions;
labels, soft gold, factors, agreement and provenance are excluded. Original
text, question descriptors, option IDs and order survive adaptation. JSON-valued
criteria are rendered only when a backend requires strings. SNLI/MNLI dataset
label indices are checked against Parquet metadata; model indices are mapped by
config names. SNLI's unannotated -1 rows are excluded and counted separately.

`benchmarks/inventory.json` is machine-readable validation. The human source,
license, compatibility and protocol inventory is `docs/benchmark-inventory.md`.
Full readiness means this lab can run every available record in the specified
suite, recording capacity and schema failures. It does not mean replication of
an entire publisher leaderboard, private tests, quantization, hardware or cost.

## Scoring

`vendored/jevbench/scoring.py` and `metrics.py` are the exact pinned upstream
scorers. Distribution keys, finite values, bounds and sum are validated; the
upstream rounding allowance and lexical tie rule are preserved. Ordinal
accuracy uses the winning level; expected-value MAE is separate. The workstation
suite's 0.75 tolerance is never substituted for upstream ordinal accuracy.
Malformed responses remain incorrect in the denominator. Unknown targets are
excluded from accuracy, not assigned invented labels.

Label-only GGUF predictions do not get fabricated probability distributions,
Brier or ECE. NLI option probabilities are explicitly adaptations, rather than
native calibrated typed decisions. Typed Decisions retains original soft gold
and reports KL and sum-squared error against it separately from hard-target
accuracy/Brier and top-label ECE. This is agreement with a sampled teacher, not
independent human truth. Dataset-specific metric definitions remain documented.

`decision_lab/benchmark_diagnostics.py` computes exact vendored Kev metrics on
clean knowable original variants. Missing distributions or rejected fields
prevent an upstream-complete native result; failures are not silently removed.
Its selective-coverage estimates are descriptive on the scored sample and never
used as production authorization thresholds.

## Runtimes

`decision_lab/specialists.py` owns Kev, Laya and general Verdict adapters. Their
vendored kernels/formatting are versioned in `vendored/manifest.json`; licenses
stay alongside source. Verdict and IFEval have relative-import-only source
changes; other retained inference/math source is unmodified.

Kev loads its safe, weights-only pointer metadata, local Qwen causal text
backbone and local LoRA, merged in FP32 with the checkpoint's fitted temperature.
It uses strict admission and the upstream reference DeltaNet/conv path on this
SM75 GPU. Laya loads encoder config and complete safetensors with strict tensor
matching, original option markers, checkpoint token budgets and option-count
calibration; escalation policy is not dispatched. Verdict uses GLiClass plus
the original engine and shipped calibrator. Its abstention mass is retained
when the original benchmark includes that label; otherwise it is dropped and
renormalized as in JevBench's published adapter. The native abstention decision
and full mass remain in raw evidence. Candidate limits fail explicitly.

These adapters use dedicated PyTorch runtimes. Upstream llama.cpp merged native
decision support in PR 29818 on 2026-10-02; the historical chat-runtime image and chat
adapter predate it. Native Kev-4B Q4/Q8 downloads are verified in
`docs/downloads/kev-4b-gguf.json` and `kev-4b-q8.json`. The separate SM75 image
pins `a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd`; `llama_decision.py` and the
`kev-4b-q8`/`kev-4b-q4` upstream runner options use its native `/v1/systemone`
distributions. `scripts/test_kev_cpp.py` runs the fixed 64/228 matched cases
with snapshots, GPU proof and sampled PID VRAM. The validated baseline
`configs/kev-cpp.json` uses F16 cache, batch 1024, one slot and context 4096.
Native option markers and pointer must fit one logical batch; batch 512
rejected a 751-token option span, batch 1024 passed the entire sample.
`results/kev-cpp-ab-v2/` holds corrected Q8/Q4 results; the initial batch512
run remains in v1. `results/kev-cpp-parallel-v1/` records a four-slot/context
16384 trial separately; all request errors are zero in corrected runs.
The original defaults and saved cloud results remain unchanged.
`kev_models.py` also pins the native 9B Q4_K_M candidate at Hub revision
`ec2bbfe6620218aee2e01cc93bb78dee2a96ed58`, tracked in
`docs/downloads/kev-9b-q4.json`. The selected downloader verifies published
size and SHA-256 before inference; a partial download is not usable.
`test_kev_cpp.py --models 9b-q4` and the upstream `kev-9b-q4` backend reuse
the validated single-slot context/batch settings and native typed head path.
The test report includes saved 4B results for comparison; task inputs and gold
are unchanged. Acquisition status and actual results remain separate.
`results/kev-9b-q4-v1/` records a completed 64/228 run: 190/214 Odysseus
fields, 244/299 upstream fields, zero request errors and no cloud calls. Its
verification checks matched input/gold content, zero generated tokens and
vendored source hashes. The tested candidate is `configs/kev-9b-cpp.json`;
the original 4B and historical defaults are preserved. Task-owned containers
are stopped after measurement; the model and image remain available.
`results/kev-memory-profile-v1/memory.json` records a fresh
64-case FP32 0.8B CUDA memory measurement, distinct from prior timing evidence.
GGUFs still use the
shared manager, verified model hashes, owned containers and actual CUDA proof.
`demo.py --backend` exposes the dedicated adapters without changing defaults.

## Runs and deployment

`scripts/upstream_benchmarks.py inventory` verifies original hashes and labels.
`run` selects a deterministic round-robin hash sample by original stratum before
predictions; the default is 12 cases per suite. `--limit 0` explicitly runs full
selected suites. `--case-file` admits validated derived fixtures separately.
Fresh output directories, input/config snapshots, implementation copies, raw
responses, timing, errors and summaries are required. No cloud client is used.

`scripts/general_benchmarks.py` owns free-text IFEval/GSM8K. IFEval preserves
original prompts/kwargs and strict/loose Google validators, with local pinned
Punkt tables and deterministic language detection. All 541 validators undergo
resource preflight before inference. GSM8K preserves numeric gold, takes five
original training demonstrations at seed 42 and applies documented #### numeric
extraction. Chat formatting, seed, token limits and truncation are recorded;
this is not a claim of publisher lighteval protocol identity. `--limit 0` selects
the full original split. Generation models alone support these tasks.

`scripts/benchmark_diagnostics.py` prepares original/reversed-option pairs.
Derived probes preserve state, descriptions and semantic targets; they are never
mixed into headline original-suite scores. Repeated local runs use new output
paths. Saved Jev reference responses are read offline; new API sets are outside
this workflow.

## Verification

Unit checks cover original scorer parity, malformed distributions, failure
denominators, no invented calibration, soft-gold KL/Brier, label-index mapping,
gold separation, deterministic sampling, scalar expected scores, exact Kev
metric reuse, semantic preservation under option permutation, original IFEval
keyword validation and numeric extraction. Live evidence covers seven backends,
all three public JevBench tiers, original NLI pairs, typed soft gold, Verdict
abstention/cardinality, ARC, IFEval/GSM8K and option-order pairs. Dedicated models
and the larger NLI additionally completed the unchanged 64-case Odysseus set.
