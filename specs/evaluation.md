# Decision evaluation contracts

## Ownership

- `decision_lab/http.py` owns credential-bearing transport, bounded timeout,
  sanitized transport errors, rejection of redirects and URL credentials,
  HTTPS for credential-bearing non-loopback endpoints, and no request retries.
- `contracts.py` owns typed descriptors, scalar answer validation and normalized
  legacy response objects. Local scores are discrete; Jev and adapted NLI scores
  may be fractional. Boolean Python values cannot masquerade as numbers.
- `llama.py` owns prompt variants and JSON generation. The pinned server's tested
  format is `response_format.json_schema.schema`. No hand-written ChatML wrapper
  is used. Templates are served by llama.cpp and checked against official model
  tokenizer templates. Direct variants convert yes/no labels to point decisions;
  local numeric estimates are not calibrated probabilities.
- `jev.py` sends one request to the official TypeSafe endpoint per evaluation.
  Binary values come from `answers[id].noul`, not `.probability`. Raw responses
  preserve distributions, confidence, model identity and token usage.
- `llama_decision.py` calls only a loopback native `/v1/systemone` endpoint.
  It validates typed probability keys, finite normalized probabilities and zero
  generated tokens, retaining the complete response. Choice/score/noul scalar
  conversion uses the same native distribution contract as specialist adapters.
  `scripts/test_kev_cpp.py` verifies downloaded GGUF hashes, runs the frozen 64
  Odysseus cases and saved upstream smoke cases, and samples actual server PID
  memory. Its separate image is pinned to a decision-capable llama.cpp revision;
  `ManagedServer(image=...)` selects it without changing the existing default.
- `kev_models.py` owns pinned 4B Q8/Q4 and 9B Q4 GGUF identities, labels and
  verified manifest lookup. `scripts/download_kev_gguf.py` acquires only the
  selected public files and records published size/hash verification. The
  native test runner accepts `--models 9b-q4` without changing the 4B aliases
  or the original default backend selection.
- `nli.py` loads local safetensors/tokenizers only, resolves entailment and
  contradiction indices from the model config, and uses eval/inference mode.
  Candidate-label distributions and rubric expectations are zero-shot adaptations,
  not native Jev behavior. CUDA timing includes synchronization through scalar
  extraction or CPU copies. Warm-up is excluded.
- `runtime.py` launches through `~/scripts/llm_manager.sh`, never a separate raw
  container launch. Names and ports are unique per invocation. Readiness checks
  this container's existence, served model identity, health and the actual server
  PID's CUDA allocation. Missing manager/Podman/NVIDIA executables fail with a dependency-guide pointer
  before run artifacts are created. Startup offload logs are saved when available; their
  absence is not treated as CPU fallback.
  Shutdown stops the exact container before terminating the manager group and
  checks that exact name again. Credentials are removed from manager environment.
- `scripts/evaluate.py` owns runs and frozen selections; `report.py` works only
  from saved files and exports `configs/local-best.json`. `settings.py` selects a
  profile by served GGUF filename. `start_local.py` verifies the saved weight hash
  before executing the manager with selected cache settings.

## Evaluation protocol

`benchmarks/workstation-v1.json` contains 24 synthetic cases, four per gate.
Each gate contributes two development and two held-out cases. The suite carries
human expectations; Jev is a comparator rather than ground truth. Binary answers
use threshold 0.5 unless an NLI threshold is selected on development cases.
Ordered scores allow absolute error at most 0.75. Wrong or invalid responses count
as failures. Exact-case and per-field accuracy are both reported; unauthorized
safety approvals are counted separately.

Local selection uses development field accuracy, then fewer unsafe approvals,
then exact cases, then median latency. Demonstrations contain development cases
from the same gate and exclude the scored case. Sampling uses fixed seed 42;
only one seed was tested. Cache precision is compared with the same selected
prompt/sampling configuration. The winner is written before held-out inference.
Earlier held-out Qwen aggregates were exposed during debugging; this is an
iterative development comparison, not an untouched external validation set.

Local results can resume explicit local runs by reading saved case outputs.
Resumption checks suite hash and saved case content. A changed implementation or
profile requires a new run directory; the existing outputs are retained as
historical evidence, not regenerated. Current resumption assumes the caller has
kept configuration unchanged; it does not verify source hashes automatically.

Jev has a separate exclusive attempt file. Existing attempts, including crashes,
forbid rerunning that output directory. Requests are sequential, never retried,
and stop at the first error. The requested reference set is exactly one pass;
its completed responses must be reused for reports and local tuning.

## Frozen-settings expanded evaluation

`benchmarks/odysseus-v2.json` contains exactly 64 scored cases: the 12 original
holdout cases unchanged and 52 authored Odysseus-related cases. The original
12 development cases remain separate. `scripts/build_odysseus_suite.py` owns
the reproducible fixture definitions and audit rationales. Existing question
descriptors are reused; new labels were assigned before inference.

`scripts/retest.py` validates preservation, IDs, labels and demonstration
separation before launching. It snapshots the suite and saved configuration,
checks GGUF hashes, and runs each downloaded backend with the previously
selected prompt/cache/NLI threshold settings. It does not tune on the expanded
suite, overwrite `configs/local-best.json`, or initialize a cloud client.
Per-model raw outputs, summaries for all 64/original 12/new 52, field failures,
runtime smokes, image/source/weight hashes and actual CUDA allocation are saved.
Only content-matched original Jev responses are compared, read offline; there
is no Jev coverage for the new 52. Outputs require a fresh directory.

The domain context came from static reads of Odysseus agent/context specs;
no Odysseus repository code ran. This suite tests the lab's simplified six-gate
rubrics, not the full Odysseus tool policy, integration, or real GUI screenshots.
Labels are author-assigned synthetic expectations. Cases sharing a gate are
correlated; the larger count does not establish production reliability.

## Full 64-case Jev pass

The full Jev comparison covers all 64 cases, separately from the local retest.
`scripts/jev_retest.py run` owns this separate 64-request operation. Its
exclusive attempt file forbids another run in the same directory, requests
are reserved on disk before transport, and the first failure ends the pass
without retry. Only state and question descriptors enter API inputs; labels
and audit rationale are evidence metadata. Full responses, request bodies
without credentials, usage, resolved model identity and summaries are saved
under `results/jev-odysseus-64-v1/`.

The `report` operation is offline. It requires complete coverage and identical
ordered case contents between the new Jev pass and saved local predictions,
then applies unchanged scoring to all 64, original 12 and new 52. It leaves
the previous 24-request reference and the historical local report unchanged.
The 24-case reference and 64-case comparison are distinct recorded runs. A fresh
output path alone does not authorize another paid campaign or reset its limits.

## Boundaries and evidence

`.env` supports `JEV_API` and `TYPESAFE_API_KEY`. It is read as data, never sourced,
never printed, and ignored by Git. Environment variables take precedence. API
keys and Authorization headers are not serialized into results. The workstation gate fixtures are
synthetic. The separate private workspace replay may contain private inputs and
requires its own explicit scope; see `workspace_replay.md`. Neither its inputs
nor raw results are published.

Local adapters make no safety enforcement claim. The application executing an
actual action must own authorization. No demo action is dispatched. Schema
validity and model confidence do not establish task correctness.

Runs save raw requests/responses, per-field evaluations, server metadata and
GPU startup logs, weight checksums, and selected configurations. Jev usage and
resolved model ID are retained. Report generation makes no network calls.
Concurrent host jobs and unpooled cloud HTTPS make latency directional; small
synthetic results do not establish calibration or production reliability.

## Additional downloaded decision assets

`scripts/download_decision_models.py` owns acquisition of pinned Kev-0.8B,
Laya typed-decisions and public general-purpose Verdict downloads, including
Kev's pinned Qwen3.5-0.8B base. The initially selected specialized Verdict 2.0
weight object does not exist on its published Git LFS server (404); metadata
and inference source references are retained separately without claiming that
checkpoint is downloaded. `docs/downloads/decision-models.json` records public
source URLs, resolved revisions, file sizes/checksums and completion status.
Large Hub/Git LFS files must match published SHA-256 and size; small Hub files
must match their Git blob IDs. Reference licenses/source files are preserved
without installation or execution. Model directories remain ignored.

`docs/decision-models.md` owns license, cost and compatibility findings. The acquisition operation itself made no inference or cloud calls. These assets
are subsequently integrated through dedicated PyTorch adapters, documented in
`benchmarks.md`. They remain unsupported by the current llama.cpp chat server;
existing default configurations are unchanged.

## Verification

- `python -m unittest discover -s tests -v` checks wire adaptation, no cloud retry,
  .env handling, invalid answer rejection and the exclusive Jev attempt guard.
- `python -m unittest -v test_local_llama` checks localhost transport and legacy
  compatibility, invalid/truncated answers and archived mock score bounds.
- Live local preflight adapts upstream chat/schema/tokenizer smokes: health,
  `/v1/models`, `/props`, forced-schema marker, capital enum, Unicode tokenization
  and official template equivalence. The actual server PID must have CUDA allocation; offload startup logs are
  retained when available.
- Live saved results cover both downloaded GGUFs and the NLI model. Public model
  repository metadata verifies the GGUF weight hashes. No weights are redownloaded.

## Default provider identifier

New Jev clients default to `jev-1.14.0` (October 15, 2026). An explicit
`model` constructor argument supports separately frozen comparisons. Saved 1.13
responses, labels, thresholds and historical attempts remain unchanged.
