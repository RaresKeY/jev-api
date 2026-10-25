# Downloaded Jev-style decision models

The recorded 15 October 2026 comparison covers Clef-Flash Q4, Mica 0.1 Q8,
Kev 4B Q8 and Jev 1.14. Its detailed run artifacts remain local at
`results/new-decisions-20261015-v1/`; the [aggregate report](model-testing-report.md)
is included. Acquisition and runtime observations below describe those dated
profiles, not the contents of a fresh checkout or current model availability.

Weights are external inputs under ignored `models/`. A directory containing
metadata alone is not sufficient for inference. The Clef/Mica
[acquisition manifest](downloads/new-decisions-20261015.json) and
[comparison protocol](comparison-flow-20261015.md) record source identities and
evaluation boundaries. Only Mica 0.1 was measured in this comparison.

Revisions, source URLs,
sizes and file checksums are recorded in
[downloads/decision-models.json](downloads/decision-models.json). That manifest
must say `complete` before the download is considered finished.

| Model | Selected assets | Free local usage | Source/weights | Existing llama.cpp chat server |
|---|---|---|---|---|
| Kev-0.8B | Kev 1.0 adapter, pointer head and tokenizer; pinned Qwen3.5-0.8B-Base weights and config | No per-request vendor fee | Apache-2.0 declared for code, adapter/head and base | Not directly compatible as a decision model |
| Laya | `convaiinnovations/laya-typed-decisions`, complete 421M checkpoint and tokenizer/config | No per-request vendor fee | Apache-2.0 declared | Not directly compatible as a decision model |
| openJev Verdict | General-purpose `heman10x/rlcd-modernbert-151m`, complete safetensors checkpoint, tokenizer/config and calibrator | No per-request vendor fee | Hub and repository declare Apache-2.0 | Not directly compatible as a decision model |

The specialized Verdict 2.0 checkpoint could not be obtained. Its GitHub media
URL returns 404, and the Git LFS batch API reports that the exact weight object
does not exist on the server. The advertised Hugging Face repository is
unavailable publicly, the author's public model listing contains only the
general-purpose model, and the GitHub repository has no releases. Downloaded
Verdict 2.0 files are metadata/source references only, not its weight checkpoint.
The manifest records the missing object's expected hash and 598,509,338-byte size.
The downloaded general-purpose Verdict is a distinct fallback, not the
checkpoint behind the reported 77.1% typed-decisions result.

The completed acquisition contains **3,294,529,391 bytes (3.29 GB)** across 51
files, including license and reference documents. Kev plus its base takes about
1.84 GB, Laya about 846 MB, and general-purpose Verdict about 609 MB. The extra
Verdict 2.0 metadata/source references are about 3.62 MB and contain no weights.
[verification.json](downloads/verification.json) records file integrity and
safetensors structure checks; no inference was run.

## Usage and licenses

These are public downloadable assets, fetched without an account, API key or
paid inference endpoint. Running locally consumes your hardware and electricity;
cloud GPU hosting, if chosen later, is a separate paid service. The upstream
Apache-2.0 declarations permit personal and commercial use under the license's
conditions, including preserving notices when distributing copies. The report
does not establish licenses for every upstream training dataset or optional
runtime dependency.

License texts are retained with the downloaded assets. Verdict's repository
`LICENSE` is an abbreviated Apache-labeled text rather than the full canonical
Apache-2.0 text; it was preserved verbatim. The canonical terms are at the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0).

Primary sources:

- [Kev-0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b) and
  [Kev code](https://github.com/jaredpalmer/kev).
- [Qwen base model](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base).
- [Laya typed-decisions card](https://huggingface.co/convaiinnovations/laya-typed-decisions)
  and [Laya code](https://github.com/NandhaKishorM/laya). The downloaded runtime
  reference documents came from its `haddock-development/laya_new` fork used
  during the initial research; inference code has not been installed.
- [Verdict code, checkpoint provenance and license](https://github.com/Heman10x-NGU/openJev-verdict-2.0).

## Runtime compatibility

These checkpoints score typed question options using custom decision heads.
Kev uses a LoRA adapter and pointer head over a Qwen backbone; Laya and Verdict
use ModernBERT encoders with custom heads. Their documented inference paths
are dedicated Python/PyTorch implementations. The lab's original llama.cpp server and
GGUF chat client do not implement those heads, question isolation, calibration
or typed output adapters. Converting a backbone alone would not reproduce the
decision model. This limitation applies to the historical pinned chat-runtime profile.
Upstream llama.cpp merged native decision-head support on 2026-10-02 in
[PR 29818](https://github.com/ggml-org/llama.cpp/pull/29818), including Kev and
the `/v1/systemone` endpoint. The official
[Kev-4B GGUFs](https://huggingface.co/ggml-org/Kev-4B-GGUF) include Q4_K_M
(3,033,489,824 bytes) and Q8_0 (4,483,801,504 bytes). These are weight-file
sizes, not measured VRAM. Acquisition is recorded separately in
[downloads/kev-4b-gguf.json](downloads/kev-4b-gguf.json); its status must be
`verified` before use. Q8 acquisition is recorded in
[downloads/kev-4b-q8.json](downloads/kev-4b-q8.json). Both files are verified.
The new `decision_lab/llama_decision.py` adapter calls this endpoint using the
separate, tested `localhost/jev-kev-server:llama-cpp-a4cb4c6-cuda12.6.0-ubuntu22.04`
image through the shared manager. The original chat image remains the default.
The introducing merge commit is `a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd`,
recorded in [downloads/kev-llama-support.json](downloads/kev-llama-support.json).
The workstation's existing image builder accepts `LLAMA_CPP_REF`,
`CUDA_ARCHITECTURES=75`, `LLAMA_BUILD_UI=OFF` and a separate `IMAGE_NAME`;
build a separate Kev candidate image rather than replacing the validated chat
image. The new adapter retains native distributions and validates probability
keys, normalization and zero generated tokens. Native full GPU offload of
33/33 layers is verified against the actual server PID.

Q8/Q4 results (`../results/kev-cpp-ab-v2/report.md`, local only) cover 64 synthetic Odysseus
cases and 228 matched original smoke cases. Both score 185/214 (86.4%) on
Odysseus with one unsafe approval; Q8 scores 239/299 upstream fields and Q4
238/299, with zero request errors. Q8 process VRAM peaks at 5,136 MiB (5.02
GiB), Q4 at 3,756 MiB (3.67 GiB). Single-slot median/p95 are 190/251 ms for
Q8 and 212/285 ms for Q4; timing is directional on the shared workstation.
The saved baseline profile is [../configs/kev-cpp.json](../configs/kev-cpp.json).

The native decision head requires the option span to fit within a logical
batch. The initial 512-token batch rejected one 751-token option span; 1,024
fixed it without changing labels or truncating inputs. Cache F16, one slot,
context 4096 and flash attention are the tested baseline. Larger option spans
or states need explicit larger batch/context and validation, not truncation.
Four-slot trial (`../results/kev-cpp-parallel-v1/report.md`, local only) uses context 16384
to preserve 4096 per slot. It scores the same 185/214 and 239/299, median 179 ms,
but uses 5.54 GiB and changes one marginal binary decision (max probability
delta 0.01655); it remains a separate performance candidate.

## Kev 9B Q4_K_M

The official native [9B GGUF](https://huggingface.co/ggml-org/Kev-9B-GGUF)
is pinned to `ec2bbfe6620218aee2e01cc93bb78dee2a96ed58` and verified against
its published SHA-256 and 6,358,923,744-byte size in
[downloads/kev-9b-q4.json](downloads/kev-9b-q4.json). Embedded metadata identifies
the Kev head and fitted temperature 2.19365; its source pointer is recorded in
[downloads/kev-9b-gguf-metadata.json](downloads/kev-9b-gguf-metadata.json).

The matched test (`../results/kev-9b-q4-v1/report.md`, local only) uses the same 64 Odysseus
and 228 upstream cases, image and single-slot settings as corrected 4B. All
33/33 layers are on CUDA, and all 292 requests return valid native probabilities
with zero generated tokens and zero request errors. Peak process VRAM is
6,390 MiB (6.24 GiB), median/p95 Odysseus response time 429/572 ms.

Odysseus accuracy is 190/214 (88.8%), 42/64 exact cases and one unsafe approval.
Compared with 4B Q8, 13 fields improve and eight regress, a net five-field gain;
safety-gate fields improve from 38/44 to 42/44 while GUI falls from 35/40 to
34/40. Upstream accuracy totals 244/299 versus 239/299 for 4B Q8, but Typed
Decisions falls from 43/60 to 36/60. This small heterogeneous smoke does not
establish universal model superiority. Latency is directional on the shared
workstation; 9B's observed median is about 2.25 times 4B Q8's.

The candidate settings are [../configs/kev-9b-cpp.json](../configs/kev-9b-cpp.json).
`scripts/download_kev_gguf.py --models 9b-q4` reproduces pinned acquisition;
`scripts/test_kev_cpp.py --models 9b-q4` tests it locally. The upstream runner's
`kev-9b-q4` backend supports the same suite adapters. Existing defaults and saved
Jev results are unchanged; no cloud inference was called.

## Additional runtime notes

A fresh 64-case FP32 Kev-0.8B measurement is saved in
memory.json (`../results/kev-memory-profile-v1/memory.json`, local only). Persistent CUDA
tensors occupied 3,020,208,128 bytes (2.81 GiB), peak live inference tensors
3,221,646,336 bytes (3.00 GiB), and cached allocator reservation peaked at
3,296,722,944 bytes (3.07 GiB). NVIDIA reported 3,334 MiB (3.26 GiB) for this
process after the run, including CUDA overhead. This is a new measurement;
the earlier benchmark did not record memory. Desktop/other process memory is
excluded. Longer inputs can require more workspace.

[ggmlc](https://github.com/monatis/ggmlc) is another native C++/CUDA option:
its `laya` executable supports Kev via `/v1/systemone` and CUDA graphs. The
[mys Kev-4B files](https://huggingface.co/mys/kev-4b-GGUF) are compiler-specific
and cannot be loaded by llama.cpp. This alternative remains untested locally.

Kev's default documented CUDA serving path uses bf16 and fused DeltaNet kernels.
The RTX 2080 Ti needs a separately validated compatible precision/kernel path;
the upstream fp32 evaluation path is a candidate. No local runtime or quality
claim is made by downloading the files.

Laya's chosen checkpoint is specialized to four synthetic workflows and its
card warns that its confidence is not calibrated for arbitrary workloads.
Verdict 2.0 is distinct from the downloaded general-purpose
`heman10x/rlcd-modernbert-151m` model; the specialized weights are unavailable.
Kev-0.8B's card explicitly cautions against tool-call routing because its
published When2Call performance is below chance.

## Download verification and scope

`scripts/download_decision_models.py` fetches public assets into project-local
directories. Hugging Face revisions are resolved to full commit IDs, large
files are verified against published LFS SHA-256 and size, and small Hub files
against their published Git blob IDs. General-purpose Verdict weights are
downloaded from its public Hub repository; the specialized Verdict 2.0 pointer
is retained as evidence and is not a downloaded checkpoint. GitHub reference files have their exact revision and
locally calculated checksums recorded.

No upstream code is executed, no package dependencies are installed, and no
Jev requests or model inference are part of this download operation. These
models are not yet integrated into `scripts/retest.py` or `configs/local-best.json`.

## Subsequent runtime validation

These download-time boundaries are historical. The models are now integrated
through `decision_lab/specialists.py`, `scripts/upstream_benchmarks.py` and
`demo.py --backend`, with dedicated FP32 CUDA inference validated on SM75.
Kev uses upstream reference DeltaNet/conv kernels and its fitted checkpoint
temperature; Laya uses strict complete checkpoint loading and original option
markers; Verdict uses the original GLiClass engine and an order-preserving
original-candidate evaluator. The frozen original retest/default config remains
unchanged. See benchmark readiness (`../results/benchmark-readiness-v1/report.md`, local only)
and [source/protocol inventory](benchmark-inventory.md).
