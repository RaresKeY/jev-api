# Structured decision models: measured performance

**15 October 2026.** Text-only decisions on frozen inputs, combining new and historical runs. Jev 1.14 improved over saved 1.13 on five of eight objective workload groups and declined on three. Clef led the new local candidates on Odysseus gates; Mica led on several reasoning and prerequisite sets.

## Test sets and harness

| Set | What was tested | Harness and scoring |
|---|---|---|
| Odysseus gates | 64 synthetic cases, 214 fields: intent, tools, routing, authorization, completion and GUI alignment | Python `decision_lab`; authored labels; Boolean/choice accuracy and expected ordinal score within 0.75. Scores exclude 12 development cases and include 12 previously exposed holdout cases plus 52 later authored cases. |
| Public upstream smoke | 228 original records, 299 questions: 12 records from each of 19 JevBench, Kev, Typed Decisions, Chinese Laya, Verdict/banking/OOS, SNLI/MNLI and ARC suites | `scripts/upstream_benchmarks.py`; pinned data and JevBench scorers; original distribution, lexical tie and ordinal-level rules. Typed Decisions gold is teacher agreement. |
| Supper Guard | Calibration mechanics, authored preferences, prerequisites and drone-message decisions | Game Godot adapters generated fixtures; Python replay preserved original validation. No new gameplay or win-rate measurement. |
| ANLI, HANS, PAWS | Selected original test records for entailment and paraphrase classification | Existing Jev Eval schemas, labels and splits; main classification reported separately from auxiliary Boolean checks. |
| Fleet and pipeline | Held-out fleet fixtures across layouts/option orders; pipeline feasibility across saved conditions | Original labels, batch validation and repeated trials retained. |
| Browser, routing and Evo | Saved browser observations/actions, routing decisions and simulation probes | Request replay; agreement with saved Jev 1.13, without correctness labels or live execution. |

Browser traces originated in a Python standard-library agent using restricted Flatpak ungoogled Chromium, CDP pipes and observed DOM controls. This comparison replayed its saved decisions without relaunching the browser.

All 12 prepared groups completed: **2,942 requests and 7,438 questions per profile**, including 64 reused Odysseus requests. Batches, rubric variants and correlated repeats are not independent tasks. Gold/rationales stayed outside model inputs. State, questions, option order and scoring were preserved; no training or prompt/threshold fitting used new results.

## Matched gates and upstream smoke, including older models

Rows share the 64 Odysseus cases and, where available, the 228-record upstream sample. **H** marks historical runs. Odysseus median latency excludes loading/warmup; comparisons across runtimes and run times are directional.

| Model as tested | Odysseus correct /214 | Exact cases /64 | Unsafe approvals | Upstream correct /299 | Median ms |
|---|---:|---:|---:|---:|---:|
| Jev 1.13.0, API **H** | 204 (95.3%) | 54 | 0 | not measured | 576 |
| Jev 1.14.0, API | 202 (94.4%) | 52 | 0 | 254 | 355 |
| Clef-Flash 9B Q4_K_M | 197 (92.1%) | 51 | 0 | 232 | 439 |
| Mica 0.1 4B Q8_0, one sequence | 187 (87.4%) | 42 | 0 | 239 | 415 |
| Kev 4B Q8_0, current engine | 183 (85.5%) | 37 | 1 | 237 | 275 |
| Kev 9B Q4_K_M **H** | 190 (88.8%) | 42 | 1 | 244 | 429 |
| Kev 4B Q8_0, earlier engine **H** | 185 (86.4%) | 38 | 1 | 239 | 190 |
| Kev 4B Q4_K_M **H** | 185 (86.4%) | 38 | 1 | 238 | 212 |
| Kev 0.8B, FP32 native **H** | 169 (79.0%) | 30 | 0 | 215 | 107 |
| Laya typed-decisions, 421M **H** | 142 (66.4%) | 20 | 2 | 190 | 46 |
| NLI DeBERTa-v3-large **H** | 137 (64.0%) | 15 | 4 | 146 | 111 |
| NLI DistilRoBERTa-base **H** | 125 (58.4%) | 12 | 2 | 130 | 12 |
| Qwen2.5-0.5B-Instruct Q4_K_M **H** | 119 (55.6%) | 7 | 6 | 121 | 330 |
| General openJev Verdict, 151M **H** | 92 (43.0%) | 6 | 6 | 145 | 37 |
| SmolLM2-135M-Instruct Q4_K_M **H** | 77 (36.0%) | 2 | 6 | 93 | 579 |

Jev 1.14 improved two Odysseus fields and regressed four, all involving browser requirements. The same Kev 4B weights fell from 185 to 183 with the newer engine, demonstrating runtime drift. Older Laya scored 47/60 on Typed Decisions despite its lower gate score.

Mica's separate exploratory eight-sequence profile scored 188/214 and 241/299, with 309/364 ms median/p95 and 5.85 GiB sampled VRAM; its answers changed. Only Mica 0.1 was measured. Verdict denotes the general checkpoint; specialized Verdict 2.0 weights were unavailable.

## Broader project replay

Percentages use original labels and include failures. Kev uses the separate **4096-token physical batch** profile. Odysseus retains the original 4096-context runs above.

| Workload, scored judgments | Jev 1.13 saved | Jev 1.14 | Clef 9B Q4 | Mica 0.1 Q8 | Kev 4B Q8 |
|---|---:|---:|---:|---:|---:|
| Supper calibration mechanics, 243 | 97.9% | 97.1% | 94.2% | 93.0% | 75.7% |
| Supper authored preferences, 131 | 65.6% | 67.9% | 65.6% | 63.4% | 29.8% |
| Supper prerequisites, 282 | 84.4% | 86.9% | 65.2% | 73.4% | 42.2% |
| ANLI selected test, 628 | 68.2% | 70.9% | 48.7% | 58.1% | 50.6% |
| HANS selected test, 223 | 93.7% | 95.1% | 92.4% | 87.0% | 83.4% |
| PAWS selected test, 181 | 82.9% | 82.3% | 77.3% | 83.4% | 77.9% |
| Fleet held-out fixtures, 86 | 87.2% | 88.4% | 52.3% | 72.1% | 54.7% |
| Pipeline feasibility, 168 | 59.5% | 64.9% | 42.3% | 37.5% | 46.4% |

The five improving objective groups are prerequisites, ANLI, HANS, fleet and pipeline. Authored strategy agreement does not establish optimal play.

| Diagnostic agreement with saved Jev 1.13 | Jev 1.14 | Clef | Mica | Kev, batch 4096 |
|---|---:|---:|---:|---:|
| Browser decisions | 47/51 | 38/51 | 31/51 | 41/51 |
| Task routing | 211/240 | 213/240 | 176/240 | 207/240 |
| Supper drone messages | 62/74 | 43/74 | 31/74 | 57/74 |
| Evo probe | 21/24 | 11/24 | 17/24 | 5/24 |

These are **agreement counts, not task-success rates**. No live browser, routing executor or simulation outcomes were tested. Two additional unlabelled prerequisite judgments remain diagnostic.

## Local deployment and measured resources

Hardware: **NVIDIA RTX 2080 Ti, 11 GB**, shared with other GPU work. Clef/Kev used llama.cpp native `/v1/systemone`, commit `df18bee11e96557b07aa7b12db0976f3c7ac9165`, CUDA 12.6, full GPU offload, F16 KV cache, one slot and rootless Podman. Mica used the author's direct-prefill readout, original tokenizer/template and shipped temperature, with label/ABI checks. These paths generated zero text tokens. Historical Kev GGUFs used `a4cb4c6`; older specialists used PyTorch, NLI used pair classification, and instruct models used constrained chat.

| Broad replay profile | Context | Physical batch | Median / p95 ms | Sampled peak process VRAM GiB | Failed requests /2942 |
|---|---:|---:|---:|---:|---:|
| Jev 1.14 API | provider-managed | provider-managed | 412 / 536 | API | 2 |
| Clef 9B Q4 | 12,288 | 12,288 | 615 / 1022 | 8.61 | 3 |
| Mica 0.1 Q8, one sequence | 16,384 | 512 | 556 / 1129 | 5.01 | 0 |
| Kev 4B Q8, initial profile | 16,384 | 512 | 467 / 1065 | 5.41 | 266 |
| Kev 4B Q8, capacity follow-up | 16,384 | 4096 | 443 / 904 | 9.01 | 0 |

Timing uses the same **2,608 new requests valid under every profile**, excluding reused gates. Mica runs in-process; others include HTTP. Sampled process VRAM includes cache/workspace and can miss peaks. Neither metric establishes dedicated-host throughput or minimum memory.

Jev had two advertised-choice validation failures; Clef had three oversized batches. Kev's 266 failures exceeded its physical batch despite fitting context. The separately frozen 4096 profile removed capacity failures without changing weights/prompts/labels; prerequisite accuracy rose from 20.6% to 42.2%. Both profiles remain recorded. Clef 16K and Kev batch 8192 failed memory preflight before scoring. Failures were never silently truncated, repaired or retried.

## Examples from the measured sets

Descriptions of actual synthetic fixtures. Expected answers illustrate the rubric, not each model's prediction.

| Fixture | Supplied situation | Expected decision |
|---|---|---|
| Supper affordability, anonymous variant | 24 wood, drone cost 24; buy now if affordable, otherwise wait, with no inferred income | `option_0`: buy one drone |
| Odysseus `tools-4` | Click Submit in Chrome using the desktop mouse; no URL fetching or file edits | GUI required; shell, filesystem and browser-fetch tools not required |
| Odysseus `odysseus-completion-03` | Help page updated, but the requested regression test was not added | Incomplete; retry; completion quality level 1 |
| Odysseus `odysseus-gui-02` | Goal is Inbox; proposed click targets the clearly visible Settings tab | Misaligned; ambiguity level 0; ask under this suite's strategy rubric |

## Scope and evidence

Synthetic labels are author-assigned; early debugging exposed some cases. Upstream samples are small and heterogeneous; repeats are correlated. One new pass versus historical API replies cannot isolate model changes from provider variation. Zero unsafe approvals on 11 safety cases does not establish authorization safety. Accuracy, calibration, strategy and execution success are distinct.

Campaign API usage, including smoke tests: **3,853,066 tokens** (3,576,277 input, 276,789 output). Estimated charge at the recorded $0.042/M input rate, free output: **$0.15020**, excluding local hardware/electricity. This records run costs, not current pricing.

Evidence: broad replay (`../results/workspace-replay-20261015-v1/report.md`, local only), analysis JSON (`../results/workspace-replay-20261015-v1/analysis.json`, local only), matched runs (`../results/new-decisions-20261015-v1/report.md`, local only), historical Kev (`../results/kev-9b-q4-v1/report.md`, local only), older specialists/upstream (`../results/benchmark-readiness-v1/report.md`, local only), [Odysseus fixtures](../benchmarks/odysseus-v2.json). Offline reporting/verification uses `scripts/analyze_workspace_replay.py`, `scripts/report_new_decisions.py` and `scripts/verify_workspace_replay.py`. Private traces are excluded; full workspace reproduction requires retained private evidence.

Model provenance: [Jev](https://docs.typesafe.ai/models), [Clef GGUF revision](https://huggingface.co/ggml-org/Clef-Flash-GGUF/tree/6b4fe4c59098479652d94f0fe11a40facbebd1ec), [Mica checkpoint revision](https://huggingface.co/sky7350/Mica-v0.1-4B/tree/9b6428b272f4708ef54a4702b52bb453a4059363), [Mica inference source](https://github.com/akivet/Mica/tree/4f47a21bfc3ca9d1fd26d789c485ea24d7c04e2e), [Kev 4B GGUF](https://huggingface.co/ggml-org/Kev-4B-GGUF). Original benchmark sources, revisions and scoring differences are catalogued in the [source inventory](benchmark-inventory.md).
