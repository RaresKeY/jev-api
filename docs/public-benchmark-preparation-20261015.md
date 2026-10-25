# Public benchmark preparation and trial estimates

Historical methodology and aggregate estimates from 2026-10-15. Machine-specific
plans, acquisition/cleanup receipts, forecasts and accounting state are excluded
from the source checkout. Reproduction prerequisites are listed in
[standalone dependencies](standalone-dependencies.md).

**Preparation snapshot: 15 October 2026.** The reported preparation did not run model inference.

The bundle contains pinned public data, source provenance and scorer evidence in
`benchmarks/public-20261015/`. Temporary acquisition checkouts are unnecessary
for reading the included corpus. Historical plans, cleanup receipts and live
campaign state remain local and are required by the archive-specific audit.

## External sources and static review

| Checkout | Pinned commit | Tracked files scanned |
|---|---|---:|
| `jevbench` | [609a664e81f4](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5) | 580 |
| `SemIf-OpenJev` | [23cf1f39fc95](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a) | 161 |
| `kev` | [dd375f27fd61](https://github.com/jaredpalmer/kev/tree/dd375f27fd61486c52204205e10a0d2e10f3b34a) | 3057 |
| `MMLU-Pro` | [f418b116db00](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/f418b116db00b065c2aea046518d8fcf74d39872) | 71 |
| `MMLU-Pro-data` | [b189ec765aa7](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be) | 7 |

Git hooks, LFS smudge, submodules, package installation and fetched runners were not executed. The static pass hashes tracked files, checks symlinks, parses Python syntax and records process, network, dynamic-code, remote-model-code and retry capabilities. No tracked symlinks or Python syntax errors were found. This is a static triage review, not a security certification.

Manual review identified `trust_remote_code=True` in MMLU-Pro’s local runner and retry paths, including `--retry_wrong`, in its API runner. Those paths are excluded. Only explicit data/notice files and reference-only scorer copies were imported. The owned runner makes one attempt per case, keeps failures in the denominator and executes no proposed browser, shell or game actions.

MMLU-Pro’s two Parquet objects match their pinned LFS SHA-256/size. Data conversion used HF’s read-only rows API with the exact same `X-Revision` on every page, consecutive indexes and no truncated cells. The JSONL export has separate hashes/page receipts; no new Parquet package was installed.

## Imported sets and provenance

| Owned suite | Requests | Labelled questions | Default selection |
|---|---:|---:|---|
| `jevbench-easy` | 48 | 48 | eligible |
| `jevbench-standard` | 72 | 72 | eligible |
| `jevbench-hard` | 111 | 111 | eligible |
| `semif-authored` | 144 | 144 | eligible |
| `semif-perturbations` | 108 | 108 | eligible |
| `kev-transfer-v9-test` | 1,264 | 1,154 | eligible |
| `kev-transfer-v9-development` | 1,264 | 1,154 | explicit development/validation opt-in |
| `mmlu-pro-test` | 12,032 | 12,032 | eligible |
| `mmlu-pro-validation` | 70 | 70 | explicit development/validation opt-in |

Every adapted row records the original file, row index, file hash, repository revision, source metadata and label mapping. Raw originals and notices remain separate. Model calls receive only state/questions; labels, rationales, provenance and MMLU CoT solutions stay outside the wire.

- JevBench data matches the previously imported public bytes. Its scorer and metrics are byte-identical to the already reviewed vendored versions.
- SemIf preserves option order on ties and mean family balanced accuracy. The owned scorer reproduces the published 144-row baseline: **0.813238160761**, versus published **0.8132381608**.
- Kev preserves ordered ties, original groups, parent links and intact controls. Each v9 split has **110 evidence-removed requests**: confidence diagnostics only. Clean knowable metrics remain separate from all-selected results.
- MMLU-Pro uses direct typed choice without CoT/few-shot demonstrations. This is a declared adaptation. The pinned full test has **12,032** rows; Mica’s card reports **10,032**, whose exact filtering is not established here.
- All **200 Kev v9 test MMLU-Pro questions** match question text in the full MMLU-Pro test. These panels overlap and must not be presented as independent evidence.
- The Mica author reports using part of JevBench hard, SemIf and Kev v9 for development selection. Keep that exposure visible; use fresh frozen project cases for additional generalization evidence.

## Gated subsets

Historical local plans (not bundled): smoke (`../benchmarks/public-20261015/plans/smoke.json`, local only), pilot (`../benchmarks/public-20261015/plans/pilot.json`, local only), full core (`../benchmarks/public-20261015/plans/core-full.json`, local only). Smoke/pilot request up to 12/64 records per suite, with label-independent suite/family round robin. Whole related groups are atomic, so actual counts can be lower. Development/validation are excluded by default.

Plans bind the corpus, exact selected cases and implementation hashes. Running requires `--execute`, the exact plan SHA-256, an explicit model, a fresh output and `--max-requests`. Full selection requires `--full --limit-per-suite 0`; development/validation require `--include-development`.

Admitted models: Jev 1.14 API, Clef-Flash 9B Q4, Mica 0.1 Q8 and retained Kev 4B Q8. Local profiles reuse the retained GPU runtimes: Clef context/batch 12,288; Kev context 16,384 and physical/logical batch 4096; Mica one sequence, context 16,384, physical batch 512. Native engine and downloaded weight identities are checked. Mica uses the managed network-disabled container and verified ABI adapter; recreate its ephemeral cache with `python scripts/prepare_mica_runtime.py` when a Mica run is actually needed.

Commands below are prepared examples, not executed model tests:

```bash
python scripts/public_trials.py estimate --plan benchmarks/public-20261015/plans/smoke.json

python scripts/public_trials.py run \
  --plan benchmarks/public-20261015/plans/smoke.json \
  --execute --approve-sha256 REVIEWED_PLAN_SHA256 \
  --model mica-01-q8 --max-requests 81 --output results/public-mica-smoke-01
```

For Jev, select `--model jev-1.14.0`, a fresh output and additionally `--live --max-new-tokens 150000` for this smoke plan. The upper estimate must fit the run cap. The existing 5M campaign, settled-call cache, 2M input-per-dataset cap, unknown-usage halt and no-retry rules remain active. The run cap conservatively includes cached response usage; actual new charges are separately reconciled. A reservation stop records pending rows and an incomplete run, rather than silently dropping them.

## Time and cost forecast

Sequential estimates per scenario. Local time sums **all three local model passes**; Jev time is one API pass. Startup, weight verification and changing input lengths/host load are additional.

| Scenario | Requests per model | Planned total tokens, with headroom | Upper token scenario | Jev input cost, planned / upper | Three local passes | Jev pass |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 81 | 47,503 | 70,093 | $0.0013 / $0.0021 | 2.2 to 8.2 min | 0.6 to 1.4 min |
| Pilot | 416 | 260,639 | 385,580 | $0.0073 / $0.0115 | 11.2 to 42.4 min | 2.9 to 7.4 min |
| Full JevBench + SemIf + Kev v9 test | 1,747 | 902,523 | 1,336,535 | $0.0245 / $0.0391 | 47.0 to 177.9 min | 12.0 to 31.2 min |
| Full default panels, including MMLU-Pro | 13,779 | 8,259,109 | 11,753,413 | $0.1981 / $0.3155 | 370.6 to 1403.2 min | 94.6 to 246.0 min |

Live admission uses cumulative input/output reservations and the active local
campaign ledger. The forecast tables describe workload sizes, not an available
account balance or permission to execute them.

Pricing verified on 15 October: **$0.042/M input tokens, output free**, [official model pricing](https://docs.typesafe.ai/models). Local runs have no vendor token charge; electricity/hardware are excluded. Token scenarios use UTF-8 wire bytes divided by 5/3/2 plus framing, an output allowance and 25% total-token headroom. These are estimates, not tokenization guarantees; live reservations enforce hard limits.

Time scenarios use the earlier matched 2,608-request cohort on the shared RTX 2080 Ti: request median through twice p95, multiplied by selected count. They are planning scenarios, not statistical run-time bounds. Long inputs and multi-question behavior can change the result. Full per-model figures and assumptions are in forecast.json (`../benchmarks/public-20261015/forecast.json`, local only).

## JevK5 and verification

The recorded comparison measured Kev, not JevK5. [JevK5's public weights](https://huggingface.co/alibiserikbay/JevK5)
are a separate candidate with no measurements in this report.

Offline audit verified **15,113 adapted rows** against original inputs/options, all prepared plan hashes, source integrity, source-group closure, scorer parity, checkout retention records and refusal of missing execution approval or an over-budget core cloud run. The historical receipt also records 13 targeted offline checks. It is separate
from the current portable test suite and does not establish fresh runtime or
provider validation.

Audit: verification.json (`../benchmarks/public-20261015/verification.json`, local only). Static inventory: [static-review.json](../benchmarks/public-20261015/static-review.json). Data: [catalog.json](../benchmarks/public-20261015/catalog.json). Contract: [public_trials.md](../specs/public_trials.md).

With the original local archive and campaign state restored, rebuild offline with `python scripts/verify_public_preparation.py` then `python scripts/report_public_preparation.py`. Create other subsets with `python scripts/public_trials.py plan --suites SUITES --limit-per-suite N --output NEW_PLAN.json`; `--families` and `--seed` provide further explicit selection controls.
