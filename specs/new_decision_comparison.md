# Current decision-model comparison

## Local Mica assets

Mica model assets are external, ignored inputs. Obtain the pinned GGUF,
tokenizer, prompt template, calibration, configuration and publisher notices
before inference or model-hash preflight. Adapter source remains under
`vendored/mica-v01`. Private storage locations and transfer receipts are not
part of this repository. See `docs/standalone-dependencies.md`.

`scripts/compare_new_decisions.py` owns the October 15 matched runs. Each model
uses the unchanged 64 Odysseus fixtures (214 fields) and original 228 upstream
smoke cases (299 fields) from `results/kev-cpp-input-v1/upstream-cases.json`.
The planner verifies original data hashes, weights and fresh output directories,
then saves cases, sources, requests/results, metrics and runtime evidence.
No model executes classified tasks, game actions, browser actions or Odysseus code.
No training or removed checkpoint restoration is performed.

## Model and runtime ownership

`download_new_decisions.py` acquires publisher-hashed Clef-Flash Q4 and Mica 0.1
Q8 weights, exact tokenizer/template and pinned Mica author source. The acquisition
manifest distinguishes the unavailable advertised Mica 0.5 repository. Original
Mica source is unmodified in `vendored/mica-v01`; notices remain alongside it.

Clef and retained Kev use `ManagedServer` through the shared LLM manager and the
existing df18bee CUDA/SM75 image. The optional explicit `ubatch` parameter leaves
old callers unchanged. Clef requires the whole joint prompt in its micro-batch:
batch and ubatch 4096, one server slot and context 4096. Kev uses batch 1024,
ubatch 512 and the same context/slot/F16 cache. Offload logs and actual NVIDIA
server PID allocations establish hardware execution. Startup/warmup are excluded.

`decision_lab/mica_native.py` loads the author's direct-prefill readout and
shipped calibration, with exact tokenizer JSON and Jinja prompt template.
Generic fast loading avoids a newer tokenizer class name in the retained Python
container, without changing tokenizer bytes. All 257 readout labels are checked.
The author binding needs one added `moe_cache_size` field for df18bee; a C probe
against pinned headers checks actual struct sizes and offsets before loading.
Isolated `LD_LIBRARY_PATH` binds the correct shared libraries. No generation,
sampler, grammar or probability fabrication is used.

One question sequence is the baseline. Eight sequences with author prefix-sharing
are a separate candidate, with context 4096 per question. Their predictions differ,
so the parallel candidate does not replace the baseline. Actual container process
GPU memory is observed from the host, and exact container removal is verified.
Mica's direct timing differs from the HTTP paths; shared-host timing is directional.
The extracted runtime cache and C probe executable are ephemeral. Recreate them
offline with `scripts/prepare_mica_runtime.py`; retained model/source evidence
and the underlying shared runtime images are preserved.

## Cloud ownership and budget

`decision_lab/jev_campaign.py` imports the versioned current transport and
output-inclusive ledger from sibling Jev Pipelines. `DatasetLedger` extends its
atomic reservation with a 2M cumulative input cap per dataset. The historical matched decision
comparison uses one existing campaign with a 5M input-plus-output cap. The
published source does not authorize spending or recreate that campaign.
`execution-authorization.json` fixes its canonical ledger directory. No old ledger
or preparation manifest is reset. Local calls do not consume provider tokens.

Each in-flight request reserves 65,536 input and 65,536 output before dispatch;
actual provider billing settles before typed interpretation. Raw safe responses
persist. Invalid typed responses remain charged. Unknown usage persistently halts
the campaign; there are no automatic retries. Primary request identities are
exact request hashes and cached calls are checked against canonical accounting.
The 64 gate requests overlap the prepared workspace bundle and must be reused.

The provider receives only `model`, `state` and `questions`. Local gold,
annotations and source paths are excluded. Scalar Jev choices
remain explicit. Original upstream distribution metrics retain their lexical
tie and ordinal rules, with explicit-choice discrepancies reported separately.

## Verification and limits

`report_new_decisions.py` recomputes scores, checks matched input/gold and frozen
source snapshots, reconciles each provider exchange with settled billing,
checks local zero-generation and sampled GPU memory, and verifies cleanup.
Floating metric reconstruction tolerates only 1e-12 roundoff, including Noul's
omitted complementary probability; discrete fields remain exact. Reporting is
offline and works independently of live model servers.

`tests/test_jev_campaign.py` protects atomic input/output ceilings, dataset caps,
persistent unknown-usage halting, explicit live opt-in and cached-call reuse.
Existing adapter/runtime tests and real semantic smokes cover the native path.
Results and paired changes live under `results/new-decisions-20261015-v1/`.

One-pass correlated synthetic and heterogeneous samples are not a general model
ranking, production safety proof or full private leaderboard reproduction. Old
and new engine/provider runs remain distinct; labels and prompts are not tuned
to force parity. These runners do not perform training or model provisioning.
