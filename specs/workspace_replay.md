# Prepared workspace replay

`scripts/replay_workspace.py` and `decision_lab/workspace_replay.py` own the
recorded 15 October 2026 replay across 2,942 requests in Jev Pipelines'
frozen workspace bundle. The four models are Jev 1.14, Clef-Flash 9B Q4,
Mica 0.1 4B Q8 and retained Kev 4B Q8. Saved Jev 1.13 replies are the matched
baseline. No old weight download, training, browser action or game execution
belongs to this workflow. Source projects are inspected as data only.

The owner-only plan and response files live in
`.work/workspace-replay-20261015-v1/`. They include private routing and browser
inputs and must not be published. `results/workspace-replay-20261015-v1/`
contains aggregate reports and payload-free paired metrics. The plan binds
original source hashes, scoring labels, exact request order and runtime choices.
Source snapshots protect the executor. Only original state and questions enter
inference; labels, rationale, file paths and local scoring metadata stay local.

The runner statically recovers original labels from calibration/prerequisite
records, Jev Eval suites, fleet plans and pipeline plans. Original calibration
and prerequisite scores must reproduce before dispatch. Choice, ordinal Score
and Noul scoring keep their original semantics. Odysseus keeps its scalar Score
tolerance. Fleet preserves legacy-label normalization and whole-batch rejection
of inconsistent advertised choices. Development and test membership, rubric
variants, authored preference labels and correlated repeats remain explicit.
No threshold is fitted on the new results.

Completed 64-case Odysseus runs are reused for all four models after checking
unchanged state/questions. New local replays use the retained df18bee image,
full GPU offload, F16 cache and one sequence. Clef requires one joint microbatch;
16K failed GPU-memory preflight and 12,288 passed. Its batch, microbatch and
context are 12,288. Kev uses context 16,384, batch 1,024 and microbatch 512.
Mica uses the verified author readout, one question sequence, context/max length
16,384 and microbatch 512. Inputs are never silently shortened or split.
Capacity and invalid-response failures remain in the denominator.

Only `jev --live` dispatches paid requests. It reuses the existing canonical
5M input-plus-output ledger and 2M input-per-dataset guards. Historical repeated
requests remain distinct new trials: first occurrence uses its primary request
hash, subsequent occurrences use frozen repeat identities. Completed primary
Odysseus calls are reused. The earlier 64/228 comparison is already charged
against the same allowance. Unknown usage halts dispatch; no automatic retry,
allowance reset or resend of an attempted local row is permitted.

Shared LLM-manager and Godot-Podman wrappers own local runtimes. Actual PID GPU
memory is sampled, owned container removal is checked, and other GPU jobs stay
untouched. Timing is directional on the shared workstation. No deployment or
change to project decision policy is selected by this evaluation.

Reporting is offline. Correctness requires original gold; browser, routing,
drone-message and Evo records without labels report agreement and probability
changes only. A replay does not establish end-to-end task success, game win
rate, production safety or an optimal routing policy.

`scripts/analyze_workspace_replay.py` separately applies captured original
owning-project interpretation rules after generic transport validation. Supper
Guard retains its tighter distribution/weighted-score rules; application
adapters reject advertised choices that do not maximize their probabilities.
These invalidate the whole batch and remain billed. Original public evaluation
choice mismatches stay diagnostics under its original rubric. Raw inference
files are immutable; interpretation failures are recorded separately and
included in all derived error-inclusive metrics. Main classification headlines
exclude auxiliary Noul questions and use the selected test membership only.
Original development-selected acceptance thresholds are captured unchanged.
The final audit recomputes reports twice with network and key readers forbidden,
checks unchanged accounting bytes, settled repeat identities, GPU evidence and
exact owned-container removal.

Report commands are `python scripts/analyze_workspace_replay.py` and final
`python scripts/verify_workspace_replay.py`. The inference runner and its
initial strict report branch remain frozen; the separate offline
`decision_lab/workspace_report.py` supersedes that report branch. It binds the
actual canonical wire hash and verifies byte-identical ordered state/questions.
The comparator reordered only the three outer envelope keys on 19 saved
requests whose original key order was model/questions/state. This does not
change state, question/option order or values; the audit records each occurrence
explicitly instead of treating the source hash as the actual wire hash.

The inherited Kev physical microbatch of 512 rejects long question/option
branches independently of its 16K context. Its initial pass is preserved in
full. `scripts/replay_kev_capacity.py` freezes a separate 8192 physical/logical
batch profile, with the same 16K context, weights, prompts, gold and membership.
The 8192 candidate failed GPU-memory startup before any new scored request.
Its source, manifest and failed startup remain separate evidence. The scored
follow-up is `scripts/replay_kev_capacity4096.py`: physical/logical batch 4096,
the same 16K context and F16 cache. This exceeds every rejected branch in the
original pass (maximum 3527 tokens) and passed both semantic and long-option
preflights before scoring. It evaluates the complete prepared cohort. This is
an explicitly recorded runtime-capacity comparison, with no paid calls and no
resubmission of an original attempt identity. Both configurations remain in
the reports and final verification. The saved 64-case Odysseus control is reused
for this model as in the primary pass; it is not a fresh capacity-profile gate
measurement. No configuration is selected using correctness on the labels.

## Publication and reproduction

Both raw `.work/` inputs and root `results/` are ignored local artifacts.
The published source does not carry the original cohort, runtime images,
campaign ledger or sibling projects. Each entry point labels these dependencies;
see `docs/standalone-dependencies.md`. Historical receipts continue to validate
original frozen source bytes, not edited public source. No historical paid pass
is rerun or accounting reset by publication preparation.
