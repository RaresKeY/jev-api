# Public benchmark intake and gated trials

`decision_lab/public_sources.py` and `scripts/prepare_public_sources.py` own
acquisition and static review. External repositories are acquired beside the workspace
under `~/external-repos`, with exact origins, detached commits and clean Git
trees. Hooks, LFS smudge, package installation, submodules and upstream runners
are not executed. Acquisition, static scan, reviewed-file export, LFS object
resolution and HF row export are separate stages.

The versioned bundle is `benchmarks/public-20261015/`. Original data, notices,
manifests and reference-only scoring sources remain unchanged under `upstream/`.
The static inventory binds every tracked file by hash, parses Python syntax and
records capability matches without importing it. It is triage, not certification.
Only explicitly selected files are exported. Original Parquet objects must match
their pinned LFS size and SHA-256. HF viewer pages must carry the same exact
`X-Revision`, complete consecutive row indexes and no truncated cells. Converted
JSONL has separate hashes and page receipts. No new Parquet reader is installed.

`decision_lab/public_benchmarks.py` owns the adapted data. Each case binds its
original file/row/hash, repository revision, upstream provenance and scoring
boundary. Only state/questions enter model calls. Gold, CoT solutions, rationales
and provenance remain local. Original question and option order survive. JevBench
uses the existing byte-identical upstream scorer. SemIf keeps original option
order on ties and mean family balanced accuracy. Kev keeps ordered ties and
separates all-selected metrics from clean knowable rows. Evidence-removed rows
have no correctness label; confidence and intact-control comparisons are separate.
MMLU-Pro is a direct typed-choice adaptation, not its original CoT evaluation.

`decision_lab/public_trials.py` and `scripts/public_trials.py` own offline plans,
estimates, verification and result summaries. Default suites exclude development
and validation. Subsets use a seed and suite/family round robin, with no label or
prediction selection. SemIf source groups and perturbations stay together. Kev
groups also close over explicit parent/control links. Limits are per-suite upper
bounds; atomic groups can leave fewer rows. Full selection requires both `--full`
and a zero limit. Development/validation require `--include-development`.

A plan binds corpus bytes, exact selected rows and executable implementation
hashes. Execution revalidates the source-verified corpus. It requires `--execute`,
the exact plan SHA-256, an explicit admitted model and `--max-requests`. Local
runtimes use the existing manager/Podman wrappers, publisher-verified weights and
the retained df18bee image. Mica uses the existing verified author adapter in a
network-disabled managed container; its ephemeral ABI/library cache must first be
prepared by the existing offline helper. No external repository code is imported.

Jev additionally requires `--live` and `--max-new-tokens`, with a floor of 131,072
for one existing conservative input/output reservation. The plan's upper estimate
must fit this cap. The canonical existing 5M campaign and 2M per-dataset input
guards remain authoritative; preparation never resets or writes accounting.
The per-run ceiling conservatively includes cached response usage. Actual new
charges are reconciled from selected primary identities. Cached primary requests
are reused by the existing transport. Unknown usage halts the campaign. There
are no automatic retries. A budget stop leaves pending rows explicitly missing
and the run incomplete; failure-inclusive planned-denominator metrics remain.

Token estimates are transparent UTF-8-byte scenarios with output allowance and
25% headroom, not exact provider counts. Time scenarios use earlier median and
twice-p95 request timings, with startup and workload differences additional.
No inference is implied by acquisition, extraction, planning, estimates or tests.

External checkouts are disposable after extraction and offline verification.
Published SemIf scorer predictions and its summary are retained as data under
`retained-evidence/`, with a separate manifest binding their original reviewed
hashes and revisions. This preserves the original corpus and prepared plan hashes.
`cleanup.json` records exact checkout identities, clean-tree and worktree checks,
removed allocated bytes and retained evidence hashes. The offline audit permits
absent checkouts only with this completed receipt and verifies the retained data,
scorer parity and plans without Git clones or network access. Original acquisition
and static review records remain historical provenance after checkout removal.

## Source checkout boundary

Generated acquisition/cleanup receipts, forecasts, plans and execution
verification receipts are local ignored artifacts. Their original machine and
accounting paths are not rewritten into public data. The included corpus,
scorers, upstream public evidence and provenance remain unchanged. A fresh
checkout can compute token estimates without historical timing results; missing
timings are null. Campaign planning/execution and historical archive checks still
need the external inputs listed in `docs/standalone-dependencies.md`.
Offline result accounting uses the original ASCII JSON request identity without
importing the external live campaign client; a pinned Unicode hash vector guards
compatibility with existing primary identities.
`report_public_preparation.py` writes its generated local report under
`results/public-benchmark-preparation-20261015/`, not over the curated public doc.
