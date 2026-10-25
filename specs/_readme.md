# Local decision evaluation specs map

This project compares downloaded GGUF, NLI and custom decision-head models with saved Jev reference
sets: the original 24-case pass and the separate 64-case pass.
It evaluates decisions; it does not execute workstation actions or enforce
permissions. Specs describe current implementation. Evaluation evidence lives in
ignored local `results/`, not in specs or the published source tree.

| Spec | Sources | Scope |
|---|---|---|
| [evaluation.md](evaluation.md) | `decision_lab/`, `scripts/`, `benchmarks/`, `configs/` | Backend contracts, selection protocol, saved evidence, runtime ownership and verification. |
| [benchmarks.md](benchmarks.md) | `benchmarks/upstream/`, `vendored/`, benchmark scripts and adapters | Original-data ownership, fidelity, metrics, dedicated runtimes and full-run readiness. |
| [new_decision_comparison.md](new_decision_comparison.md) | new-model download/comparison/report scripts, Mica adapter, workspace campaign bridge | Matched Clef/Mica/Kev/Jev runs, ABI checks, shared budget and source recovery. |
| [workspace_replay.md](workspace_replay.md) | workspace replay runner, frozen scoring plan, shared campaign | All prepared project cohorts, saved baselines, local runtime capacity and diagnostic boundaries. |
| [public_trials.md](public_trials.md) | public source intake, owned adapters, subset planner and execution gates | Pinned external checkouts, static review, provenance, grouped subsets, cost estimates and shared-budget admission. |
| [publication.md](publication.md) | `.gitignore`, public docs, script labels, HTTP/runtime helpers | Publication exclusions, standalone dependencies, credential boundaries and export checks. |

Read the relevant spec before changing its contracts, and update it with source
changes. This directory is tracked with the source. Credentials, models, private
work/runtime data, root results and machine-specific preparation records remain
ignored. See `publication.md` for the public boundary.
