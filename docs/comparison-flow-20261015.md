# Comparison protocol

The evaluation compares fixed synthetic workstation fixtures and original public
benchmark records. Labels, rationales, dataset provenance and scoring metadata
stay outside model requests. Only the declared model, state and questions enter
inference. The lab classifies supplied data; it never executes a proposed action.

Local runs use verified weight hashes, explicit runtime profiles, real GPU
allocation evidence and fresh output directories. Source datasets, score rules,
question order, model revisions and capacity failures remain explicit. Changes
in runtime or quantization are separate profiles, not silently repaired trials.
Development cases select settings; scored cases do not tune those settings.

Cloud experiments require explicit execution flags, persistent input/output
reservations and reconciliation of actual usage. Unknown spend halts dispatch.
Requests are never automatically retried. A completed or interrupted attempt
cannot be silently rerun. Existing campaign state is local and is never reset
by source acquisition, reporting or this repository's public preparation.

Reports distinguish synthetic label agreement, original benchmark accuracy,
agreement with saved decisions and actual task success. A replay does not launch
a browser, game or workstation action. Small correlated cohorts do not establish
production authorization safety or general model superiority.

Historical results and private replay evidence are not bundled. See
[standalone dependencies](standalone-dependencies.md) for the external inputs,
shared runtime and original campaign records needed by archive-only helpers.
