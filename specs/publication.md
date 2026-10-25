# Publication boundary

Source ownership: `.gitignore`, `README.md`, `docs/standalone-dependencies.md`,
script dependency docstrings, `configs/kev-cpp.json`, `configs/kev-9b-cpp.json`,
`decision_lab/http.py`, and `decision_lab/runtime.py`.

The committed tree contains source, synthetic fixtures, public source data,
configs and preserved upstream provenance/notices. No project license is added.
Credentials, models, root results, private inputs, runtime state and generated
machine/accounting-specific preparation records remain ignored at their original
local paths. Root `/results/` ignores do not exclude upstream public evidence
in nested benchmark directories. Do not force-add ignored artifacts.

Public documents omit personal session pointers, private storage links and chat
instruction history. Public runtime configs retain model identities and settings,
but not raw host image IDs or builder/vendor label inventories.
Generated acquisition and cleanup metadata is excluded rather than rewriting
original dataset/scorer bytes or weakening integrity checks. Historical result
references are local paths, not promises of bundled artifacts.

Every script declares missing standalone dependencies in its module docstring;
the dependency guide maps entry points to external runtimes, data and projects.
Shared-manager preflight rejects missing executables before creating run output.
The Mica smoke entry point has no import-time inference. Token estimates work
without ignored historical timing results, reporting unavailable timings as null.
Optional-library tests skip explicitly when dependencies are absent. Private
archive tests require original frozen source, inputs and explicit
`JEV_TEST_PRIVATE_REPLAY=1`; portable tests never inspect that cohort by default.
The preparation-report generator writes only to ignored `results/`, preserving
the curated public summary.

The included `request_json` transport rejects redirects and requires HTTPS for
credential-bearing requests except for loopback HTTP. URLs cannot embed credentials. Transport errors expose neither endpoint
nor response bodies, and requests are never automatically retried. The workspace
campaign comparator imports a separate, unbundled transport from Jev Pipelines;
these helper-level guarantees do not certify that external implementation.

Verify the staged/exported tree for configured credential values, personal home
and session paths, private host/storage URLs and unexpected outputs. Run the
portable unit tests both in the owning checkout and a clean staged-tree export,
with no key, model, result directory or sibling project. Preserve ignored local
evidence and existing accounting while preparing a commit.
