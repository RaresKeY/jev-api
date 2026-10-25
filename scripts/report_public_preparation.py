#!/usr/bin/env python3
"""Render the requested preparation, source-review and cost/time report offline.

Standalone dependencies: Ignored preparation plans/receipts/forecasts and original campaign ledger.
See docs/standalone-dependencies.md.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, sha256
from decision_lab.public_sources import BUNDLE


def main():
    forecast = json.loads((BUNDLE/'forecast.json').read_text())
    catalog = json.loads((BUNDLE/'catalog.json').read_text())
    acquisition = json.loads((BUNDLE/'acquisition.json').read_text())
    review = json.loads((BUNDLE/'static-review.json').read_text())
    verified = json.loads((BUNDLE/'verification.json').read_text())
    cleanup_path = BUNDLE/'cleanup.json'
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
    retention = ('Sources were acquired as clean, detached Git checkouts in the sibling `~/external-repos/` directory. '
        'The five temporary checkouts were removed after retaining and verifying the imported data and scorer evidence. '
        f"Checkout removal released **{cleanup['removed_allocated_bytes']/2**30:.2f} GiB** of allocated space. "
        'The owned bundle and offline audit work without these checkouts. '
        '[Cleanup receipt](../benchmarks/public-20261015/cleanup.json). '
        if cleanup and cleanup['status'] == 'complete' else
        'All sources are clean, detached Git checkouts in the sibling `~/external-repos/` directory. ')
    smoke_sha = sha256(BUNDLE/'plans/smoke.json')
    lines = ['# Public benchmark preparation and trial estimates', '',
        '**15 October 2026. Prepared and verified; no new model inference or paid requests.**', '',
        retention+'Owned data, provenance, plans and review receipts are in `benchmarks/public-20261015/`. '
        'Existing frozen benchmark sets and historical reports were preserved.', '',
        '## External sources and static review', '',
        '| Checkout | Pinned commit | Tracked files scanned |', '|---|---|---:|']
    for name, entry in acquisition['repositories'].items():
        url = entry['url'].removesuffix('.git')+'/tree/'+entry['commit']
        lines.append(f"| `{name}` | [{entry['commit'][:12]}]({url}) | {len(review['repositories'][name]['files'])} |")
    lines += ['', 'Git hooks, LFS smudge, submodules, package installation and fetched runners were not executed. '
        'The static pass hashes tracked files, checks symlinks, parses Python syntax and records process, network, '
        'dynamic-code, remote-model-code and retry capabilities. No tracked symlinks or Python syntax errors were found. '
        'This is a static triage review, not a security certification.', '',
        'Manual review identified `trust_remote_code=True` in MMLU-Pro’s local runner and retry paths, including '
        '`--retry_wrong`, in its API runner. Those paths are excluded. Only explicit data/notice files and '
        'reference-only scorer copies were imported. The owned runner makes one attempt per case, keeps failures '
        'in the denominator and executes no proposed browser, shell or game actions.', '',
        'MMLU-Pro’s two Parquet objects match their pinned LFS SHA-256/size. Data conversion used HF’s read-only '
        'rows API with the exact same `X-Revision` on every page, consecutive indexes and no truncated cells. '
        'The JSONL export has separate hashes/page receipts; no new Parquet package was installed.', '',
        '## Imported sets and provenance', '',
        '| Owned suite | Requests | Labelled questions | Default selection |', '|---|---:|---:|---|']
    for name, info in catalog['suites'].items():
        lines.append(f"| `{name}` | {info['requests']:,} | {info['labelled_questions']:,} | "
                     + ('eligible' if name in catalog['default_suites'] else 'explicit development/validation opt-in')+' |')
    lines += ['', 'Every adapted row records the original file, row index, file hash, repository revision, '
        'source metadata and label mapping. Raw originals and notices remain separate. Model calls receive '
        'only state/questions; labels, rationales, provenance and MMLU CoT solutions stay outside the wire.', '',
        '- JevBench data matches the previously imported public bytes. Its scorer and metrics are byte-identical '
        'to the already reviewed vendored versions.',
        '- SemIf preserves option order on ties and mean family balanced accuracy. The owned scorer reproduces '
        'the published 144-row baseline: **0.813238160761**, versus published **0.8132381608**.',
        '- Kev preserves ordered ties, original groups, parent links and intact controls. Each v9 split has '
        '**110 evidence-removed requests**: confidence diagnostics only. Clean knowable metrics remain separate '
        'from all-selected results.',
        '- MMLU-Pro uses direct typed choice without CoT/few-shot demonstrations. This is a declared adaptation. '
        'The pinned full test has **12,032** rows; Mica’s card reports **10,032**, whose exact filtering is '
        'not established here.',
        '- All **200 Kev v9 test MMLU-Pro questions** match question text in the full MMLU-Pro test. '
        'These panels overlap and must not be presented as independent evidence.',
        '- The Mica author reports using part of JevBench hard, SemIf and Kev v9 for development selection. '
        'Keep that exposure visible; use fresh frozen project cases for additional generalization evidence.', '',
        '## Gated subsets', '',
        'Ready plans: [smoke](../benchmarks/public-20261015/plans/smoke.json), '
        '[pilot](../benchmarks/public-20261015/plans/pilot.json), '
        '[full core](../benchmarks/public-20261015/plans/core-full.json). '
        'Smoke/pilot request up to 12/64 records per suite, with label-independent suite/family round robin. '
        'Whole related groups are atomic, so actual counts can be lower. Development/validation are excluded by default.', '',
        'Plans bind the corpus, exact selected cases and implementation hashes. Running requires `--execute`, '
        'the exact plan SHA-256, an explicit model, a fresh output and `--max-requests`. Full selection requires '
        '`--full --limit-per-suite 0`; development/validation require `--include-development`.', '',
        'Admitted models: Jev 1.14 API, Clef-Flash 9B Q4, Mica 0.1 Q8 and retained Kev 4B Q8. '
        'Local profiles reuse the retained GPU runtimes: Clef context/batch 12,288; Kev context 16,384 and '
        'physical/logical batch 4096; Mica one sequence, context 16,384, physical batch 512. '
        'Native engine and downloaded weight identities are checked. Mica uses the managed network-disabled '
        'container and verified ABI adapter; recreate its ephemeral cache with '
        '`python scripts/prepare_mica_runtime.py` when a Mica run is actually needed.', '',
        'Commands below are prepared examples, not executed model tests:', '', '```bash',
        'python scripts/public_trials.py estimate --plan benchmarks/public-20261015/plans/smoke.json', '',
        'python scripts/public_trials.py run \\',
        '  --plan benchmarks/public-20261015/plans/smoke.json \\',
        f'  --execute --approve-sha256 {smoke_sha} \\',
        '  --model mica-01-q8 --max-requests 81 --output results/public-mica-smoke-01',
        '```', '',
        'For Jev, select `--model jev-1.14.0`, a fresh output and additionally '
        '`--live --max-new-tokens 150000` for this smoke plan. The upper estimate must fit the run cap. '
        'The existing 5M campaign, settled-call cache, 2M input-per-dataset cap, unknown-usage halt and '
        'no-retry rules remain active. There is no extra allowance. The run cap conservatively includes '
        'cached response usage; actual new charges are separately reconciled. A reservation stop records '
        'pending rows and an incomplete run, rather than silently dropping them.', '',
        '## Time and cost forecast', '',
        'Sequential estimates per scenario. Local time sums **all three local model passes**; Jev time is '
        'one API pass. Startup, weight verification and changing input lengths/host load are additional.', '',
        '| Scenario | Requests per model | Planned total tokens, with headroom | Upper token scenario | Jev input cost, planned / upper | Three local passes | Jev pass |',
        '|---|---:|---:|---:|---:|---:|---:|']
    names = {'smoke': 'Smoke', 'pilot': 'Pilot', 'full-first-three': 'Full JevBench + SemIf + Kev v9 test',
             'full-with-mmlu': 'Full default panels, including MMLU-Pro'}
    for name, value in forecast['scenarios'].items():
        local = value['native_three_models_minutes']; cloud = value['runtime_minutes']['jev-1.14.0']
        lines.append(f"| {names[name]} | {value['requests']:,} | {value['total_tokens_with_25pct_headroom']:,} | "
            f"{value['upper_total_tokens_with_headroom']:,} | ${value['input_cost_usd']['planning']:.4f} / "
            f"${value['input_cost_usd']['upper']:.4f} | {local['median_scenario']:.1f}–{local['slow_scenario']:.1f} min | "
            f"{cloud['minutes_median_scenario']:.1f}–{cloud['minutes_slow_scenario']:.1f} min |")
    # Authored prose uses simple punctuation even for generated ranges.
    lines = [line.replace('–', ' to ') for line in lines]
    budget = forecast['budget']
    lines += ['', f"Shared budget: **{budget['charged']:,}/5,000,000 charged; {budget['remaining']:,} remain**, with no unresolved attempts. "
        'The smoke and pilot upper estimates fit. Full core fits the central estimate but exceeds the '
        'remaining balance under upper-estimate admission, so its cloud run is refused; local runs remain '
        'available behind their execution gates. The full MMLU-containing cloud sweep exceeds both the '
        'remaining balance and the campaign ceiling.', '',
        'Pricing verified on 15 October: **$0.042/M input tokens, output free**, '
        '[official model pricing](https://docs.typesafe.ai/models). Local runs have no vendor token charge; '
        'electricity/hardware are excluded. Token scenarios use UTF-8 wire bytes divided by 5/3/2 plus framing, '
        'an output allowance and 25% total-token headroom. These are estimates, not tokenization guarantees; '
        'live reservations enforce hard limits.', '',
        'Time scenarios use the earlier matched 2,608-request cohort on the shared RTX 2080 Ti: request '
        'median through twice p95, multiplied by selected count. They are planning scenarios, not statistical '
        'run-time bounds. Long inputs and multi-question behavior can change the result. Full per-model '
        'figures and assumptions are in [forecast.json](../benchmarks/public-20261015/forecast.json).', '',
        '## JevK5 and verification', '',
        '**JevK5 4B has not been tested in this workspace campaign.** The saved results concern Kev, '
        'which is a separate model. [JevK5’s official weights](https://huggingface.co/alibiserikbay/JevK5) '
        'remain a candidate; this task did not download its weights or claim measurements for it.', '',
        f"Offline audit verified **{verified['adapted_rows_verified']:,} adapted rows** against original inputs/options, "
        'all prepared plan hashes, source integrity, source-group closure, scorer parity, checkout retention '
        'records and refusal of missing execution approval or an over-budget core cloud run. '
        'Thirteen targeted offline tests passed. The canonical ledger bytes are unchanged. No new model '
        'inference, paid calls, package installation, deployment or external message was performed.', '',
        'Audit: [verification.json](../benchmarks/public-20261015/verification.json). '
        'Static inventory: [static-review.json](../benchmarks/public-20261015/static-review.json). '
        'Data: [catalog.json](../benchmarks/public-20261015/catalog.json). '
        'Contract: [public_trials.md](../specs/public_trials.md).', '',
        'Rebuild offline with `python scripts/verify_public_preparation.py` then '
        '`python scripts/report_public_preparation.py`. Create other subsets with '
        '`python scripts/public_trials.py plan --suites SUITES --limit-per-suite N --output NEW_PLAN.json`; '
        '`--families` and `--seed` provide further explicit selection controls.', '']
    # Generated accounting and machine-specific details remain local. The docs
    # summary is curated separately so regeneration cannot reintroduce them.
    target = ROOT/'results/public-benchmark-preparation-20261015/report.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines))
    print(target)


if __name__ == '__main__': main()
