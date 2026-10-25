"""Offline integrity checks and report for the matched October 15 comparison.

Standalone dependencies: Ignored results, ../jev-pipelines, original campaign ledger/authorization and Podman cleanup checks.
See docs/standalone-dependencies.md.
"""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, evaluate_values, summarize, sha256, write_json
from decision_lab.benchmarks import score_record, summarize_rows
from decision_lab.jev_campaign import AUTHORIZATION, DatasetLedger, transport

OUTPUT = ROOT/'results/new-decisions-20261015-v1'
RUNS = {
    'Jev 1.14': 'jev-1.14.0',
    'Clef-Flash 9B Q4': 'clef-flash-q4-v2',
    'Mica 0.1 4B Q8, single sequence': 'mica-01-q8-v2',
    'Mica 0.1 4B Q8, eight sequences': 'mica-01-q8-eight-seqs',
    'Kev 4B Q8, current engine': 'kev-4b-q8-control',
}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def same_metrics(first, second):
    # Mica's Noul wire omits p(false), so 1-p(true) can differ from its original
    # softmax p(false) at machine precision. Discrete fields remain exact.
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(same_metrics(first[k], second[k]) for k in first)
    if isinstance(first, float) and isinstance(second, float):
        return math.isfinite(first) and math.isfinite(second) and math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    return type(first) is type(second) and first == second


def paired(first, second):
    changes = []
    for a, b in zip(first, second):
        if a['id'] != b['id'] or a['case'] != b['case']:
            raise ValueError('Paired inputs or gold differ')
        for key in a['fields']:
            old, new = a['fields'][key]['passed'], b['fields'][key]['passed']
            if old != new:
                changes.append(dict(id=a['id'], gate=a['case']['gate'], field=key, before=old, after=new))
    return dict(improved=sum(not x['before'] and x['after'] for x in changes),
                regressed=sum(x['before'] and not x['after'] for x in changes), changes=changes)


def audit():
    cases = read(ROOT/'benchmarks/odysseus-v2.json')['cases']
    upstream = read(ROOT/'results/kev-cpp-input-v1/upstream-cases.json')
    authorization = read(AUTHORIZATION)
    ledger = DatasetLedger(Path(authorization['ledger_directory']).expanduser(), authorization)
    budget = ledger.summary()
    if budget['charged_total'] > 5_000_000 or budget['unresolved'] or budget['halted']:
        raise ValueError('Campaign budget is over-limit or unresolved')
    summaries = {}
    for label, name in RUNS.items():
        out = OUTPUT/name
        manifest, summary = read(out/'manifest.json'), read(out/'summary.json')
        if manifest['status'] != 'complete':
            raise ValueError('Incomplete run: '+name)
        for path, expected in manifest['sources'].items():
            if sha256(out/'source-snapshot'/path) != expected:
                raise ValueError('Frozen source snapshot changed')
        cohort = read(out/'cases.json')
        if cohort != dict(odysseus=cases, upstream=upstream):
            raise ValueError('Case content or labels changed')
        workstation = rows(out/'odysseus-responses.jsonl')
        original = rows(out/'upstream-responses.jsonl')
        if len(workstation) != 64 or len(original) != 228:
            raise ValueError('Case coverage changed')
        if summarize(workstation) != summary['odysseus']:
            raise ValueError('Workstation summary mismatch')
        for row, case in zip(workstation, cases):
            if row['case'] != case or row['fields'] != evaluate_values(row.get('values',{}),case):
                raise ValueError('Workstation scoring mismatch')
        groups = {}
        for row, case in zip(original, upstream):
            if row['case'] != case:
                raise ValueError('Original upstream content changed')
            if 'error' in row:
                recomputed = score_record(case, predictions={})
            else:
                probabilities = {}
                for key, q in case['questions'].items():
                    answer = row['raw']['answers'][key]
                    probabilities[key] = ({'false':1-answer['noul'],'true':answer['noul']}
                                          if q['type']=='noul' else answer['probabilities'])
                recomputed = score_record(case, probabilities)
            if not same_metrics(row['fields'], recomputed):
                raise ValueError('Upstream original scorer mismatch')
            groups.setdefault(case['suite'], []).append(row)
        for suite, items in groups.items():
            expected = dict(summarize_rows(items), errors=sum('error' in r for r in items))
            if summary['upstream'][suite] != expected:
                raise ValueError('Upstream summary mismatch')
        if label == 'Jev 1.14':
            calls = Path(authorization['ledger_directory']).expanduser()/'calls'
            for row in workstation+original:
                case = row['case']
                request = dict(model='jev-1.14.0', state=case['state'], questions=case['questions'])
                identity = 'primary--'+transport.digest(request)
                saved = read(calls/(identity+'.json'))
                if saved['request'] != request or saved['response'] != row['raw']:
                    raise ValueError('Raw Jev response or input differs from accounted call')
                transport.validate(saved['response'], request)
                ledger.verify(identity, saved['dataset'], transport.digest(request), saved['response']['usage'])
        else:
            memory = read(out/'memory.json')
            if not memory['samples'] or memory['peak_process_mib'] != max(r['used_mib'] for r in memory['samples']):
                raise ValueError('GPU memory evidence mismatch')
            if not name.startswith('mica'):
                runtime = read(out/'runtime/runtime.json')
                container = runtime['settings']['LLM_CONTAINER']
                if subprocess.run(['podman','container','exists',container], stdout=subprocess.DEVNULL).returncode != 1:
                    raise ValueError('Owned native container remains')
            elif not memory['removed'] or subprocess.run(['podman','container','exists',memory['container']], stdout=subprocess.DEVNULL).returncode != 1:
                raise ValueError('Owned Mica container remains')
            if any(r.get('raw',{}).get('usage',{}).get('output_tokens') != 0 for r in workstation+original if 'error' not in r):
                raise ValueError('Local decision path generated output tokens')
        summaries[label] = summary
    current = rows(OUTPUT/'jev-1.14.0/odysseus-responses.jsonl')
    old = rows(ROOT/'results/jev-odysseus-64-v1/responses.jsonl')
    delta = paired(old, current)
    mica_delta = paired(rows(OUTPUT/'mica-01-q8-v2/odysseus-responses.jsonl'),
                        rows(OUTPUT/'mica-01-q8-eight-seqs/odysseus-responses.jsonl'))
    kev_delta = paired(rows(ROOT/'results/kev-cpp-ab-v2/q8/odysseus-responses.jsonl'),
                       rows(OUTPUT/'kev-4b-q8-control/odysseus-responses.jsonl'))
    result = dict(ok=True, matched_cases=292, odysseus_fields=214, upstream_fields=299,
                  budget=budget, jev_113_to_114=delta, mica_single_to_parallel=mica_delta,
                  kev_old_to_current_engine=kev_delta,
                  explicit_jev_choice_mismatches=sum(len(r.get('explicit_choice_mismatches',[]))
                      for r in rows(OUTPUT/'jev-1.14.0/upstream-responses.jsonl')),
                  checks=['Frozen inputs/gold and source snapshots','Recomputed original scoring',
                          'Each Jev request and response matches canonical settled billing',
                          'Global token ceiling and no unresolved calls','Local zero generation',
                          'Actual process GPU allocation and sampled memory','Owned containers removed'],
                  floating_metric_reconstruction_tolerance=1e-12, cloud_calls_from_audit=0)
    write_json(OUTPUT/'verification.json', result)
    return summaries, result


def report():
    summaries, verification = audit()
    lines = ['# Matched Jev and new local decision models', '',
        'October 15, 2026. Same 64 synthetic Odysseus cases (214 fields) and 228 original upstream smoke cases (299 fields) as the earlier Kev comparison. Prompts, labels, option order, score tolerance and original metrics are unchanged. No training, prompt changes or confidence tuning. Both Mica runtime modes are reported explicitly; the parallel mode is exploratory.', '',
        '| Model | Gates | Exact /64 | Unsafe approvals | Median / p95 ms | Process VRAM GiB | Upstream |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, summary in summaries.items():
        s=summary['odysseus']; name=RUNS[label]
        memory=OUTPUT/name/'memory.json'
        vram=f"{read(memory)['peak_process_mib']/1024:.2f}" if memory.exists() else 'API'
        lines.append(f"| {label} | {s['field_correct']}/{s['fields']} ({s['field_accuracy']:.1%}) | {s['exact']} | {s['unsafe_approvals']} | {s['median_ms']:.0f} / {s['p95_ms']:.0f} | {vram} | {summary['upstream_correct']}/{summary['upstream_fields']} |")
    old=read(ROOT/'results/jev-odysseus-64-v1/summary.json')
    lines += [f"| Jev 1.13, saved | {old['field_correct']}/{old['fields']} ({old['field_accuracy']:.1%}) | {old['exact']} | {old['unsafe_approvals']} | {old['median_ms']:.0f} / {old['p95_ms']:.0f} | API | not run |", '',
        'Clef is the strongest newly tested local model on these gates. Mica uses less memory and its eight-question-sequence mode is faster in this run, but its answers change, so the single-sequence configuration remains the baseline and the parallel configuration remains a separate candidate. The same Kev weights score 183/214 on the current engine versus saved 185/214, and 237/299 upstream versus saved 239/299. Keep engine drift distinct from model changes. None of the zero unsafe-approval counts establishes production authorization safety.', '',
        'Jev 1.14 is two fields below saved 1.13: two improve and four regress. All four regressions concern whether browser access is required; the improvements concern a completion quality score and GUI ambiguity. This one-pass comparison does not separate model changes from ordinary provider variation. It does not show a blanket upgrade. The upstream results favor 1.14 over the local models here, but no matched upstream 1.13 pass exists.', '',
        '## Original upstream suites', '',
        '| Suite | '+' | '.join(summaries)+' |', '|---|'+'---:|'*len(summaries)]
    for suite in next(iter(summaries.values()))['upstream']:
        lines.append('| '+suite+' | '+' | '.join(f"{s['upstream'][suite]['correct']}/{s['upstream'][suite]['fields']}" for s in summaries.values())+' |')
    b=verification['budget']
    lines += ['', '## Billing and authorization', '',
        f"Exactly {b['attempts']} primary Jev attempts, {b['input_tokens']:,} input plus {b['output_tokens']:,} output = **{b['charged_total']:,}/5,000,000 tokens**. **{b['limit']-b['charged_total']:,} remain** in the same workspace campaign. No retries, unresolved charges or extra allowance. Input charge at $0.042/M: **${b['input_tokens']*.042/1e6:.8f}**. Local runs use no vendor tokens.", '',
        'The canonical ledger is `~/.local/state/jev-campaigns/workspace-jev114-20261015/session.json`. Its locked reservations include input and output before dispatch, with a 2M cumulative input cap per dataset. Invalid responses remain charged; unknown usage persistently halts dispatch. Current requests overlap the 64 prepared gate requests, so a future prepared replay must reuse those accounted primary identities rather than spend again.', '',
        '## Deployment and protocol boundaries', '',
        '- Clef-Flash Q4 and retained Kev Q8 use the existing df18bee CUDA/SM75 image through the shared LLM manager. Clef requires its complete joint prompt in one micro-batch: batch/ubatch 4096; Kev retains batch 1024/ubatch 512. Both use one server slot, context 4096 and F16 KV. Actual startup logs and process VRAM are retained.',
        '- Mica 0.1 Q8 uses the pinned author direct-prefill readout and shipped temperature 1.1244734010661372. No sampler, grammar or text generation. The exact author prompt, tokenizer JSON and Jinja template are retained; all 257 readout labels verify as distinct single tokens. The retained container generic fast tokenizer loads these exact bytes, with networking disabled.',
        '- The author ctypes binding predates one context field in df18bee. Our adapter inserts `moe_cache_size`; a compiled C probe against pinned runtime headers checks struct size and offsets before model loading. The extracted libraries are isolated through LD_LIBRARY_PATH. No author source bytes were changed.',
        '- Mica single-sequence mode uses 4096 context and ubatch 512. The separate eight-sequence mode uses 4096 per question and author prefix-sharing. Both are reported; neither was selected by editing gold or prompts. Mica timing is direct in-process inference; the other local paths include HTTP. All timing and VRAM comparisons are directional on this shared workstation, with other GPU users left untouched.',
        '- The ordinary upstream score remains the original distribution scorer, including lexical ties and winning-level ordinal accuracy. Odysseus uses the established scalar conversions. Jev explicit choices are retained and mismatch diagnostics are separate, never silently repaired.',
        '- Mica 0.5 is advertised in the current author repository, but the public weight API returned 401 and the public inventory lists only 0.1. No substitute is labeled 0.5. Clef image/vision projectors were not downloaded; these are text-only matched tests.',
        '- Removed model weights and the old virtualenv were not restored. Saved old results remain available. No Odysseus code, classified action, browser action, training, remote GPU or deployment ran.', '',
        '## Evidence and reproduction', '',
        'Per-model directories retain cases, source snapshots, raw safe responses, per-field scores, summaries, GPU proof and settings. `verification.json` independently recomputes metrics, checks source snapshots and matched gold, reconciles every Jev call with canonical billing, and verifies local zero-generation and container removal. Failed preflight attempts remain diagnostic evidence and are excluded from scored runs.', '',
        'Run `python scripts/report_new_decisions.py` from the project to verify and regenerate this report offline. New inference requires a fresh output path with `scripts/compare_new_decisions.py`; Jev additionally requires an explicit authorized `jev --live`. Do not repeat this cloud cohort into another directory: its accounted primary request identities are reused.', '',
        'Recovered user flow: [comparison-flow-20261015.md](../../docs/comparison-flow-20261015.md). Model/source hashes: [acquisition manifest](../../docs/downloads/new-decisions-20261015.json). Primary sources: [Clef](https://huggingface.co/Cloudflare/clef-flash), [official GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF), [Mica public weights](https://huggingface.co/sky7350/Mica-v0.1-4B), [author runtime](https://github.com/akivet/Mica), [Jev versions and price](https://docs.typesafe.ai/models).', '',
        'Small correlated synthetic and heterogeneous smoke cohorts do not establish universal model ranking or replicate private publisher leaderboards. Accuracy, probability calibration and workload safety remain distinct.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUTPUT/'report.md'), verified=True, budget=b), indent=2))


if __name__ == '__main__':
    report()
