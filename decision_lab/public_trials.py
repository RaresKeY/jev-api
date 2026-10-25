"""Offline plans and estimates; execution requires a matching hash and explicit gates."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .evaluation import ROOT, sha256, write_json
from .public_sources import BUNDLE
from .public_benchmarks import DEFAULT_SUITES, digest, load_corpus, select, summarize

MODEL_IDS = ['jev-1.14.0', 'clef-flash-q4', 'mica-01-q8', 'kev-4b-q8']
LEDGER = Path.home()/'.local/state/jev-campaigns/workspace-jev114-20261015/session.json'
RESERVATION = 131072
IMPLEMENTATION = ['decision_lab/public_benchmarks.py', 'decision_lab/public_trials.py',
    'decision_lab/public_execution.py', 'decision_lab/public_sources.py', 'scripts/public_trials.py', 'decision_lab/benchmarks.py',
    'decision_lab/jev_campaign.py', 'decision_lab/llama_decision.py', 'decision_lab/mica_native.py',
    'decision_lab/runtime.py', 'decision_lab/http.py', 'decision_lab/evaluation.py',
    'decision_lab/specialists.py', 'vendored/jevbench/scoring.py', 'vendored/jevbench/metrics.py',
    '../jev-pipelines/jev_pipelines/current_client.py', '../jev-pipelines/jev_pipelines/bounded.py',
    'docs/downloads/new-decisions-20261015.json', 'docs/downloads/kev-4b-q8.json',
    'results/new-decisions-20261015-v1/clef-flash-q4-v2/manifest.json']


def budget_snapshot():
    """Read accounting without opening keys, constructing a client or rewriting the ledger."""
    data = json.loads(LEDGER.read_text())
    if data['campaign'] != 'workspace-jev114-20261015' or data['limit'] != 5_000_000:
        raise ValueError('Unexpected shared campaign')
    rows = list(data['entries'].values())
    charged = sum(r['total_charge'] for r in rows)
    return dict(limit=data['limit'], charged=charged, remaining=data['limit']-charged,
        halted=data['halted'], unresolved=sum(r['status'] != 'settled' for r in rows),
        ledger_sha256=sha256(LEDGER), ledger_directory=str(LEDGER.parent))


def wire(case):
    """The only model-visible fields. Never pass the case or its provenance wholesale."""
    return dict(state=case['state'], questions=case['questions'])


def campaign_identity(request):
    """Match the campaign's ASCII JSON wire hash without importing its live client."""
    encoded = json.dumps(request, ensure_ascii=True, separators=(',', ':')).encode()
    return 'primary--'+hashlib.sha256(encoded).hexdigest()


def estimate(cases):
    sizes = [len(json.dumps(wire(c), ensure_ascii=False, separators=(',', ':')).encode()) for c in cases]
    lower_input = sum(math.ceil(n/5)+32 for n in sizes)
    planning_input = sum(math.ceil(n/3)+64 for n in sizes)
    upper_input = sum(math.ceil(n/2)+128 for n in sizes)
    output = sum(32+12*len(q.get('criteria', [0, 1])) for c in cases for q in c['questions'].values())
    planning_total = math.ceil(1.25*(planning_input+output))
    upper_total = math.ceil(1.25*(upper_input+output))
    historical = ROOT/'results/workspace-replay-20261015-v1/analysis.json'
    analysis = json.loads(historical.read_text()) if historical.is_file() else None
    timings = {}
    for model in MODEL_IDS if analysis else ():
        key = 'kev-4b-q8-large-batch' if model == 'kev-4b-q8' else model
        observed = analysis['runtime_resources'][key]
        timings[model] = dict(minutes_median_scenario=len(cases)*observed['median_ms']/60_000,
            minutes_slow_scenario=len(cases)*2*observed['p95_ms']/60_000)
    return dict(requests=len(cases), questions=sum(len(c['questions']) for c in cases),
        requests_by_suite=dict(Counter(c['suite'] for c in cases)),
        provider_input_tokens=dict(lower=lower_input, planning=planning_input, upper=upper_input),
        provider_output_tokens_planning=output, total_tokens_with_25pct_headroom=planning_total,
        upper_total_tokens_with_headroom=upper_total,
        input_cost_usd=dict(lower=lower_input*.042/1_000_000,
            planning=planning_input*.042/1_000_000, upper=upper_input*.042/1_000_000),
        price=dict(model='jev-1.14.0', usd_per_million_input=.042, output_free=True,
                   verified_date='2026-10-15', source='https://docs.typesafe.ai/models'),
        runtime_minutes=timings or None, native_three_models_minutes=dict(
            median_scenario=sum(timings[m]['minutes_median_scenario'] for m in MODEL_IDS if m != 'jev-1.14.0'),
            slow_scenario=sum(timings[m]['minutes_slow_scenario'] for m in MODEL_IDS if m != 'jev-1.14.0')) if timings else None,
        assumptions=['Token estimates use UTF-8 wire bytes / 5, / 3 or / 2 plus framing, not provider tokenization.',
            '25% headroom includes estimated input and output; global live reservations remain authoritative.',
            'Runtime scenarios require local historical results; absent evidence is reported as null.',
            'When available, scenarios multiply historical common-cohort median and twice p95 by request count.',
            'Sequential, shared GPU; these are planning scenarios, not statistical run-time bounds.',
            'Model startup, prompt-length differences, thermal/load changes and manual work are additional.',
            'Local inference has no API token fee; electricity and hardware are excluded.'])


def make_plan(output, suites=None, limit=12, seed='public-trials-20261015', families=None,
              full=False, include_development=False):
    if (limit == 0) != full:
        raise ValueError('Full selection requires both --full and --limit-per-suite 0')
    selected_suites = suites or DEFAULT_SUITES
    if any(s.endswith(('-development', '-validation')) for s in selected_suites) and not include_development:
        raise ValueError('Development/validation selection requires --include-development')
    cases = select(load_corpus(), selected_suites, limit, seed, families)
    plan = dict(schema='gated-public-trial/v1', selection=dict(suites=selected_suites, limit_per_suite=limit,
        seed=seed, families=families, full=full, include_development=include_development,
        grouping='Whole source groups, balanced round robin over suite/family; no labels used in selection'),
        models=MODEL_IDS, cases=cases, cases_sha256=digest(cases),
        corpus_catalog_sha256=sha256(BUNDLE/'catalog.json'),
        implementation={f: sha256(ROOT/f) for f in IMPLEMENTATION},
        estimate=estimate(cases), budget_at_preparation=budget_snapshot(),
        authorization=dict(inference_executed=False, paid_calls_authorized_by_preparation=False,
            required=['explicit run --execute', '--approve-sha256 of this plan', 'explicit model',
                      '--max-requests', 'Jev additionally --live and --max-new-tokens'],
            campaign='Existing shared 5M total, never a new allowance', retries=False, actions=False))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return dict(plan=str(output), sha256=sha256(output), estimate=plan['estimate'])


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan.get('schema') != 'gated-public-trial/v1' or digest(plan['cases']) != plan['cases_sha256']:
        raise ValueError('Invalid plan or changed selected cases')
    if sha256(BUNDLE/'catalog.json') != plan['corpus_catalog_sha256']:
        raise ValueError('Corpus identity changed since planning')
    for relative, expected in plan['implementation'].items():
        if relative not in IMPLEMENTATION or sha256(ROOT/relative) != expected:
            raise ValueError('Implementation changed; create a fresh reviewed plan')
    if set(plan['implementation']) != set(IMPLEMENTATION):
        raise ValueError('Incomplete implementation binding')
    # Check selected rows against the source-verified corpus, not only against a self-signed hash.
    corpus = {c['id']: c for rows in load_corpus().values() for c in rows}
    if len({c['id'] for c in plan['cases']}) != len(plan['cases']):
        raise ValueError('Duplicate planned case IDs')
    for case in plan['cases']:
        if corpus.get(case['id']) != case:
            raise ValueError('Planned case differs from imported original')
    return plan


def check_gates(path, plan, model, execute, approved, max_requests, live=False, max_new_tokens=None):
    if not execute or approved != sha256(path):
        raise ValueError('Execution requires --execute and the exact reviewed --approve-sha256')
    if model not in MODEL_IDS or model not in plan['models']:
        raise ValueError('Select an admitted model explicitly')
    if type(max_requests) is not int or max_requests < len(plan['cases']):
        raise ValueError('Explicit --max-requests must cover the selected plan')
    if model == 'jev-1.14.0':
        budget = budget_snapshot()
        if not live or type(max_new_tokens) is not int or max_new_tokens < RESERVATION:
            raise ValueError('Paid calls require --live and --max-new-tokens >= 131072 for one reservation')
        if budget['halted'] or budget['unresolved'] or max_new_tokens > budget['remaining']:
            raise ValueError('Shared campaign halted, unresolved or insufficient for the requested cap')
        if plan['estimate']['upper_total_tokens_with_headroom'] > max_new_tokens:
            raise ValueError('Estimated upper usage exceeds this run cap; prepare a smaller subset')
    elif live or max_new_tokens is not None:
        raise ValueError('Cloud authorization flags do not apply to local models')


def run_client(client, plan, output, model, max_new_tokens=None):
    """One attempt per selected row, errors retained. Callable with an offline fixture client."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', model=model, selection=plan['selection'],
        cases_sha256=plan['cases_sha256'], planned_requests=len(plan['cases']),
        no_actions=True, retries=False)
    write_json(output/'manifest.json', manifest)
    rows, spent, fatal = [], 0, None
    before_entries = set(json.loads(LEDGER.read_text())['entries']) if model == 'jev-1.14.0' else set()
    try:
        with (output/'responses.jsonl').open('x') as handle:
            for case in plan['cases']:
                if model == 'jev-1.14.0' and spent+RESERVATION > max_new_tokens:
                    manifest['stop_reason'] = 'Per-run reservation ceiling; remaining selected rows stay missing'
                    break
                row = dict(id=case['id'])
                client.last_raw = None
                start = time.perf_counter()
                try:
                    if model == 'jev-1.14.0':
                        client.dataset = case['suite']
                    row['distributions'] = client.distributions(**wire(case))
                except Exception as error:
                    row['error'] = type(error).__name__
                    if type(error).__name__ == 'FatalCloudError':
                        fatal = error
                row.update(raw=client.last_raw, elapsed_ms=(time.perf_counter()-start)*1000)
                if model == 'jev-1.14.0' and row['raw']:
                    # Conservative run bound includes cached responses, which cost no new vendor tokens.
                    usage = row['raw'].get('usage', {})
                    spent += usage.get('input_tokens', 0)+usage.get('output_tokens', 0)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False)+'\n'); handle.flush()
                if fatal:
                    break
    finally:
        summary = summarize(plan['cases'], rows)
        manifest.update(status='complete' if len(rows) == len(plan['cases']) and not fatal else 'incomplete',
            completed_requests=len(rows), pending_requests=len(plan['cases'])-len(rows),
            accounted_response_tokens_conservative=spent)
        if model == 'jev-1.14.0':
            manifest['campaign_after'] = budget_snapshot()
            entries = json.loads(LEDGER.read_text())['entries']
            identities = {campaign_identity(dict(model=model, **wire(c))) for c in plan['cases']}
            manifest['new_paid_tokens'] = sum(v['total_charge'] for k, v in entries.items()
                                             if k in identities and k not in before_entries)
        else:
            manifest['new_paid_tokens'] = 0
        write_json(output/'summary.json', summary)
        write_json(output/'manifest.json', manifest)
    if fatal:
        raise fatal
    return summary
