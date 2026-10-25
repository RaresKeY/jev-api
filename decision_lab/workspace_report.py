"""Offline report over immutable workspace inference and canonical billing."""
from collections import defaultdict
import json
from pathlib import Path

from . import workspace_replay as replay


def base_report(plan, models=None):
    ledger_dir=Path.home()/'.local/state/jev-campaigns'/replay.CAMPAIGN
    ledger=json.loads((ledger_dir/'session.json').read_text())
    entries=list(ledger['entries'].values())
    budget=dict(limit=ledger['limit'],attempts=len(entries),halted=ledger['halted'],
        input_tokens=sum(e['input_tokens'] for e in entries),output_tokens=sum(e['output_tokens'] for e in entries),
        charged_total=sum(e['total_charge'] for e in entries),unresolved=sum(e['status']!='settled' for e in entries))
    if budget['charged_total']>5_000_000 or budget['unresolved']:
        raise RuntimeError('Canonical allowance is exceeded or unresolved')
    verification=dict(plan_sha256=replay.sha256(replay.PLAN),sources_checked=len(plan['sources']),
        models={},outer_envelope_key_order_changes=[])
    metrics={}
    for model in ('jev-1.13.0',)+tuple(models or replay.MODELS):
        rows=[]
        for j in plan['jobs']:
            baseline=j['baseline_response']['answers'];error=None
            if model=='jev-1.13.0':answers=baseline
            else:
                path=replay.DIRECTORY/model/(j['job_id']+'.json')
                if not path.exists():continue
                saved=json.loads(path.read_text())
                if saved['request_sha256']!=j['request_sha256']:raise RuntimeError('Replay job binding changed')
                if saved.get('status')!='complete':continue
                answers=saved.get('answers',{});error=saved.get('error')
                if model=='jev-1.14.0':
                    identity=j['cloud_identity']
                    canonical=json.loads((ledger_dir/'calls'/(identity+'.json')).read_text())
                    request=canonical['request'];entry=ledger['entries'][identity]
                    # The comparator builds model/state/questions at the envelope.
                    # Nineteen old saved envelopes used model/questions/state.
                    # Bind the ACTUAL wire hash and require exact ordered content.
                    if (request!=j['request'] or
                        any(replay.encode(request[k])!=replay.encode(j['request'][k]) for k in ('state','questions')) or
                        canonical['response']!=saved['raw'] or entry['status']!='settled' or
                        saved['raw'].get('model')!='jev-1.14.0' or
                        entry['request_sha256']!=replay.digest(request) or entry['dataset']!=canonical['dataset'] or
                        any(entry[k]!=saved['raw']['usage'][k] for k in ('input_tokens','output_tokens')) or
                        entry['total_charge']!=sum(saved['raw']['usage'][k] for k in ('input_tokens','output_tokens'))):
                        raise RuntimeError('Cloud replay differs from canonical request, response or billing')
                    if replay.encode(request)!=replay.encode(j['request']):
                        if list(request)==list(j['request']):raise RuntimeError('Unexpected wire ordering change')
                        verification['outer_envelope_key_order_changes'].append(j['job_id'])
                elif not error and saved['raw']['usage']['output_tokens']!=0:
                    raise RuntimeError('Unexpected local text generation')
            if not error:replay.validate_answers(answers,j['request'],strict=j['group']=='fleet-refinement')
            fields={q:replay.field_score(None if error else answers.get(q),s) for q,s in j['scoring'].items()}
            oldfields={q:replay.field_score(baseline.get(q),s) for q,s in j['scoring'].items()}
            agreement=0
            for q,question in j['request']['questions'].items():
                a=answers.get(q) if not error else None;b=baseline[q];kind=question['type']
                if a:agreement+=(a['choice']==b['choice'] if kind=='choice' else
                    (a['noul']>=.5)==(b['noul']>=.5) if kind=='noul' else abs(a['score']-b['score'])<=.75)
            rows.append(dict(job_id=j['job_id'],group=j['group'],strata=j['strata'],fields=fields,
                correct=sum(f['correct'] for f in fields.values()),gold_fields=len(fields),
                agreement=agreement,questions=len(j['request']['questions']),error=error,
                improved=sum(f['correct'] and not oldfields[q]['correct'] for q,f in fields.items()),
                regressed=sum(not f['correct'] and oldfields[q]['correct'] for q,f in fields.items())))
        metrics[model]=rows
        verification['models'][model]=dict(completed=len(rows),errors=sum(bool(r['error']) for r in rows))
    replay.save(replay.PUBLIC/'verification.json',verification)
    replay.save(replay.PUBLIC/'budget.json',dict(**budget,remaining=budget['limit']-budget['charged_total'],
        input_cost_usd=budget['input_tokens']*.042/1_000_000))
    return metrics
