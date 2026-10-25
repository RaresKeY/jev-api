"""Frozen workspace replays. Gold stays local; no project or browser actions."""
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from .evaluation import ROOT, sha256

WORKSPACE = ROOT.parent
PIPELINES = WORKSPACE / 'jev-pipelines'
CAMPAIGN = 'workspace-jev114-20261015'
DIRECTORY = ROOT / '.work/workspace-replay-20261015-v1'
PLAN = DIRECTORY / 'plan.json'
PUBLIC = ROOT / 'results/workspace-replay-20261015-v1'
OLD_RUNS = ROOT / 'results/new-decisions-20261015-v1'
MODELS = ('jev-1.14.0', 'clef-flash-q4', 'mica-01-q8', 'kev-4b-q8')
OLD_LOCAL = {'jev-1.14.0':'jev-1.14.0', 'clef-flash-q4':'clef-flash-q4-v2',
             'mica-01-q8':'mica-01-q8-v2', 'kev-4b-q8':'kev-4b-q8-control'}
NORMALIZE = dict(ok='consistent', not_ok='inconsistent', wrong='inconsistent',
                 uncertain='insufficient', consistent='consistent',
                 inconsistent='inconsistent', insufficient='insufficient')


def encode(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    import hashlib
    return hashlib.sha256(encode(value)).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.chmod(0o600)
    temp.replace(path)


class Sources:
    def __init__(self):
        self.hashes = {}
        self.cache = {}

    def read(self, path):
        path = Path(path).resolve()
        key = str(path.relative_to(WORKSPACE))
        if key not in self.cache:
            self.hashes[key] = sha256(path)
            self.cache[key] = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                               if path.suffix == '.jsonl' else json.loads(path.read_text()))
        return self.cache[key]


def record_at(sources, job):
    value = sources.read(WORKSPACE / job['source'])
    pointer = job['pointer']
    if pointer.startswith('line:'):
        number, _, rest = pointer[5:].partition('/')
        value = value[int(number)-1]
        pointer = '/' + rest if rest else ''
    for token in pointer.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def prepare():
    # Run the owning offline integrity check before deriving scoring metadata.
    import sys
    sys.path.insert(0, str(PIPELINES))
    import prepare_upgrade
    prepare_upgrade.check()
    sources = Sources()
    bundle = sources.read(PIPELINES/'.work'/CAMPAIGN/'requests.json')
    original_plan = sources.read(PIPELINES/'plans/experiment-v1.json')
    followup = sources.read(PIPELINES/'plans/feasibility-v2.json')
    public_suite = sources.read(PIPELINES/'data/suite-v1.json')
    public_gold = {r['uid']:r['gold'] for rows in public_suite.values() for r in rows}
    public_gold.update({r['uid']:r['gold'] for r in sources.read(PIPELINES/'data/dependency-v1.json')})
    public_gold.update({r['uid']:r['gold'] for r in followup['rows']})
    v1 = {j['id']:j for j in original_plan['fixed_jobs']+original_plan['search_nodes']}
    v2 = {j['id']:j for j in followup['jobs']}
    anli = {r['uid']:{'e':'entailment','c':'contradiction','n':'neutral'}[r['label']]
            for rows in sources.read(WORKSPACE/'jev-eval/data/suite.json').values() for r in rows}
    clean = {r['uid']:r['label'] for splits in sources.read(WORKSPACE/'jev-eval/data/clean-v1/suite.json').values()
             for rows in splits.values() for r in rows}
    fleet = {}
    for path in sorted((PIPELINES/'results').glob('fleet-refinement-20261014*/**/plan.json')):
        for job in sources.read(path).get('jobs', []):
            key = digest(job['request'])
            mapping = {q:dict(kind='choice', expected=[r['gold']], mapping=NORMALIZE,
                              label_kind='mechanical', split=r['split'], family=r['family'])
                       for q,r in job['mapping'].items()}
            if key in fleet and fleet[key] != mapping:
                raise RuntimeError('Ambiguous fleet scoring provenance')
            fleet[key] = mapping
    jobs = []
    occurrences = Counter()
    for item in bundle['jobs']:
        job = deepcopy(item)
        group, meta, qs = job['group'], job['baseline_metadata'], job['request']['questions']
        expected = {}
        strata = {k:meta[k] for k in ('variant','family','split','label_kind') if k in meta}
        if group == 'odysseus-gates':
            original = record_at(sources, job)
            expected = {q:dict(kind=qs[q]['type'], expected=[v] if type(v) is not bool else v,
                              mode='odysseus', label_kind='mechanical')
                        for q,v in original['case']['expected'].items()}
            strata['gate'] = original['case']['gate']
            job['odysseus_id'] = original['id']
        elif group in ('supper-calibration', 'supper-prerequisites') and 'expected' in meta:
            if len(qs) != 1:
                raise RuntimeError('Original Supper Guard independent decision changed')
            q = next(iter(qs))
            expected[q] = dict(kind=qs[q]['type'], expected=meta['expected'],
                mapping=meta.get('option_mapping', {}), label_kind=meta.get('label_kind', 'mechanical'),
                mode='ordinal' if qs[q]['type']=='score' else 'ordinary')
        elif group in ('anli','hans','paws'):
            gold = (anli if group == 'anli' else clean)[meta['uid']]
            expected['relationship'] = dict(kind='choice', expected=[gold], label_kind='published')
            for q in qs:
                if q.startswith('valid_'):
                    expected[q] = dict(kind='noul', expected=q[6:]==gold, label_kind='published-check')
        elif group == 'fleet-refinement':
            request = dict(job['request'], model=job['baseline_requested_model'])
            expected = fleet[digest(request)]
        elif group == 'pipeline-feasibility-batching':
            saved = record_at(sources, job)
            ident = saved['identity'].split('--',1)[1]
            if '/feasibility-v2/' in job['source']:
                old = v2[ident]
                expected['next'] = dict(kind='choice', expected=[public_gold[old['uid']]], label_kind='mechanical')
                strata.update(variant=old['variant'], split=old['split'], dataset='dependency')
            else:
                old = v1[ident]
                strata.update(variant=old.get('condition','search'), dataset=old['dataset'])
                if 'mapping' in old:
                    for q,(uid,check) in old['mapping'].items():
                        gold = public_gold[uid]
                        expected[q] = dict(kind=qs[q]['type'], expected=(check[6:]==gold if check.startswith('valid_') else [gold]),
                            label_kind='mechanical' if old['dataset']=='dependency' else 'published')
                else:
                    records = old['request']['state']['records']
                    values = list(records.values()) if isinstance(records,dict) else records
                    good = all(r['proposed_label']==public_gold[uid] for r,uid in zip(values,old['record_ids'],strict=True))
                    expected['all_valid'] = dict(kind='noul', expected=good, label_kind='published-check')
            old_request = dict(old['request'], model=job['request']['model'])
            if encode(old_request) != encode(job['request']):
                raise RuntimeError('Pipeline scoring plan differs from frozen request')
        elif group == 'browser-decisions':
            saved = record_at(sources, job)
            strata['purpose'] = saved.get('purpose', 'action' if 'action' in qs else 'detector')
        job['scoring'] = expected
        job['strata'] = strata
        # Exact historical repeat trials remain separate; completed primary gates are reused.
        key = job['request_sha256']
        occurrence = occurrences[key]
        occurrences[key] += 1
        job['cloud_identity'] = ('primary--'+key if occurrence==0 else 'repeat--'+job['job_id'])
        job['dataset'] = group if group not in ('anli','hans','paws') else group
        if group == 'pipeline-feasibility-batching':
            job['dataset'] = strata['dataset'].replace('hans-search','hans')
        jobs.append(job)
    plan = dict(schema='workspace-model-replay/v1', campaign=CAMPAIGN, models=list(MODELS),
        authorization='Explicit replay scope: all frozen workload groups, including browser decisions, across the declared models.',
        no_actions=True, no_retries=True, no_truncation=True, context=16384,
        clef_context=12288,
        mica_sequences=1, sources=sources.hashes, jobs=jobs,
        scoring='Original local gold and rubrics. No new labels or tuning. Diagnostics have no correctness score.')
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise RuntimeError('Existing frozen workspace replay changed')
        return plan
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    DIRECTORY.chmod(0o700)
    save(PLAN, plan)
    for path in [Path(__file__), ROOT/'scripts/replay_workspace.py', ROOT/'decision_lab/mica_native.py',
                 ROOT/'decision_lab/jev_campaign.py']:
        dest=DIRECTORY/'source-snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path,dest)
    coverage={g:dict(requests=sum(j['group']==g for j in jobs),
        gold_fields=sum(len(j['scoring']) for j in jobs if j['group']==g),
        diagnostic_fields=sum(len(j['request']['questions'])-len(j['scoring']) for j in jobs if j['group']==g))
        for g in dict.fromkeys(j['group'] for j in jobs)}
    PUBLIC.mkdir(parents=True, exist_ok=True)
    save(PUBLIC/'coverage.json',dict(plan_sha256=sha256(PLAN), coverage=coverage, context=plan['context'],
        models=list(MODELS), original_requests=len(jobs), private_evidence=str(DIRECTORY),
        notes=['Completed Odysseus results reused.', 'Gold and source metadata stay local.',
               'Repeated old trials remain repeated new trials. One shared 5M API budget.']))
    return plan


def load_plan():
    plan=json.loads(PLAN.read_text())
    for path,expected in plan['sources'].items():
        if sha256(WORKSPACE/path) != expected:
            raise RuntimeError('Frozen source changed: '+path)
    for name in ('decision_lab/workspace_replay.py','scripts/replay_workspace.py',
                 'decision_lab/mica_native.py','decision_lab/jev_campaign.py'):
        if sha256(ROOT/name) != sha256(DIRECTORY/'source-snapshot'/name):
            raise RuntimeError('Frozen executor changed: '+name)
    for j in plan['jobs']:
        if digest(j['request']) != j['request_sha256']:
            raise RuntimeError('Frozen request changed')
    return plan


def normalized(client, request):
    distributions=client.distributions(request['state'],request['questions'])
    raw=client.last_raw
    answers={}
    for name,q in request['questions'].items():
        p=distributions[name]
        original=raw.get('answers',{}).get(name,{})
        a=dict(type=q['type'])
        if q['type']=='noul':
            a['noul']=p['true']
        else:
            a['probabilities']=p
            a['confidence']=original.get('confidence',max(p.values()))
            if q['type']=='choice':a['choice']=original.get('choice',max(p,key=p.get))
            else:a['score']=original.get('score',sum(int(k)*v for k,v in p.items()))
        answers[name]=a
    return answers


def field_score(answer, spec):
    kind, expected=spec['kind'],spec['expected']
    invalid=answer is None
    probabilities={}
    if invalid: prediction=None
    elif kind=='noul':
        prediction=answer['noul']>=.5
        probabilities={False:1-answer['noul'],True:answer['noul']}
    elif kind=='score':
        probabilities={int(k):v for k,v in answer['probabilities'].items()}
        prediction=(int(max(answer['probabilities'],key=answer['probabilities'].get))
                    if spec.get('mode')=='ordinal' else answer['score'])
    else:
        mapping=spec.get('mapping',{})
        prediction=mapping.get(answer['choice'],answer['choice'])
        for k,v in answer['probabilities'].items():
            label=mapping.get(k,k)
            probabilities[label]=probabilities.get(label,0)+v
    expected_labels=[expected] if type(expected) is bool else expected
    correct=(False if invalid else abs(prediction-expected[0])<=.75
             if kind=='score' and spec.get('mode')=='odysseus' else prediction in expected_labels)
    mass=None if invalid else sum(probabilities.get(v,0) for v in expected_labels)
    brier=None
    if not invalid and len(expected_labels)==1 and spec.get('label_kind')!='judgment':
        brier=(sum((p-int(k==expected_labels[0]))**2 for k,p in probabilities.items())
               if kind!='noul' else (answer['noul']-int(expected))**2)
    return dict(prediction=prediction,correct=correct,expected_mass=mass,brier=brier,
                label_kind=spec.get('label_kind','mechanical'),invalid=invalid)


def validate_answers(answers, request, strict=False):
    if set(answers)!=set(request['questions']):raise ValueError('Answer IDs differ')
    for name,q in request['questions'].items():
        a=answers[name]
        if a.get('type')!=q['type']:raise ValueError('Answer type differs')
        if q['type']=='noul':
            if type(a.get('noul')) not in (float,int) or not math.isfinite(a['noul']) or not 0<=a['noul']<=1:
                raise ValueError('Invalid Noul')
        else:
            p=a['probabilities']
            keys=set(q['criteria']) if q['type']=='choice' else {str(i) for i in range(len(q['criteria']))}
            if set(p)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p.values()) or abs(sum(p.values())-1)>.025:
                raise ValueError('Invalid distribution')
            if q['type']=='choice' and (a['choice'] not in p or strict and p[a['choice']]+1e-9<max(p.values())):
                raise ValueError('Invalid explicit choice')


def reuse_gates(model, jobs):
    path=OLD_RUNS/OLD_LOCAL[model]/'odysseus-responses.jsonl'
    saved={r['id']:r for r in (json.loads(line) for line in path.read_text().splitlines())}
    result={}
    for j in jobs:
        if j['group']!='odysseus-gates':continue
        old=saved[j['odysseus_id']]
        if encode(old['case']['state'])!=encode(j['request']['state']) or encode(old['case']['questions'])!=encode(j['request']['questions']):
            raise RuntimeError('Completed gate input changed')
        result[j['job_id']]=dict(job_id=j['job_id'],group=j['group'],request_sha256=j['request_sha256'],
            reused=True,source=str(path.relative_to(ROOT)),source_sha256=sha256(path),
            answers=old['raw']['answers'],raw=old['raw'],latency_seconds=old['latency_ms']/1000,
            error=old.get('error'))
    return result


def execute(client, model, plan, live=False):
    output=DIRECTORY/model
    output.mkdir(parents=True,exist_ok=True)
    output.chmod(0o700)
    gate_rows=reuse_gates(model,plan['jobs'])
    started=time.time()
    for i,j in enumerate(plan['jobs']):
        path=output/(j['job_id']+'.json')
        if path.exists():
            old=json.loads(path.read_text())
            if old['request_sha256']!=j['request_sha256']:raise RuntimeError('Saved replay input changed')
            if old.get('status')!='complete':
                raise RuntimeError('Unresolved recorded attempt must not be retried automatically')
            continue
        row=dict(job_id=j['job_id'],group=j['group'],request_sha256=j['request_sha256'])
        if j['job_id'] in gate_rows:
            row=gate_rows[j['job_id']]
        else:
            # Persist intent before inference. A crash is not an automatic retry.
            save(path,dict(**row,status='attempted'))
            start=time.perf_counter()
            try:
                if model=='jev-1.14.0':
                    client.dataset=j['dataset']
                    raw=client.send(j['request']['state'],j['request']['questions'],identity=j['cloud_identity'])
                    answers=raw['answers']
                else:
                    answers=normalized(client,j['request'])
                    raw=client.last_raw
                row.update(answers=answers,raw=raw)
                validate_answers(answers,j['request'],strict=j['group']=='fleet-refinement')
            except Exception as error:
                from .errors import FatalCloudError
                row['error']=type(error).__name__
                row['raw']=getattr(client,'last_raw',None)
                if isinstance(error,FatalCloudError):
                    row.update(status='halted',latency_seconds=time.perf_counter()-start)
                    save(path,row)
                    raise
            row['latency_seconds']=time.perf_counter()-start
        row['status']='complete'
        save(path,row)
        if (i+1)%25==0 or i+1==len(plan['jobs']):
            print(json.dumps(dict(model=model,completed=i+1,total=len(plan['jobs']),group=j['group'],
                elapsed_seconds=round(time.time()-started),error=row.get('error'))),flush=True)
    save(output/'execution.json',dict(model=model,complete=True,requests=len(plan['jobs']),
        plan_sha256=sha256(PLAN),finished_at=time.time(),elapsed_seconds=time.time()-started,
        budget=client.ledger.summary() if model=='jev-1.14.0' else None))


def report():
    plan=load_plan()
    ledger_dir=Path.home()/'.local/state/jev-campaigns'/CAMPAIGN
    ledger=json.loads((ledger_dir/'session.json').read_text())
    entries=list(ledger['entries'].values())
    budget=dict(limit=ledger['limit'],attempts=len(entries),halted=ledger['halted'],
        input_tokens=sum(e['input_tokens'] for e in entries),output_tokens=sum(e['output_tokens'] for e in entries),
        charged_total=sum(e['total_charge'] for e in entries),unresolved=sum(e['status']!='settled' for e in entries))
    if budget['charged_total']>5_000_000:raise RuntimeError('Canonical allowance exceeded')
    summaries={}
    metrics={}
    verification=dict(plan_sha256=sha256(PLAN),sources_checked=len(plan['sources']),models={})
    for model in ('jev-1.13.0',)+MODELS:
        groups=defaultdict(list)
        detailed=[]
        errors=Counter()
        for j in plan['jobs']:
            baseline=j['baseline_response']['answers']
            error=None
            if model=='jev-1.13.0':
                answers=baseline
                validate_answers(answers,j['request'],strict=j['group']=='fleet-refinement')
            else:
                path=DIRECTORY/model/(j['job_id']+'.json')
                if not path.exists():continue
                saved=json.loads(path.read_text())
                if saved['request_sha256']!=j['request_sha256']:raise RuntimeError('Replay hash mismatch')
                if saved.get('status')!='complete':continue
                if model=='jev-1.14.0':
                    identity=j['cloud_identity']
                    canonical=json.loads((ledger_dir/'calls'/(identity+'.json')).read_text())
                    entry=ledger['entries'][identity]
                    if (encode(canonical['request'])!=encode(j['request']) or
                        canonical['response']!=saved['raw'] or entry['status']!='settled' or
                        entry['request_sha256']!=j['request_sha256'] or
                        any(entry[k]!=saved['raw']['usage'][k] for k in ('input_tokens','output_tokens')) or
                        entry['total_charge']!=sum(saved['raw']['usage'][k] for k in ('input_tokens','output_tokens'))):
                        raise RuntimeError('Cloud replay does not match canonical paid evidence')
                elif not saved.get('error') and saved['raw']['usage']['output_tokens']!=0:
                    raise RuntimeError('Local decision model unexpectedly generated output')
                answers=saved.get('answers',{})
                error=saved.get('error')
                if not error:validate_answers(answers,j['request'],strict=j['group']=='fleet-refinement')
            fields={q:field_score(None if error else answers.get(q),s) for q,s in j['scoring'].items()}
            oldfields={q:field_score(baseline.get(q),s) for q,s in j['scoring'].items()}
            agreement=0
            for q,question in j['request']['questions'].items():
                a=answers.get(q) if not error else None
                b=baseline[q]
                kind=question['type']
                if a:
                    agreement+=(a['choice']==b['choice'] if kind=='choice' else
                                (a['noul']>=.5)==(b['noul']>=.5) if kind=='noul' else
                                abs(a['score']-b['score'])<=.75)
            entry=dict(job_id=j['job_id'],group=j['group'],strata=j['strata'],fields=fields,
                correct=sum(f['correct'] for f in fields.values()),gold_fields=len(fields),
                agreement=agreement,questions=len(j['request']['questions']),error=error,
                improved=sum(f['correct'] and not oldfields[q]['correct'] for q,f in fields.items()),
                regressed=sum(not f['correct'] and oldfields[q]['correct'] for q,f in fields.items()))
            groups[j['group']].append(entry)
            detailed.append(entry)
            if error:errors[j['group']]+=1
        result={}
        for group,rows in groups.items():
            result[group]=aggregate(rows)
            by_stratum={}
            for key in sorted({k for r in rows for k in r['strata']}):
                by_stratum[key]={str(v):aggregate([r for r in rows if r['strata'].get(key)==v])
                                 for v in sorted({r['strata'].get(key) for r in rows},key=str)}
            result[group]['strata']=by_stratum
        summaries[model]=result
        metrics[model]=detailed
        verification['models'][model]=dict(completed=sum(len(r) for r in groups.values()),errors=dict(errors))
    save(PUBLIC/'summary.json',summaries)
    # Per-job IDs and scores contain no request text or source payloads.
    save(PUBLIC/'paired-metrics.json',metrics)
    save(PUBLIC/'verification.json',verification)
    save(PUBLIC/'budget.json',dict(**budget,remaining=budget['limit']-budget['charged_total'],
        input_cost_usd=budget['input_tokens']*.042/1_000_000))
    columns=('jev-1.13.0',)+MODELS
    lines=['# Prepared workspace model comparison','',
        'Matched saved requests and original local labels. Old Jev results are saved baselines. '
        'Models receive unchanged state/questions. No browser actions, game execution, prompt tuning or new thresholds.','',
        '| Workload | Old Jev 1.13 | Jev 1.14 | Clef 9B Q4 | Mica 0.1 4B Q8 | Kev 4B Q8 |',
        '|---|---:|---:|---:|---:|---:|']
    for group in dict.fromkeys(j['group'] for j in plan['jobs']):
        cells=[]
        for model in columns:
            r=summaries[model].get(group)
            cells.append('pending' if not r else f"{r['correct']}/{r['gold_fields']}" if r['gold_fields'] else
                         f"{r['agreement']}/{r['questions']} agree")
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines+=['','Correctness cells count scored fields. Cells marked agree have no gold and show decision agreement '
        'with old Jev, not task success. Supper calibration separates mechanical answers from authored preferences '
        'in summary.json. All results include errors in their denominators; repeated trials and batched questions '
        'are correlated, not independent examples. Context limits are 12,288 tokens for Clef and 16,384 for the other new local replays; '
        'the completed 4,096-context Odysseus runs are reused. No evidence is truncated.','']
    lines += [f"Canonical paid usage: {budget['charged_total']:,}/5,000,000 total tokens; "
              f"{budget['limit']-budget['charged_total']:,} remain. {budget['unresolved']} unresolved attempts. "
              'All costs include the earlier 64/228 comparison. No fresh allowance.','']
    (PUBLIC/'report.md').write_text('\n'.join(lines))
    print(json.dumps(verification),flush=True)
    return summaries


def aggregate(rows):
    fields=[f for r in rows for f in r['fields'].values()]
    return dict(requests=len(rows),errors=sum(bool(r['error']) for r in rows),
        gold_fields=sum(r['gold_fields'] for r in rows),correct=sum(r['correct'] for r in rows),
        agreement=sum(r['agreement'] for r in rows),questions=sum(r['questions'] for r in rows),
        improved=sum(r['improved'] for r in rows),regressed=sum(r['regressed'] for r in rows),
        mean_expected_mass=statistics.mean([f['expected_mass'] for f in fields if f['expected_mass'] is not None])
            if any(f['expected_mass'] is not None for f in fields) else None,
        mean_brier=statistics.mean([f['brier'] for f in fields if f['brier'] is not None])
            if any(f['brier'] is not None for f in fields) else None,
        by_label_kind={k:dict(fields=sum(f['label_kind']==k for f in fields),
            correct=sum(f['correct'] for f in fields if f['label_kind']==k)) for k in sorted({f['label_kind'] for f in fields})})
