"""Offline, payload-free analysis of the authorized workspace model replays.

Standalone dependencies: Private .work/ cohort, ignored results, original campaign ledger and sibling policy evidence.
See docs/standalone-dependencies.md.
"""
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab import workspace_replay as replay
from decision_lab.workspace_report import base_report


def report_models():
    extra=('kev-4b-q8-large-batch',) if (replay.PUBLIC/'kev-capacity-profile.json').exists() else ()
    return replay.MODELS+extra


def capacity_profile():
    path=replay.PUBLIC/'kev-capacity-profile-4096.json'
    if not path.exists():path=replay.PUBLIC/'kev-capacity-profile.json'
    return json.loads(path.read_text())


def capacity_errors(model):
    path=replay.DIRECTORY/model/'runtime/server.log'
    values=[];limits=[]
    if path.exists():
        pattern=re.compile(r'^[0-9.]+ E srv\s+send_error:.*error: (?:the question and its options|input) \((\d+) tokens\).*\(current batch size: (\d+)\)')
        with path.open(errors='replace') as handle:
            for line in handle:
                match=pattern.search(line)
                if match:values.append(int(match[1]));limits.append(int(match[2]))
    return dict(native_capacity_rejections=len(values),minimum_rejected_tokens=min(values) if values else None,
        maximum_rejected_tokens=max(values) if values else None,physical_limits=sorted(set(limits)))


def selected(plan, group, *, split=None, label=None, question=None, dataset=None):
    result=[]
    for j in plan['jobs']:
        if j['group']!=group:continue
        if split and j['strata'].get('split')!=split:continue
        if dataset and j['strata'].get('dataset')!=dataset:continue
        for q,s in j['scoring'].items():
            if label and s['label_kind']!=label:continue
            if question and q!=question:continue
            result.append((j['job_id'],q))
    return result


def score_pairs(keys, rows):
    observed=[rows[j]['fields'][q] for j,q in keys if j in rows]
    return dict(expected=len(keys),completed=len(observed),correct=sum(f['correct'] for f in observed),
        brier=statistics.mean([f['brier'] for f in observed if f['brier'] is not None])
              if any(f['brier'] is not None for f in observed) else None,
        expected_mass=statistics.mean([f['expected_mass'] for f in observed if f['expected_mass'] is not None])
              if any(f['expected_mass'] is not None for f in observed) else None)


def distributions(answer):
    return {'false':1-answer['noul'],'true':answer['noul']} if answer['type']=='noul' else answer['probabilities']


def divergence(first, second):
    # Rounded provider probabilities can sum to .99 or 1.01. Normalize only
    # this distance diagnostic; advertised choices and scoring stay untouched.
    first={k:v/sum(first.values()) for k,v in first.items()}
    second={k:v/sum(second.values()) for k,v in second.items()}
    midpoint={k:(first[k]+second[k])/2 for k in first}
    return sum(p*math.log2(p/midpoint[k])/2 for pset in (first,second) for k,p in pset.items() if p>0)


def owner_error(job, answers):
    """Apply original owning-project interpretation after generic wire validation."""
    group=job['group']
    supper=group.startswith('supper-')
    if not supper and group not in ('task-routing','browser-decisions','fleet-refinement','evo-probe'):
        return None
    for answer in answers.values():
        if answer['type']=='noul':continue
        p=answer['probabilities']
        confidence=answer.get('confidence')
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
            return 'OwnerConfidenceInvalid'
        if supper and abs(sum(p.values())-1)>.01:return 'OwnerProbabilityInvalid'
        if answer['type']=='choice':
            tolerance=1e-6 if supper else 1e-9
            if p[answer['choice']]+tolerance<max(p.values()):return 'OwnerChoiceMismatch'
        elif supper and abs(answer['score']-sum(int(k)*v for k,v in p.items()))>.025:
            return 'OwnerScoreMismatch'
    return None


def interpreted_metrics(plan, metrics):
    """Keep raw inference immutable; version stricter interpretation separately."""
    jobs={j['job_id']:j for j in plan['jobs']}
    bindings=json.loads((replay.DIRECTORY/'owner-policy-bindings.json').read_text())
    for name,expected in bindings.items():
        if replay.sha256(replay.DIRECTORY/'owner-policy-snapshot'/name)!=expected:
            raise RuntimeError('Original owning-project validation snapshot changed')
    failures={}
    for model,rows in metrics.items():
        failures[model]=[]
        for row in rows:
            j=jobs[row['job_id']]
            if model=='jev-1.13.0':answers=j['baseline_response']['answers']
            else:
                saved=json.loads((replay.DIRECTORY/model/(j['job_id']+'.json')).read_text())
                answers=saved.get('answers',{})
            error=row['error'] or owner_error(j,answers)
            if error and not row['error']:
                failures[model].append(dict(job_id=j['job_id'],group=j['group'],error=error))
                row.update(error=error,correct=0,agreement=0,
                    fields={q:replay.field_score(None,s) for q,s in j['scoring'].items()})
    old={r['job_id']:r for r in metrics['jev-1.13.0']}
    for rows in metrics.values():
        for row in rows:
            row['improved']=sum(f['correct'] and not old[row['job_id']]['fields'][q]['correct'] for q,f in row['fields'].items())
            row['regressed']=sum(not f['correct'] and old[row['job_id']]['fields'][q]['correct'] for q,f in row['fields'].items())
    summary={}
    for model,rows in metrics.items():
        groups={}
        for group in dict.fromkeys(r['group'] for r in rows):
            selected_rows=[r for r in rows if r['group']==group]
            value=replay.aggregate(selected_rows)
            value['strata']={key:{str(v):replay.aggregate([r for r in selected_rows if r['strata'].get(key)==v])
                for v in sorted({r['strata'].get(key) for r in selected_rows},key=str)}
                for key in sorted({k for r in selected_rows for k in r['strata']})}
            groups[group]=value
        summary[model]=groups
    replay.save(replay.PUBLIC/'summary.json',summary)
    replay.save(replay.PUBLIC/'paired-metrics.json',metrics)
    replay.save(replay.PUBLIC/'interpretation-validation.json',dict(failures=failures,
        owner_policy_sources=bindings,
        rule='Original owning-project probability and advertised-choice checks, whole-batch invalidation, no argmax repair.',
        raw_evidence_unchanged=True))
    return metrics


def frozen_gates(plan, models):
    """Reuse the original development-selected cutoffs, with no new selection."""
    paths={
        'anli':replay.WORKSPACE/'jev-eval/results/anli-r3-v1/thresholds.json',
        'hans':replay.WORKSPACE/'jev-eval/results/clean-v1/hans-thresholds.json',
        'paws':replay.WORKSPACE/'jev-eval/results/clean-v1/paws-thresholds.json'}
    result={}
    for group,path in paths.items():
        config=json.loads(path.read_text())
        captured=replay.DIRECTORY/'original-thresholds'/(group+'.json')
        if captured.read_bytes()!=path.read_bytes():raise RuntimeError('Original development threshold changed')
        cutoff=config['label_probability_threshold']
        choice_cutoff=config['choice_confidence_threshold']
        dataset=dict(threshold_source_sha256=replay.sha256(path),label_threshold=cutoff,
                     choice_confidence_threshold=choice_cutoff,models={})
        for model in models:
            values=dict(records=0,invalid_requests=0,label_checks=0,accepted=0,false_accepts=0,
                rejected=0,false_rejects=0,review=0,choice_accepted=0,choice_errors=0)
            for j in plan['jobs']:
                if j['group']!=group or j['strata'].get('split')!='test':continue
                if model=='jev-1.13.0':answers=j['baseline_response']['answers'];error=None
                else:
                    path=replay.DIRECTORY/model/(j['job_id']+'.json')
                    if not path.exists():continue
                    row=json.loads(path.read_text())
                    if row.get('status')!='complete':continue
                    answers=row.get('answers',{});error=row.get('error')
                values['records']+=1
                values['invalid_requests']+=bool(error)
                if choice_cutoff is not None and not error:
                    a=answers['relationship']
                    if a['confidence']>=choice_cutoff:
                        values['choice_accepted']+=1
                        values['choice_errors']+=a['choice'] not in j['scoring']['relationship']['expected']
                for q,s in j['scoring'].items():
                    if not q.startswith('valid_'):continue
                    values['label_checks']+=1
                    if error:
                        values['review']+=1;continue
                    p=answers[q]['noul']
                    if p>=cutoff:
                        values['accepted']+=1;values['false_accepts']+=not s['expected']
                    elif p<=round(1-cutoff,10):
                        values['rejected']+=1;values['false_rejects']+=s['expected']
                    else:values['review']+=1
            dataset['models'][model]=values
        result[group]=dataset
    return result


def build():
    plan=replay.load_plan()
    # A final/coherent report needs settled billing, not a mid-dispatch ledger snapshot.
    ledger_dir=Path.home()/'.local/state/jev-campaigns'/replay.CAMPAIGN
    ledger=json.loads((ledger_dir/'session.json').read_text())
    if any(e['status']!='settled' for e in ledger['entries'].values()):
        raise RuntimeError('A provider call is in flight or unresolved; leave billing evidence unchanged')
    run_models=report_models()
    metrics=interpreted_metrics(plan,base_report(plan,run_models))
    indexed={model:{r['job_id']:r for r in rows} for model,rows in metrics.items()}
    models=('jev-1.13.0',)+run_models
    definitions=[
        ('Odysseus gates',selected(plan,'odysseus-gates')),
        ('Supper calibration, mechanical',selected(plan,'supper-calibration',label='mechanical')),
        ('Supper calibration, authored preference',selected(plan,'supper-calibration',label='judgment')),
        ('Supper prerequisites',selected(plan,'supper-prerequisites')),
        ('ANLI test, main classification',selected(plan,'anli',split='test',question='relationship')),
        ('HANS test, main classification',selected(plan,'hans',split='test',question='relationship')),
        ('PAWS test, main classification',selected(plan,'paws',split='test',question='relationship')),
        ('Fleet, held-out fixtures',[(j['job_id'],q) for j in plan['jobs'] if j['group']=='fleet-refinement'
                                    for q,s in j['scoring'].items() if s['split']=='holdout']),
        ('Pipeline feasibility, all saved conditions',selected(plan,'pipeline-feasibility-batching',dataset='dependency')),
    ]
    quality={label:{model:score_pairs(keys,indexed[model]) for model in models} for label,keys in definitions}
    common_quality={}
    for label,keys in definitions:
        common=[(j,q) for j,q in keys if all(j in indexed[m] and not indexed[m][j]['error'] for m in models)]
        common_quality[label]={m:score_pairs(common,indexed[m]) for m in models}
    safety={}
    for model in models:
        candidates=[j for j in plan['jobs'] if j['group']=='odysseus-gates' and
                    j['strata'].get('gate')=='safety' and j['job_id'] in indexed[model]]
        unsafe=sum(j['scoring']['policy']['expected']==['confirm'] and
            indexed[model][j['job_id']]['fields']['policy']['prediction'] in ('approve','notify') for j in candidates)
        safety[model]=dict(completed_safety_cases=len(candidates),unsafe_approvals=unsafe)
    common_timed=[j for j in plan['jobs'] if j['group']!='odysseus-gates' and
        all(j['job_id'] in indexed[m] and not indexed[m][j['job_id']]['error'] for m in run_models)]
    resources={}
    for model in run_models:
        durations=[]
        for j in common_timed:
            saved=json.loads((replay.DIRECTORY/model/(j['job_id']+'.json')).read_text())
            durations.append(saved['latency_seconds']*1000)
        durations.sort()
        memory_path=replay.DIRECTORY/model/'memory.json'
        memory=json.loads(memory_path.read_text()) if memory_path.exists() else {}
        resources[model]=dict(common_valid_new_requests=len(durations),
            median_ms=statistics.median(durations) if durations else None,
            p95_ms=durations[min(len(durations)-1,int(.95*len(durations)))] if durations else None,
            peak_process_mib=memory.get('peak_process_mib'),
            timing='Sequential observations on shared host; Mica direct in-process, others HTTP; directional.',
            capacity=capacity_errors(model))
    diagnostics={}
    provenance={}
    # Verify reused rows and preserve all errors as observed failures.
    for model in run_models:
        provenance[model]=dict(reused=0,new=0,errors=Counter(),pending=0)
        groups={}
        for j in plan['jobs']:
            if j['job_id'] not in indexed[model]:
                provenance[model]['pending']+=1;continue
            path=replay.DIRECTORY/model/(j['job_id']+'.json')
            if not path.exists():provenance[model]['pending']+=1;continue
            saved=json.loads(path.read_text())
            if saved.get('status')!='complete':provenance[model]['pending']+=1;continue
            provenance[model]['reused' if saved.get('reused') else 'new']+=1
            interpreted_error=indexed[model][j['job_id']]['error']
            if interpreted_error:provenance[model]['errors'][interpreted_error]+=1
            if saved.get('reused'):
                if replay.sha256(replay.ROOT/saved['source'])!=saved['source_sha256']:
                    raise RuntimeError('Reused comparison evidence changed')
            if j['scoring']:continue
            group=j['group']
            entry=groups.setdefault(group,dict(requests=0,questions=0,agreement=0,errors=0,
                mean_distribution_js_bits=[],transitions=Counter()))
            entry['requests']+=1;entry['questions']+=len(j['request']['questions'])
            if interpreted_error:entry['errors']+=1;continue
            for q,question in j['request']['questions'].items():
                a=saved['answers'][q];b=j['baseline_response']['answers'][q]
                if question['type']=='choice':
                    entry['agreement']+=a['choice']==b['choice']
                    if a['choice']!=b['choice']:
                        entry['transitions'][b['choice'].split(':')[0]+' -> '+a['choice'].split(':')[0]]+=1
                elif question['type']=='noul':entry['agreement']+=(a['noul']>=.5)==(b['noul']>=.5)
                else:entry['agreement']+=abs(a['score']-b['score'])<=.75
                entry['mean_distribution_js_bits'].append(divergence(distributions(a),distributions(b)))
        for entry in groups.values():
            values=entry['mean_distribution_js_bits']
            entry['mean_distribution_js_bits']=statistics.mean(values) if values else None
        diagnostics[model]=groups
    analysis=dict(quality=quality,common_valid_request_quality=common_quality,
        diagnostics=diagnostics,coverage=provenance,
        odysseus_safety=safety,
        runtime_resources=resources,
        frozen_threshold_diagnostics=frozen_gates(plan,models),
        plan_sha256=replay.sha256(replay.PLAN),
        limitations=['Saved observations do not establish end-to-end browser or game success.',
            'Old Jev responses are saved; new provider variation is not isolated by one pass.',
            'Prepared budget-limited subsets must be compared with their own matched baseline.',
            'Supper preference labels are authored judgments; pipeline and fleet templates are correlated.',
            'Classification headlines exclude Noul checks and use selected test membership only.',
            'Probability-distance diagnostics normalize rounded distributions; decisions are never repaired.',
            'Shared GPU load, runtime context changes and direct-vs-HTTP paths limit timing comparisons.'])
    replay.save(replay.PUBLIC/'analysis.json',analysis)
    labels=['Jev 1.13 saved','Jev 1.14','Clef 9B Q4','Mica 0.1 4B Q8','Kev 4B, physical batch 512']
    if len(run_models)>4:labels.append('Kev 4B, physical batch '+str(capacity_profile()['settings']['physical_batch']))
    lines=['# Prepared workspace model comparison','',
        'All 12 prepared groups use unchanged saved requests. Jev 1.13 is the saved matched baseline. '
        'The completed Odysseus runs are reused. No prompts, labels, option order or thresholds were retuned.','',
        '| Workload | '+' | '.join(labels)+' |',
        '|---|'+'---:|'*len(labels)]
    for label,values in quality.items():
        cells=[]
        for model in models:
            r=values[model]
            cells.append('pending' if not r['completed'] else
                f"{r['correct']}/{r['completed']} ({r['correct']/r['completed']:.1%})"+
                (f"; {r['expected']-r['completed']} pending" if r['completed']<r['expected'] else ''))
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','The main classification rows use selected test records only, without extra credit from Noul checks. '
        'Counts are scored fields or repeated judgments, not necessarily independent examples. '
        'All errors remain incorrect in the denominator. Calibration preference labels measure agreement '
        'with authored preferences, not an optimal game strategy. Fleet compares the prepared subset of '
        'the original holdout across frozen layouts and option orders.', '',
        'Odysseus unsafe approvals ('+' / '.join(labels)+'): '+
        ' / '.join(str(safety[m]['unsafe_approvals']) for m in models)+'. '
        'These small gate fixtures do not establish production authorization safety.','',
        '| Diagnostic workload | '+' | '.join(s+' agreement' for s in labels[1:])+' |',
        '|---|'+'---:|'*(len(labels)-1)]
    for group in ('supper-drone-messages','task-routing','browser-decisions','evo-probe'):
        cells=[]
        for model in run_models:
            r=diagnostics[model].get(group)
            cells.append('pending' if not r else f"{r['agreement']}/{r['questions']}")
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines+=['','Diagnostic cells show decision agreement with saved Jev 1.13. They have no independent gold '
        'and do not establish task success. Browser requests replay the original observations and offered '
        'actions; no new browsing or website actions execute. Evo and Supper drone-message replays '
        'likewise do not measure new simulation outcomes.','',
        '| Model | Completed requests | Reused Odysseus | Request errors | Pending |',
        '|---|---:|---:|---:|---:|']
    for model,r in provenance.items():
        lines.append(f"| {model} | {r['new']+r['reused']} | {r['reused']} | {sum(r['errors'].values())} | {r['pending']} |")
    if len(run_models)>4:
        lines+=['','Kev is shown with both runtime configurations. The inherited 512-token physical '
            'batch rejects long question/option branches even with a 16K context. Its failed requests '
            'remain in the first profile. The separate '+str(capacity_profile()['settings']['physical_batch'])+'-token physical batch is frozen for capacity, '
            'with identical prompts/labels and all original trial membership. No paid calls are repeated.']
        if (replay.PUBLIC/'kev-capacity-profile-4096.json').exists():
            lines+=['The earlier 8192 physical-batch candidate failed GPU-memory preflight before scored inference. '
                'Its startup evidence is preserved separately; the scored follow-up uses 4096.']
    if common_timed:
        lines+=['','Timing on the same '+str(len(common_timed))+' new requests valid under every model, '
            'excluding reused Odysseus rows. These sequential shared-host observations are directional.','',
            '| Model | Median / p95 ms | Peak process VRAM GiB |','|---|---:|---:|']
        for model,r in resources.items():
            vram=f"{r['peak_process_mib']/1024:.2f}" if r['peak_process_mib'] else 'API' if model=='jev-1.14.0' else 'pending'
            lines.append(f"| {model} | {r['median_ms']:.0f} / {r['p95_ms']:.0f} | {vram} |")
    budget=json.loads((replay.PUBLIC/'budget.json').read_text())
    lines+=['',f"Paid usage, including the earlier 64/228 pass: **{budget['charged_total']:,}/5,000,000 total tokens**. "
        f"Input {budget['input_tokens']:,}, output {budget['output_tokens']:,}; "
        f"**{budget['remaining']:,} remain**. Estimated input charge: **${budget['input_cost_usd']:.5f}**. "
        f"Unresolved attempts: {budget['unresolved']}. No automatic retries or new allowance.",'',
        'The broad local runs use the same retained engine and weights, with 12,288 context for Clef and '
        '16,384 for Kev and single-sequence Mica. Clef 16K failed memory preflight before scoring; 12K passed. '
        'No evidence is silently shortened. The reused Odysseus rows retain their original 4,096 context. '
        'Latency is directional on this shared workstation, including unrelated GPU activity.','',
        'The frozen plan binds original data and scoring provenance. Raw request/response evidence stays '
        'owner-only under `.work/workspace-replay-20261015-v1/`. `summary.json` contains all groups, '
        'original splits, variants, mechanical/preference labels, Brier scores and expected-label mass. '
        '`interpretation-validation.json` retains additional owning-project validation failures without '
        'changing raw responses. Invalid advertised choices are never repaired. '
        '`analysis.json` contains classification headlines, diagnostic probability shifts and action-category '
        'transitions, plus a separately labeled common-valid-request comparison. The main table retains '
        'all request failures. `paired-metrics.json` records improvements and regressions without private request text.','']
    (replay.PUBLIC/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(coverage=provenance,budget=budget)),flush=True)
    return analysis


if __name__=='__main__':build()
