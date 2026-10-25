"""Source-preserving benchmark adapters. Gold and provenance never enter inputs."""
import hashlib,json,math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from .evaluation import ROOT,sha256
from .specialists import labels
from vendored.jevbench.scoring import score_task,score_label
from vendored.jevbench.metrics import brier_score,ece_top_label,ordinal_mae
UPSTREAM=ROOT/'benchmarks/upstream'
NLI_LABELS=['entailment','neutral','contradiction']


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


@lru_cache(maxsize=None)
def source_hash(path):
    return sha256(path)


def record(suite,id,state,questions,expected,source,index,**extra):
    return dict(suite=suite,id=f'{suite}/{id}',state=state,questions=questions,expected=expected,
                source=dict(path=str(source.relative_to(ROOT)),row=index,sha256=source_hash(source)),**extra)


def load_suites():
    import pyarrow.parquet as pq
    suites={}
    for tier,file in [('easy','easy'),('standard','original'),('hard','hard')]:
        suite=f'jevbench-{tier}';path=UPSTREAM/f'jevbench/datasets/public/{file}.jsonl';rows=[]
        for i,r in enumerate(jsonl(path)):
            expected=r['expected']
            if r['question']['type']=='noul':expected={'yes':'true','no':'false'}.get(expected,expected)
            rows.append(record(suite,r['id'],r['state'],{'decision':r['question']},{'decision':str(expected) if expected is not None else None},path,i,
                stratum=f"{r['family']}/{r['question']['type']}",original=r,metric='jevbench-exact'))
        suites[suite]=rows
    for suite,name in [('kev-transfer','v4/transfer-v4'),('kev-decision','v7/decision-v7'),('kev-hard','hard-v1')]:
        path=UPSTREAM/f'kev/evals/{name}/test.jsonl';rows=[]
        for i,r in enumerate(jsonl(path)):
            questions={k:{a:b for a,b in q.items() if a not in {'label','src'}} for k,q in r['questions'].items()}
            expected={k:str(q['label']).lower() if q['type']=='noul' else str(q['label']) for k,q in r['questions'].items()}
            family=r.get('_meta',{}).get('source',next(iter(r['questions'].values())).get('src','unknown'))
            rows.append(record(suite,r.get('_meta',{}).get('id',i),r['state'],questions,expected,path,i,stratum=str(family),original=r,metric='hard-label'))
        suites[suite]=rows
    suite='typed-decisions';path=UPSTREAM/'typed-decisions/all/test-00000-of-00001.parquet';rows=[]
    for i,r in enumerate(pq.read_table(path).to_pylist()):
        state,questions,gold=[json.loads(r[k]) for k in ['state','questions','gold']]
        rows.append(record(suite,r['id'],state,questions,{k:str(g['label']) for k,g in gold.items()},path,i,stratum=r['workflow'],gold={k:g['probabilities'] for k,g in gold.items()},original=r,metric='soft-gold'))
    suites[suite]=rows
    path=UPSTREAM/'laya/research/evals/zh_decision_bench.jsonl';suite='laya-zh';rows=[]
    for i,r in enumerate(jsonl(path)):
        rows.append(record(suite,i,r['state'],r['questions'],{k:str(v).lower() if isinstance(v,bool) else str(v) for k,v in r['expected'].items()},path,i,stratum=r.get('tags',['unknown'])[0],original=r,metric='hard-label'))
    suites[suite]=rows
    for suffix in ['test','real_banking_test','slice_cardinality_k3','slice_cardinality_k25','slice_missing_option','slice_distant_oos']:
        path=UPSTREAM/f'verdict/data/{suffix}.jsonl';suite='verdict-'+suffix;rows=[]
        for i,r in enumerate(jsonl(path)):
            q=dict(type='choice',instructions=r['question'],criteria={c['id']:c['description'] for c in r['candidates']})
            rows.append(record(suite,r['id'],r['text'],{'decision':q},{'decision':r['target_id']},path,i,stratum=r.get('abstention_subtype','unknown'),original=r,metric='hard-label'))
        suites[suite]=rows
    for suite,folder,pattern in [('snli','snli','**/test-*.parquet'),('mnli-matched','mnli','**/validation_matched-*.parquet'),('mnli-mismatched','mnli','**/validation_mismatched-*.parquet')]:
        rows=[]
        for path in sorted((UPSTREAM/folder).glob(pattern)):
            table=pq.read_table(path)
            features=json.loads(table.schema.metadata[b'huggingface'])['info']['features']
            native_names=features['label']['names']
            if native_names!=NLI_LABELS:raise ValueError(f'Unexpected dataset label order: {native_names}')
            for i,r in enumerate(table.to_pylist()):
                if r['label']==-1:continue
                q=dict(type='choice',instructions='What is the relation of the hypothesis to the premise?',criteria=dict(entailment='The hypothesis follows from the premise.',neutral='The hypothesis may or may not be true given the premise.',contradiction='The hypothesis contradicts the premise.'))
                expected=native_names[r['label']]
                rows.append(record(suite,i,{'premise':r['premise'],'hypothesis':r['hypothesis']},{'decision':q},{'decision':expected},path,i,stratum=expected,native_pair=[r['premise'],r['hypothesis']],original=r,metric='native-nli'))
        suites[suite]=rows
    for name in ['ARC-Easy','ARC-Challenge']:
        suite=name.lower();rows=[]
        for path in sorted((UPSTREAM/'arc'/name).glob('test-*.parquet')):
            for i,r in enumerate(pq.read_table(path).to_pylist()):
                q=dict(type='choice',instructions=r['question'],criteria=dict(zip(r['choices']['label'],r['choices']['text'])))
                rows.append(record(suite,r['id'],{'question':r['question']},{'decision':q},{'decision':r['answerKey']},path,i,stratum=str(len(q['criteria'])),original=r,metric='task-equivalent-choice'))
        suites[suite]=rows
    # Free-response math is preserved and inventoried, but not relabeled into fabricated options.
    path=next((UPSTREAM/'gsm8k/main').glob('test-*.parquet'))
    suites['gsm8k']=[record('gsm8k',i,{'question':r['question']},{},{},path,i,stratum='math',original=r,metric='free-response-unimplemented') for i,r in enumerate(pq.read_table(path).to_pylist())]
    return suites


def validate_record(r):
    if not r['questions']:return
    if set(r['questions'])!=set(r['expected']):raise ValueError('Question/target IDs differ')
    for name,q in r['questions'].items():
        keys=labels(q)
        if len(set(keys))!=len(keys) or not keys:raise ValueError('Invalid label set')
        target=r['expected'][name]
        if target is not None and target not in keys:raise ValueError(f"Target {target} absent: {r['id']}/{name}")
        if 'gold' in r:
            p=r['gold'][name]
            if set(p)!=set(keys) or any(not math.isfinite(v) or v<0 for v in p.values()) or abs(sum(p.values())-1)>.001:raise ValueError('Invalid original soft gold')


def partial(rows,limit):
    """Round-robin source strata, hash order fixed before predictions; no gold-based tuning."""
    groups=defaultdict(list)
    for r in rows:groups[r['stratum']].append(r)
    for group in groups.values():group.sort(key=lambda r:hashlib.sha256(('upstream-smoke-v1/'+r['id']).encode()).hexdigest())
    selected=[]
    while len(selected)<min(limit,len(rows)):
        for key in sorted(groups):
            if groups[key]:selected.append(groups[key].pop(0))
            if len(selected)>=min(limit,len(rows)):break
    return selected


def score_record(r,distributions=None,predictions=None):
    fields={}
    for name,q in r['questions'].items():
        keys=labels(q);target=r['expected'][name]
        task=SimpleNamespace(labels=keys,expected=int(target) if q['type']=='score' and target is not None else target,question=q)
        if distributions is not None:
            scored=score_task(distributions.get(name),task)
        else:scored=score_label((predictions or {}).get(name),task)
        # Upstream scorer's lexical tie rule is retained, including binary false on a tie.
        probs=scored.get('probs')
        if probs and target is not None:
            scored['hard_brier']=brier_score(probs,target,keys)
            if 'gold' in r:
                gold=r['gold'][name]
                scored['soft_brier']=sum((probs[k]-gold[k])**2 for k in keys)
                scored['kl_from_gold']=sum(v*math.log(v/max(probs[k],1e-12)) for k,v in gold.items() if v>0)
            if q['type']=='score':scored['ordinal_error']=abs(float(target)-scored['ordinal_ev'])
        fields[name]=scored
    return fields


def summarize_rows(rows):
    fields=[f for r in rows for f in r['fields'].values() if f.get('correct') is not None]
    valid=[f for f in fields if f['valid']];cal=[(max(f['probs'].values()),f['correct']) for f in valid if f.get('probs')]
    result=dict(cases=len(rows),fields=len(fields),correct=sum(f['correct'] for f in fields),valid=len(valid),accuracy=sum(f['correct'] for f in fields)/len(fields) if fields else None,exact=sum(all(f.get('correct') for f in r['fields'].values()) for r in rows),calibration_n=len(cal),ece=ece_top_label(cal)['ece'] if cal else None)
    for key in ['hard_brier','soft_brier','kl_from_gold','ordinal_error']:
        values=[f[key] for f in fields if key in f];result[key]=sum(values)/len(values) if values else None
    return result
