#!/usr/bin/env python3
"""Validate/inventory upstream suites; run explicit local subsets or full suites.

Standalone dependencies: Original suite data, optional dataset/runtime packages, ignored weights and shared GPU runtime.
See docs/standalone-dependencies.md.
"""
import argparse,gc,json,sys,time,platform,re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,write_json,sha256
from decision_lab.benchmarks import load_suites,validate_record,partial,score_record,summarize_rows
from decision_lab.specialists import KevClient,LayaClient,VerdictClient
from decision_lab.nli import NLIClient
from decision_lab.llama import LlamaServerClient
from decision_lab.runtime import ManagedServer
from decision_lab.llama_decision import LlamaDecisionClient
from decision_lab.kev_models import verified_model
from scripts.evaluate import smoke

BACKENDS=['qwen','smol','nli-small','nli-large','laya','verdict','kev','kev-4b-q8','kev-4b-q4','kev-9b-q4']


def verify_sources():
    files=[]
    for path in [ROOT/'benchmarks/upstream/manifest.json',*sorted((ROOT/'benchmarks/upstream').glob('*-manifest.json'))]:
        manifest=json.loads(path.read_text())
        for bundle in manifest['bundles'].values():
            for f in bundle['files']:
                p=ROOT/f['path']
                if p.stat().st_size!=f['bytes'] or sha256(p)!=f['sha256']:raise ValueError(f"Original source changed: {p}")
                files.append(f['path'])
    return files


def inventory(suites):
    files=verify_sources();out={}
    for name,rows in suites.items():
        ids=[r['id'] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate case IDs: {name}')
        for r in rows:validate_record(r)
        out[name]=dict(cases=len(rows),questions=sum(len(r['questions']) for r in rows),strata=sorted(set(r['stratum'] for r in rows)),sources=sorted(set(r['source']['path'] for r in rows)),metric=rows[0]['metric'],full_runner_ready=True,typed_runner_ready=name!='gsm8k',runner='scripts/general_benchmarks.py' if name=='gsm8k' else 'scripts/upstream_benchmarks.py')
    free_path=ROOT/'benchmarks/upstream/ifeval/input_data.jsonl'
    if free_path.exists():
        from decision_lab.benchmarks import jsonl
        free=jsonl(free_path)
        out['ifeval']=dict(cases=len(free),questions=sum(len(r['instruction_id_list']) for r in free),strata=sorted({i for r in free for i in r['instruction_id_list']}),sources=[str(free_path.relative_to(ROOT))],metric='original-ifeval-strict-loose',full_runner_ready=True,typed_runner_ready=False,runner='scripts/general_benchmarks.py')
    out['gsm8k'].update(questions=out['gsm8k']['cases'],metric='original-gsm8k-free-response')
    return dict(status='validated',original_files=len(files),suites=out,cloud_requests=0,source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'decision_lab/benchmarks.py',ROOT/'decision_lab/specialists.py',ROOT/'decision_lab/nli.py']})


def normalized_questions(questions):
    def text(v,fallback=''):
        if v in (None,''):return fallback
        return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)
    result={}
    for name,q in questions.items():
        nq=dict(q,instructions=text(q['instructions']))
        if q['type']=='choice':nq['criteria']={k:text(v,k) for k,v in q['criteria'].items()}
        if q['type']=='score':nq['criteria']=[text(v,str(i)) for i,v in enumerate(q['criteria'])]
        result[name]=nq
    return result


def evaluate(client,case):
    if case['suite'].startswith('verdict-') and hasattr(client,'candidate_distributions'):
        r=case['original']
        return score_record(case,client.candidate_distributions(r['text'],r['question'],r['candidates']))
    if case.get('native_pair') and hasattr(client,'pair_distributions'):
        return score_record(case,{'decision':client.pair_distributions([case['native_pair']])[0]})
    if hasattr(client,'distributions'):
        return score_record(case,client.distributions(case['state'],case['questions']))
    q=normalized_questions(case['questions'])
    values=client.evaluate(case['state'],q)
    # A scalar label is not a calibrated distribution. Do not invent one-hot probabilities.
    predictions={name:('true' if values[name]>.5 else 'false') if question['type']=='noul' else str(values[name]) for name,question in q.items()}
    return score_record(case,predictions=predictions)


def run(client,cases,out):
    rows=[]
    with open(out/'responses.jsonl','x') as file:
        for i,case in enumerate(cases):
            t=time.perf_counter();row=dict(id=case['id'],suite=case['suite'],case=case)
            try:
                row['fields']=evaluate(client,case);row['raw']=client.last_raw
            except Exception as e:
                row.update(error=f'{type(e).__name__}: {str(e)[:500]}',fields=score_record(case,predictions={}),raw=getattr(client,'last_raw',None))
            row['elapsed_ms']=(time.perf_counter()-t)*1000;rows.append(row)
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
            if (i+1)%12==0:print(out.name,i+1,len(cases),flush=True)
    grouped=defaultdict(list)
    for r in rows:grouped[r['suite']].append(r)
    result={name:summarize_rows(r) for name,r in grouped.items()}
    for name,r in grouped.items():result[name]['errors']=sum('error' in x for x in r)
    write_json(out/'summary.json',result)
    return result


def report(output):
    manifest=json.loads((output/'manifest.json').read_text());summary=json.loads((output/'summary.json').read_text())
    lines=['# Upstream partial benchmark results','','This is a fixed, deterministic smoke sample. It is not a leaderboard replication or a model-selection result. No Jev API calls.','','| Suite | '+ ' | '.join(summary)+' |','|---|'+'---:|'*len(summary)]
    for suite in manifest['suite_names']:
        cells=[]
        for backend,groups in summary.items():
            g=groups.get(suite)
            cells.append(f"{g['correct']}/{g['fields']} ({g['accuracy']:.1%}); errors {g['errors']}" if g else 'unavailable')
        lines.append('| '+suite+' | '+' | '.join(cells)+' |')
    lines+=['','Distributions, hard/soft Brier, KL from original soft gold, ordinal MAE and ECE are in summary.json. GGUF label-only answers have no invented calibration. Native NLI uses the original three-way label mapping; other models answer the same pair as a typed choice.','','See docs/benchmark-inventory.md for source licenses, unavailable suites, training overlap and protocol differences.','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','run','report'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--backends',nargs='+',choices=BACKENDS,default=BACKENDS[:7])
    parser.add_argument('--suites',nargs='+')
    parser.add_argument('--case-file',type=Path,help='Explicit derived fixture JSON; gold is never sent to models')
    parser.add_argument('--limit',type=int,default=12,help='Per-suite cases. 0 explicitly selects full suites.')
    parser.add_argument('--decision-image',default='localhost/jev-kev-server:llama-cpp-a4cb4c6-cuda12.6.0-ubuntu22.04',help='Pinned decision-capable llama.cpp image for Kev GGUF backends')
    args=parser.parse_args()
    if args.action=='report':report(args.output);return
    suites=load_suites();inv=inventory(suites)
    if args.action=='inventory':
        write_json(ROOT/'benchmarks/inventory.json',inv);print(json.dumps(inv,indent=2));return
    if args.output is None:parser.error('--output is required for run')
    if args.limit<0:parser.error('--limit must be >= 0')
    selected=args.suites or [n for n in suites if n!='gsm8k']
    if any(n not in suites or n=='gsm8k' for n in selected):parser.error('Unknown typed suite; use scripts/general_benchmarks.py for free-response suites')
    cases=[r for n in selected for r in (partial(suites[n],args.limit) if args.limit else suites[n])]
    if args.case_file:
        if args.suites:parser.error('--case-file and --suites are mutually exclusive')
        payload=json.loads(args.case_file.read_text());cases=payload['cases'] if isinstance(payload,dict) else payload
        for case in cases:
            validate_record(case)
            if sha256(ROOT/case['source']['path'])!=case['source']['sha256']:raise ValueError('Derived original source hash changed')
        if len({r['id'] for r in cases})!=len(cases):raise ValueError('Duplicate derived IDs')
        selected=sorted({r['suite'] for r in cases})
        if args.limit:cases=partial(cases,args.limit)
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/local-best.json').read_text())
    write_json(args.output/'cases.json',cases)
    manifest=dict(case_file=str(args.case_file) if args.case_file else None,status='started',mode='partial' if args.limit else 'full',suite_names=selected,case_count=len(cases),case_hash=sha256(args.output/'cases.json'),limit=args.limit,inventory=inv,config=config,cloud_requests=0,python=platform.python_version(),protocol='GGUF constrained label choice, zero-shot with frozen sampler/cache; native dedicated head probabilities; native NLI pair when available')
    write_json(args.output/'manifest.json',manifest)
    import shutil
    sources=[Path(__file__),*sorted((ROOT/'decision_lab').glob('*.py')),*sorted((ROOT/'vendored').rglob('*.py'))]
    for source in sources:
        target=args.output/'source-snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    results={}
    for backend in args.backends:
        out=args.output/backend;out.mkdir()
        print(backend,'loading',flush=True)
        try:
            if backend in {'kev-4b-q8','kev-4b-q4','kev-9b-q4'}:
                model_name={'kev-4b-q8':'q8','kev-4b-q4':'q4','kev-9b-q4':'9b-q4'}[backend]
                model,_=verified_model(model_name)
                with ManagedServer(model,out/'runtime',cache='f16',image=args.decision_image,batch=1024) as server:
                    client=LlamaDecisionClient(server.base_url)
                    client.distributions('Cancel my order.',{'decision':dict(type='choice',instructions='What does the customer want?',criteria={'cancel':'Cancel order','track':'Track order'})})
                    write_json(out/'smoke.json',client.last_raw)
                    results[backend]=run(client,cases,out)
            elif backend in {'qwen','smol'}:
                m=config['models'][0 if backend=='qwen' else 1]
                if sha256(ROOT/m['path'])!=m['sha256']:raise ValueError('GGUF changed')
                with ManagedServer(ROOT/m['path'],out/'runtime',cache=m['cache']) as server:
                    smoke(server,m['id'],out/'smoke.json')
                    client=LlamaServerClient(server.base_url,profile=m['profile'])
                    results[backend]=run(client,cases,out)
            else:
                if backend.startswith('nli'):client=NLIClient(ROOT/('models/nli-distilroberta-base' if backend=='nli-small' else 'models/nli-deberta-v3-large'))
                elif backend=='laya':client=LayaClient(ROOT/'models/laya-typed-decisions')
                elif backend=='verdict':client=VerdictClient(ROOT/'models/openjev-verdict')
                else:client=KevClient(ROOT/'models/kev-0.8b',ROOT/'models/qwen3.5-0.8b-base')
                # Warm-up on public toy data; excluded from scored sample and timing.
                warm=dict(type='choice',instructions='What does the customer want?',criteria={'cancel':'Cancel order','track':'Track order'})
                client.distributions('Cancel my order.',{'decision':warm})
                results[backend]=run(client,cases,out)
                write_json(out/'runtime.json',dict(device=client.device,weight_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/('models/nli-distilroberta-base' if backend=='nli-small' else 'models/nli-deberta-v3-large' if backend=='nli-large' else 'models/laya-typed-decisions' if backend=='laya' else 'models/openjev-verdict' if backend=='verdict' else 'models/kev-0.8b')).glob('*.safetensors')}))
            del client
        except Exception as e:
            write_json(out/'load-error.json',dict(error=f'{type(e).__name__}: {str(e)[:500]}'));print(backend,'failed',type(e).__name__,str(e)[:150],flush=True)
        gc.collect()
        if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()
        write_json(args.output/'summary.json',results)
    manifest['status']='complete' if len(results)==len(args.backends) else 'partial-backend-failure'
    write_json(args.output/'manifest.json',manifest);report(args.output)

if __name__=='__main__':main()
