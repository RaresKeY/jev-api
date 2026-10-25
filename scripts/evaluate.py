#!/usr/bin/env python3
"""Repeatable local tuning; Jev reference is an exclusive one-pass operation.

Standalone dependencies: Ignored weights, shared manager/Podman/CUDA image and NVIDIA tools for local runs; explicit key for Jev.
See docs/standalone-dependencies.md.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.env import load_env
from decision_lab.evaluation import ROOT,SUITE,write_json,sha256,evaluate_values,summarize,rank
from decision_lab.llama import LlamaServerClient
from decision_lab.runtime import ManagedServer


def run_cases(client,cases,path,examples=(),threshold=.5,nli_profile=None):
    path=Path(path)
    rows=[]
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if any(row['case'] not in cases for row in rows) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Saved cases differ from the requested suite')
    done={row['id'] for row in rows}
    with open(path,'a' if path.exists() else 'x') as file:
        for case in cases:
            if case['id'] in done:continue
            start=time.perf_counter()
            row=dict(case=case,id=case['id'],threshold=threshold)
            try:
                if nli_profile:
                    values=client.evaluate(case['state'],case['questions'],nli_profile)
                else:
                    shots=[example for example in examples if example['gate']==case['gate'] and example['id']!=case['id']]
                    values=client.evaluate(case['state'],case['questions'],shots)
                row.update(values=values,fields=evaluate_values(values,case,threshold))
            except Exception as error:
                row['error']=str(error)
            row['raw']=client.last_raw
            row['latency_ms']=(time.perf_counter()-start)*1000
            file.write(json.dumps(row,ensure_ascii=False)+'\n')
            file.flush()
            rows.append(row)
    return rows


def local(output,resume=False):
    if output.exists() and not resume:
        raise RuntimeError('Local output directory already exists; choose a new run name')
    output.mkdir(parents=True,exist_ok=True)
    if resume and (output/'inventory.json').exists():
        if json.loads((output/'inventory.json').read_text())['suite_sha256'] != sha256(SUITE):
            raise RuntimeError('Cannot resume after changing the suite')
    suite=json.loads(SUITE.read_text())['cases']
    dev=[case for case in suite if case['split']=='dev']
    holdout=[case for case in suite if case['split']=='holdout']
    models=sorted((ROOT/'models').rglob('*.gguf'))
    inventory=dict(suite_sha256=sha256(SUITE),models=[dict(path=str(model.relative_to(ROOT)),bytes=model.stat().st_size,sha256=sha256(model)) for model in models],image=subprocess_image(),selection='Dev only; field accuracy then unsafe approvals then exact cases then median latency. Holdout never selects settings.')
    write_json(output/'inventory.json',inventory)
    profiles=[
        dict(name='legacy',prompt='legacy',split=False,schema_style='legacy'),
        dict(name='legacy-native-schema',prompt='legacy',split=False,schema_style='native-top-level'),
        dict(name='plain-batch',prompt='plain',split=False),
        dict(name='plain-split',prompt='plain',split=True),
        dict(name='examples-batch',prompt='plain',split=False,examples=True),
        dict(name='examples-split',prompt='plain',split=True,examples=True),
        dict(name='direct',prompt='direct',split=True),
        dict(name='direct-examples',prompt='direct',split=True,examples=True),
        dict(name='reasoned',prompt='reasoned',split=True),
    ]
    results={}
    for model in models:
        model_id=model.parent.name
        model_out=output/model_id
        model_out.mkdir(exist_ok=True)
        runs=[]
        saved_profiles=all((model_out/(profile['name']+'-dev.jsonl')).exists() for profile in profiles) and (model_out/'model-card-sampling-dev.jsonl').exists()
        with (nullcontext(None) if resume and saved_profiles else ManagedServer(model,model_out/'q8')) as server:
            if not (model_out/'smoke.json').exists():smoke(server,model_id,model_out/'smoke.json')
            for candidate in profiles:
                profile=dict(temperature=0,top_k=0,top_p=1,min_p=0,repeat_penalty=1,**candidate)
                client=LlamaServerClient(server.base_url if server else None,profile=profile)
                examples=dev if candidate.get('examples') else []
                rows=run_cases(client,dev,model_out/(profile['name']+'-dev.jsonl'),examples)
                summary=summarize(rows)
                runs.append(dict(profile=profile,cache='q8_0',summary=summary))
                print(model_id,profile['name'],summary,flush=True)
            best=max(runs,key=lambda run:rank(run['summary']))
            settings=(dict(temperature=.7,top_k=20,top_p=.8,min_p=0,repeat_penalty=1.1) if 'qwen' in model_id else dict(temperature=.2,top_k=0,top_p=.9,min_p=0,repeat_penalty=1))
            profile=dict(best['profile'],name='model-card-sampling',**settings)
            client=LlamaServerClient(server.base_url if server else None,profile=profile)
            rows=run_cases(client,dev,model_out/'model-card-sampling-dev.jsonl',dev if profile.get('examples') else [])
            runs.append(dict(profile=profile,cache='q8_0',summary=summarize(rows)))
            greedy_best=max(runs,key=lambda run:rank(run['summary']))
        with (nullcontext(None) if resume and (model_out/'f16-dev.jsonl').exists() else ManagedServer(model,model_out/'f16',cache='f16')) as server:
            profile=greedy_best['profile']
            client=LlamaServerClient(server.base_url if server else None,profile=profile)
            rows=run_cases(client,dev,model_out/'f16-dev.jsonl',dev if profile.get('examples') else [])
            runs.append(dict(profile=profile,cache='f16',summary=summarize(rows)))
        # Freeze winner before any holdout inference.
        best=max(runs,key=lambda run:rank(run['summary']))
        write_json(model_out/'selection.json',dict(runs=runs,winner=best))
        with (nullcontext(None) if resume and (model_out/'holdout.jsonl').exists() else ManagedServer(model,model_out/'selected',cache=best['cache'])) as server:
            client=LlamaServerClient(server.base_url if server else None,profile=best['profile'])
            rows=run_cases(client,holdout,model_out/'holdout.jsonl',dev if best['profile'].get('examples') else [])
            held=summarize(rows)
            results[model_id]=dict(selected=best,holdout=held)
            print(model_id,'HOLDOUT',held,flush=True)
    nli_path=ROOT/'models/nli-distilroberta-base'
    if nli_path.exists():
        from decision_lab.nli import NLIClient
        client=NLIClient(nli_path)
        nli_out=output/'nli-distilroberta-base';nli_out.mkdir(exist_ok=True)
        runs=[]
        for premise in ('json','text'):
            for binary in ('entailment','entailment-vs-contradiction'):
                profile=dict(premise=premise,binary=binary)
                rows=run_cases(client,dev,nli_out/(premise+'-'+binary+'-dev.jsonl'),nli_profile=profile)
                for threshold in (.3,.5,.7):
                    scored=[dict(row,fields=evaluate_values(row.get('values',{}),row['case'],threshold)) for row in rows]
                    runs.append(dict(profile=profile,threshold=threshold,summary=summarize(scored)))
        best=max(runs,key=lambda run:rank(run['summary']))
        write_json(nli_out/'selection.json',dict(runs=runs,winner=best))
        rows=run_cases(client,holdout,nli_out/'holdout.jsonl',threshold=best['threshold'],nli_profile=best['profile'])
        results['nli-distilroberta-base']=dict(selected=best,holdout=summarize(rows))
        print('nli','HOLDOUT',results['nli-distilroberta-base']['holdout'],flush=True)
    write_json(output/'summary.json',results)


def subprocess_image():
    import subprocess
    return json.loads(subprocess.check_output(['podman','image','inspect','llama-server-local:latest']))[0]['Id']


def smoke(server,model_id,path):
    from decision_lab.http import request_json
    # Adapted public upstream chat/schema tests; no downloaded upstream executables.
    client=LlamaServerClient(server.base_url if server else None,profile=dict(temperature=0,top_k=0,top_p=1,min_p=0,repeat_penalty=1))
    enum_schema={'type':'object','properties':{'answer':{'type':'string','enum':['Paris','Rome']}},'required':['answer'],'additionalProperties':False}
    response=client.chat_json([{'role':'user','content':'What is the capital of France? Return JSON with answer.'}],enum_schema)
    forced_schema={'type':'object','properties':{'marker':{'const':'constraint-active-42'}},'required':['marker'],'additionalProperties':False}
    forced=client.chat_json([{'role':'user','content':'Return JSON with marker set to some unrelated text.'}],forced_schema)
    if forced != {'marker':'constraint-active-42'}:
        raise RuntimeError('Schema constraints are inactive')
    messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'Hello café 世界' }]
    rendered=request_json(server.base_url+'/apply-template',{'messages':messages,'add_generation_prompt':True})
    tokenizer=json.loads((ROOT/'docs'/('qwen_tokenizer.json' if 'qwen' in model_id else 'smol_tokenizer.json')).read_text())['data']
    from jinja2.sandbox import SandboxedEnvironment
    expected=SandboxedEnvironment().from_string(tokenizer['chat_template']).render(messages=messages,add_generation_prompt=True,tools=None)
    tokens=request_json(server.base_url+'/tokenize',{'content':'café 世界','add_special':False})
    write_json(path,dict(health=True,schema_smoke=response,capital_correct=response=={'answer':'Paris'},forced_constraint=forced,template_match=rendered['prompt']==expected,actual_template=rendered,expected_template=expected,unicode_tokens=tokens,raw=client.last_raw))
    if rendered['prompt']!=expected:
        raise RuntimeError('Chat template differs from official model template')


def jev(output):
    from decision_lab.jev import JevClient
    cases=json.loads(SUITE.read_text())['cases']
    if len(cases)>24:raise RuntimeError('Jev reference exceeds the 24-request budget')
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'attempt.json'
    # An existing attempt (including a crash) forbids all network calls.
    with open(lock,'x') as file:
        file.write(json.dumps(dict(status='started',suite_sha256=sha256(SUITE),maximum_requests=24,no_retries=True)))
    client=JevClient()
    cases=json.loads(SUITE.read_text())['cases']
    rows=[]
    with open(output/'responses.jsonl','x') as file:
        for case in cases:
            start=time.perf_counter()
            row=dict(case=case,id=case['id'])
            try:
                values=client.evaluate(case['state'],case['questions'])
                row.update(values=values,fields=evaluate_values(values,case),raw=client.last_raw)
            except Exception as error:
                row['error']=str(error)
            row['latency_ms']=(time.perf_counter()-start)*1000
            rows.append(row);file.write(json.dumps(row)+'\n');file.flush()
            print('jev',case['id'],'error' if row.get('error') else 'saved',flush=True)
            if row.get('error'):
                break
    write_json(output/'summary.json',summarize(rows))
    write_json(lock,dict(status='complete' if len(rows)==len(cases) and not rows[-1].get('error') else 'stopped',requests=len(rows),suite_sha256=sha256(SUITE),maximum_requests=24,no_retries=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend',choices=['local','jev'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Reuse saved local case outputs; never retries Jev')
    args=parser.parse_args()
    load_env(ROOT/'.env')
    if args.backend=='local':local(args.output,args.resume)
    else:jev(args.output)

if __name__=='__main__':
    main()
