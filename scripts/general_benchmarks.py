#!/usr/bin/env python3
"""Original IFEval validators and GSM8K math tasks via local llama.cpp chat.

Standalone dependencies: Optional benchmark packages/data, ignored weights and shared manager/Podman/CUDA runtime.
See docs/standalone-dependencies.md.
"""
import argparse,json,random,re,sys,time
from decimal import Decimal,InvalidOperation
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,write_json,sha256
from decision_lab.llama import LlamaServerClient
from decision_lab.runtime import ManagedServer
from decision_lab.benchmarks import partial,jsonl


def original_cases(suite):
    if suite=='ifeval':
        path=ROOT/'benchmarks/upstream/ifeval/input_data.jsonl'
        return [dict(id=f"ifeval/{r['key']}",stratum=r['instruction_id_list'][0],original=r,source_sha256=sha256(path)) for r in jsonl(path)]
    import pyarrow.parquet as pq
    path=next((ROOT/'benchmarks/upstream/gsm8k/main').glob('test-*.parquet'))
    return [dict(id=f'gsm8k/{i}',stratum='math',original=r,source_sha256=sha256(path)) for i,r in enumerate(pq.read_table(path).to_pylist())]


def numeric_answer(text):
    matches=re.findall(r'####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)',text)
    if not matches:return None
    try:return str(Decimal(matches[-1].replace(',','')).normalize())
    except InvalidOperation:return None


def score_ifeval(case,response):
    from vendored.ifeval import evaluation_lib
    random.seed(42)
    r=case['original'];inp=evaluation_lib.InputExample(**r)
    strict=evaluation_lib.test_instruction_following_strict(inp,{inp.prompt:response})
    loose=evaluation_lib.test_instruction_following_loose(inp,{inp.prompt:response})
    return dict(prompt_strict=strict.follow_all_instructions,instruction_strict=strict.follow_instruction_list,prompt_loose=loose.follow_all_instructions,instruction_loose=loose.follow_instruction_list)


def prompt_gsm8k(case,shots):
    import pyarrow.parquet as pq
    if shots:
        path=next((ROOT/'benchmarks/upstream/gsm8k/main').glob('train-*.parquet'))
        examples=pq.read_table(path).to_pylist();examples=random.Random(42).sample(examples,shots)
    else:examples=[]
    prefix='Solve the math problem. End with #### followed by the numeric answer.\n\n'
    return prefix+'\n\n'.join([f"Question: {r['question']}\nAnswer: {r['answer']}" for r in examples]+[f"Question: {case['original']['question']}\nAnswer:"])


def generate(client,prompt,max_tokens):
    if not client.model:client.model=client._request('/v1/models')['data'][0]['id']
    settings={k:v for k,v in client.profile.items() if k in {'temperature','top_k','top_p','min_p','repeat_penalty'}}
    body=dict(model=client.model,messages=[{'role':'user','content':prompt}],max_tokens=max_tokens,seed=42,cache_prompt=False,**settings)
    result=client._request('/v1/chat/completions',body)
    client.last_raw={'request':body,'response':result}
    return result['choices'][0]['message']['content'],result['choices'][0].get('finish_reason')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['ifeval','gsm8k'],required=True)
    p.add_argument('--backends',nargs='+',choices=['qwen','smol'],default=['qwen','smol'])
    p.add_argument('--limit',type=int,default=12,help='0 runs the full original split')
    p.add_argument('--shots',type=int,choices=[0,5],default=5)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.limit<0 or args.max_tokens<=0:p.error('Invalid limit/token budget')
    import nltk
    from langdetect import DetectorFactory
    nltk.data.path=[str(ROOT/'benchmarks/upstream/nltk-data')];DetectorFactory.seed=0
    cases=original_cases(args.suite);cases=partial(cases,args.limit) if args.limit else cases
    # Preflight every original validator with a dummy response, before launching.
    if args.suite=='ifeval':
        for case in original_cases('ifeval'):score_ifeval(case,'A short preflight response.\nThis checks resources.')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'cases.json',cases)
    cfg=json.loads((ROOT/'configs/local-best.json').read_text())
    manifest=dict(status='started',suite=args.suite,count=len(cases),cloud_requests=0,case_sha256=sha256(args.output/'cases.json'),source_sha256=sha256(Path(__file__)),max_tokens=args.max_tokens,gsm8k_shots=args.shots,demonstration_seed=42,config=cfg,mode='partial' if args.limit else 'full',protocol='Original IFEval prompt and validators, native served chat template; GSM8K original question/answer and 5 original training demos, custom #### extraction. Quantized chat inference is not publisher lighteval log-likelihood replication.')
    write_json(args.output/'manifest.json',manifest);summaries={}
    for backend in args.backends:
        m=cfg['models'][0 if backend=='qwen' else 1];out=args.output/backend;out.mkdir();rows=[]
        if sha256(ROOT/m['path'])!=m['sha256']:raise ValueError('GGUF weight changed')
        with ManagedServer(ROOT/m['path'],out/'runtime',cache=m['cache']) as server:
            client=LlamaServerClient(server.base_url,profile=m['profile'])
            with open(out/'responses.jsonl','x') as f:
                for case in cases:
                    prompt=case['original']['prompt'] if args.suite=='ifeval' else prompt_gsm8k(case,args.shots)
                    row=dict(id=case['id'],case=case);start=time.perf_counter()
                    try:
                        response,finish=generate(client,prompt,args.max_tokens)
                        row.update(response=response,finish_reason=finish,raw=client.last_raw)
                        if args.suite=='ifeval':row['score']=score_ifeval(case,response)
                        else:
                            expected=numeric_answer(case['original']['answer']);pred=numeric_answer(response)
                            if expected is None:raise ValueError('Original numeric gold absent')
                            row['score']=dict(correct=pred==expected,predicted=pred,expected=expected)
                    except Exception as e:row.update(error=f'{type(e).__name__}: {str(e)[:300]}')
                    row['elapsed_ms']=(time.perf_counter()-start)*1000;rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(backend,case['id'],row.get('score',row.get('error')),flush=True)
        summary=dict(cases=len(rows),errors=sum('error' in r for r in rows),truncated=sum(r.get('finish_reason')=='length' for r in rows))
        if args.suite=='ifeval':
            n=sum(len(c['original']['instruction_id_list']) for c in cases)
            summary.update(prompt_strict=sum(r.get('score',{}).get('prompt_strict',False) for r in rows)/len(rows),prompt_loose=sum(r.get('score',{}).get('prompt_loose',False) for r in rows)/len(rows),instruction_strict=sum(sum(r.get('score',{}).get('instruction_strict',[])) for r in rows)/n,instruction_loose=sum(sum(r.get('score',{}).get('instruction_loose',[])) for r in rows)/n,instructions=n)
        else:summary['accuracy']=sum(r.get('score',{}).get('correct',False) for r in rows)/len(rows)
        summaries[backend]=summary;write_json(args.output/'summary.json',summaries)
    manifest['status']='complete';write_json(args.output/'manifest.json',manifest)

if __name__=='__main__':main()
