#!/usr/bin/env python3
"""Canonical gate demo. Prints decisions; never executes actions or grants permission.

Standalone dependencies: Selected server/model runtime or explicit Jev key; specialist backends need optional packages and ignored weights.
See docs/standalone-dependencies.md.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.env import load_env
from decision_lab.evaluation import ROOT,SUITE
from decision_lab.settings import configured_llama


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['llama','jev','kev','laya','verdict','nli-small','nli-large'],default='llama')
    parser.add_argument('--llama-url',default=None)
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--dump-payloads',action='store_true')
    args=parser.parse_args()
    load_env(ROOT/'.env')
    cases=json.loads(SUITE.read_text())['cases']
    if args.dump_payloads:
        print(json.dumps([{'model':'jev-latest','state':case['state'],'questions':case['questions']} for case in cases],indent=2))
        return
    if args.backend=='jev':
        from decision_lab.jev import JevClient
        client=JevClient();examples=[]
    elif args.backend!='llama':
        from decision_lab.specialists import KevClient,LayaClient,VerdictClient
        from decision_lab.nli import NLIClient
        if args.backend=='kev':client=KevClient(ROOT/'models/kev-0.8b',ROOT/'models/qwen3.5-0.8b-base')
        elif args.backend=='laya':client=LayaClient(ROOT/'models/laya-typed-decisions')
        elif args.backend=='verdict':client=VerdictClient(ROOT/'models/openjev-verdict')
        else:client=NLIClient(ROOT/('models/nli-distilroberta-base' if args.backend=='nli-small' else 'models/nli-deberta-v3-large'))
        examples=[]
        print(f'Local backend: {args.backend}; device: {client.device}')
    else:
        client,examples=configured_llama(args.llama_url)
        print(f'Local model: {client.model}; settings: {client.profile.get("name","default")}')
    def evaluate(state,questions,case_id=None):
        if args.backend=='jev':return client.evaluate(state,questions)
        if args.backend.startswith('nli'):
            profile=json.loads((ROOT/'configs/local-best.json').read_text())['nli']['profile']
            return client.evaluate(state,questions,profile)
        shots=[example for example in examples if set(questions)<=set(example['questions']) and example['id']!=case_id]
        return client.evaluate(state,questions,shots)
    try:
        if args.interactive:
            questions=next(case['questions'] for case in cases if case['gate']=='tools')
            while True:
                task=input('Task (empty to exit): ').strip()
                if not task:break
                print(json.dumps(evaluate({'task':task},questions),indent=2))
        else:
            for gate in ('reply','tools','routing','safety','completion','gui'):
                case=next(case for case in cases if case['gate']==gate)
                print(case['id'],json.dumps(evaluate(case['state'],case['questions'],case['id'])))
    except (RuntimeError,ValueError,KeyError) as error:
        parser.exit(2,f'Decision evaluation failed: {error}\n')

if __name__=='__main__':main()
