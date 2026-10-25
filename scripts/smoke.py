#!/usr/bin/env python3
"""Small accuracy smoke against the current server or local NLI weights.

Standalone dependencies: Selected compatible server or ignored NLI weights and optional PyTorch/Transformers.
See docs/standalone-dependencies.md.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.env import load_env
from decision_lab.evaluation import ROOT,SUITE,summarize
from decision_lab.settings import configured_llama
from scripts.evaluate import run_cases
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['tools','replies','nli'],default='tools')
    parser.add_argument('--llama-url',default=None)
    args=parser.parse_args();load_env(ROOT/'.env')
    cases=json.loads(SUITE.read_text())['cases']
    try:
        if args.suite=='nli':
            from decision_lab.nli import NLIClient
            client=NLIClient(ROOT/'models/nli-distilroberta-base')
            config=json.loads((ROOT/'configs/local-best.json').read_text())['nli']
            examples=[];profile=config['profile'];threshold=config['threshold']
        else:
            gate='reply' if args.suite=='replies' else 'tools'
            cases=[case for case in cases if case['gate']==gate]
            client,examples=configured_llama(args.llama_url);profile=None;threshold=.5
        with tempfile.TemporaryDirectory() as directory:
            rows=run_cases(client,cases,Path(directory)/'smoke.jsonl',examples,threshold,profile)
        summary=summarize(rows)
        for row in rows:
            print(row['id'],row.get('error') or row['values'])
        print(json.dumps(summary,indent=2))
        raise SystemExit(0 if summary['exact']==summary['cases'] else 1)
    except (RuntimeError,ValueError,KeyError,OSError) as error:
        parser.exit(2,f'Smoke failed: {error}\n')

if __name__=='__main__':main()
