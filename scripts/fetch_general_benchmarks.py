#!/usr/bin/env python3
"""Preserve native NLI/LLM datasets and publisher cards at resolved revisions.

Standalone dependencies: Network/TLS access to public dataset files and local storage; no inference.
See docs/standalone-dependencies.md.
"""
import sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_decision_models import read,download
from decision_lab.evaluation import ROOT,write_json
REPOS={'snli':('datasets','stanfordnlp/snli',lambda n: '/test-' in n or n=='README.md'),
'mnli':('datasets','nyu-mll/multi_nli',lambda n:'validation_' in n and n.endswith('.parquet') or n=='README.md'),
'arc':('datasets','allenai/ai2_arc',lambda n:'/test-' in n or n=='README.md'),
'gsm8k':('datasets','openai/gsm8k',lambda n:n.startswith('main/test-') or n=='README.md'),
'qwen-card':('models','Qwen/Qwen2.5-0.5B-Instruct',lambda n:n in {'README.md','LICENSE'}),
'smol-card':('models','HuggingFaceTB/SmolLM2-135M-Instruct',lambda n:n in {'README.md','LICENSE'}),
'nli-card':('models','cross-encoder/nli-distilroberta-base',lambda n:n in {'README.md','all_nli_best_performance.csv'})}
def prepare(item):
 name,(kind,repo,select)=item
 with read(f'https://huggingface.co/api/{kind}/{repo}?blobs=true') as r:meta=json.load(r)
 write_json(ROOT/f'benchmarks/upstream/{name}/metadata.json',meta)
 files=[]
 for e in meta['siblings']:
  n=e['rfilename']
  if not select(n):continue
  files.append(dict(path=f'benchmarks/upstream/{name}/{n}',url=f"https://huggingface.co/{'datasets/' if kind=='datasets' else ''}{repo}/resolve/{meta['sha']}/{n}",bytes=e['size'],expected_sha256=e.get('lfs',{}).get('sha256'),git_blob_sha1=None if e.get('lfs') else e['blobId']))
 return name,dict(repository=repo,revision=meta['sha'],declared_license=meta.get('cardData',{}).get('license'),files=files)
def main():
 with ThreadPoolExecutor(max_workers=7) as pool: bundles=dict(pool.map(prepare,REPOS.items()))
 with ThreadPoolExecutor(max_workers=7) as pool:
  pending=[(b,[pool.submit(download,f) for f in b['files']]) for b in bundles.values()]
  for b,fs in pending:b['files']=[f.result() for f in fs]
 write_json(ROOT/'benchmarks/upstream/general-manifest.json',dict(status='verified',bundles=bundles))
if __name__=='__main__':main()
