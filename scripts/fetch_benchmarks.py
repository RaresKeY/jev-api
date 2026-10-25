#!/usr/bin/env python3
"""Download original benchmark bytes at fixed commits; verify Git/LFS identities.

Standalone dependencies: Network/TLS access to pinned public sources and local storage; no inference.
See docs/standalone-dependencies.md.
"""
import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_decision_models import read,download
from decision_lab.evaluation import ROOT,write_json
SOURCES={
 'kev':('jaredpalmer/kev','84847f0a883d900f7de5b7a57eaa341ca7f9a6b4',[
 'README.md','LICENSE','kev/metrics.py','kev/suite.py','kev/evaluate.py','kev/model.py','kev/checkpoint.py','kev/data.py','kev/api.py','kev/device.py',
 'evals/v4/transfer-v4/test.jsonl','evals/v4/transfer-v4/manifest.json','evals/v7/decision-v7/test.jsonl','evals/v7/decision-v7/manifest.json','evals/hard-v1/test.jsonl','evals/hard-v1/manifest.json','evals/breadth-v1/manifest.json']),
 'laya':('NandhaKishorM/laya','fa9a2a7070b1789912a49ae24603bbfb1a78b001',['README.md','LICENSE','BENCHMARKS.md','docs/evals.md','laya/agent.py','laya/common.py','laya/evals.py','research/evals/fixture.jsonl','research/evals/zh_decision_bench.jsonl']),
 'verdict':('Heman10x-NGU/openJev-verdict-2.0','bff28567cff463b833bf044f351a8b7945d53e07',['README.md','LICENSE','core/engine_encoder.py','core/formatting.py','core/primitives.py','core/calibration.py','scripts/evaluate.py','data/test.jsonl','data/dataset_manifest.json','data/synthetic_test.jsonl','data/real_banking_test.jsonl','data/real_benchmark_manifest.json','data/slice_cardinality_k3.jsonl','data/slice_cardinality_k25.jsonl','data/slice_missing_option.jsonl','data/slice_distant_oos.jsonl','reports/verdict2_base_test.json']),
 'jevbench':('fstandhartinger/jevbench','bb05a335bc809e61b20c0f745d25499a82b326fc',['README.md','LICENSE','datasets/manifest.json','datasets/public/easy.jsonl','datasets/public/original.jsonl','datasets/public/hard.jsonl','datasets/public/HARD-TIER.md','jevbench/tasks.py','jevbench/scoring.py','jevbench/metrics.py','jevbench/adapters/verdict_local.py','jevbench/adapters/laya_local.py'])}
def main():
 bundles={}
 for name,(repo,rev,names) in SOURCES.items():
  treepath=ROOT/f'benchmarks/upstream/{name}-tree.json'
  if name=='jevbench': treepath=ROOT/'benchmarks/upstream/jevbench-tree-metadata.json'
  if treepath.exists(): tree=json.loads(treepath.read_text())
  else:
   with read(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1') as r:tree=json.load(r)
  entries={x['path']:x for x in tree['tree'] if x['type']=='blob'}
  files=[];missing=[]
  for p in names:
   if p not in entries:missing.append(p);continue
   e=entries[p];files.append(dict(path=f'benchmarks/upstream/{name}/{p}',url=f'https://raw.githubusercontent.com/{repo}/{rev}/{p}',bytes=e['size'],git_blob_sha1=e['sha']))
  bundles[name]=dict(repository=repo,revision=rev,files=files,missing_paths=missing)
 meta=json.loads((ROOT/'benchmarks/upstream/typed-data-metadata.json').read_text())
 files=[]
 for e in meta['siblings']:
  n=e['rfilename']
  if 'test-' not in n and n not in {'README.md','eval.yaml'}:continue
  files.append(dict(path=f'benchmarks/upstream/typed-decisions/{n}',url=f"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/{meta['sha']}/{n}",bytes=e['size'],expected_sha256=e.get('lfs',{}).get('sha256'),git_blob_sha1=None if e.get('lfs') else e['blobId']))
 bundles['typed-decisions']=dict(repository='LocalLLaMA/typed-decisions',revision=meta['sha'],files=files,declared_license='Apache-2.0')
 with ThreadPoolExecutor(max_workers=8) as pool:
  futures=[(b,[pool.submit(download,f) for f in b['files']]) for b in bundles.values()]
  for b,fs in futures:b['files']=[f.result() for f in fs]
 write_json(ROOT/'benchmarks/upstream/manifest.json',dict(status='verified',bundles=bundles))
if __name__=='__main__':main()
