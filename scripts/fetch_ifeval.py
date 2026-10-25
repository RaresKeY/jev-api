#!/usr/bin/env python3
"""Fetch Google's original IFEval inputs and deterministic evaluators, pinned.

Standalone dependencies: Network/TLS access to public sources and local storage; no inference.
See docs/standalone-dependencies.md.
"""
import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_decision_models import read,download
from decision_lab.evaluation import ROOT,write_json
repo='google-research/google-research'
def main():
 with read(f'https://api.github.com/repos/{repo}/commits/master') as r:rev=json.load(r)['sha']
 with read(f'https://api.github.com/repos/{repo}/contents/instruction_following_eval?ref={rev}') as r:entries=json.load(r)
 with read(f'https://api.github.com/repos/{repo}/contents/instruction_following_eval/data?ref={rev}') as r:entries+=json.load(r)
 files=[]
 for e in entries:
  if e['name'] not in {'input_data.jsonl','instructions.py','instructions_registry.py','instructions_util.py','evaluation_lib.py','README.md','requirements.txt'}:continue
  files.append(dict(path=f"benchmarks/upstream/ifeval/{e['name']}",url=f"https://raw.githubusercontent.com/{repo}/{rev}/{e['path']}",bytes=e['size'],git_blob_sha1=e['sha']))
 bundles={'ifeval':dict(repository=repo,revision=rev,declared_license='Apache-2.0',files=files)}
 with ThreadPoolExecutor(max_workers=6) as pool:bundles['ifeval']['files']=list(pool.map(download,files))
 write_json(ROOT/'benchmarks/upstream/ifeval-manifest.json',dict(status='verified',bundles=bundles))
if __name__=='__main__':main()
