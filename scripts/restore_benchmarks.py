#!/usr/bin/env python3
"""Restore original benchmark assets from verified pinned manifests, not moving heads.

Standalone dependencies: Pinned local manifests plus network/TLS access to original public sources; no inference.
See docs/standalone-dependencies.md.
"""
import json,sys,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT
from scripts.download_decision_models import download

def main():
 manifests=[ROOT/'benchmarks/upstream/manifest.json',*sorted((ROOT/'benchmarks/upstream').glob('*-manifest.json'))]
 files={f['path']:f for p in manifests for b in json.loads(p.read_text())['bundles'].values() for f in b['files']}
 with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(download,files.values()))
 archive=ROOT/'benchmarks/upstream/nltk-data/punkt_tab.zip'
 if archive.exists():
  base=archive.parent/'tokenizers';base.mkdir(exist_ok=True)
  with zipfile.ZipFile(archive) as z:
   if any(not (base/n).resolve().is_relative_to(base.resolve()) for n in z.namelist()):raise ValueError('Unsafe corpus archive path')
   z.extractall(base)
 print(f'Restored/verified {len(files)} original files at pinned revisions')
if __name__=='__main__':main()
