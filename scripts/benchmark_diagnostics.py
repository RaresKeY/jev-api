#!/usr/bin/env python3
"""Compute exact upstream Kev metrics offline and prepare reproducible order probes.

Standalone dependencies: Original benchmark data and ignored result files for the selected diagnostics.
See docs/standalone-dependencies.md.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,write_json
from decision_lab.benchmarks import load_suites,partial
from decision_lab.benchmark_diagnostics import kev_native,option_order_probe

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);args=p.parse_args()
 if args.run:
  output={}
  for source in sorted(args.run.glob('*/responses.jsonl')):
   rows=[json.loads(l) for l in source.read_text().splitlines()];output[source.parent.name]=kev_native(rows)
  write_json(args.run/'kev-native-metrics.json',output)
 cases=partial(load_suites()['jevbench-standard'],12)
 probes=[r for c in cases for r in [c,option_order_probe(c)]]
 write_json(ROOT/'benchmarks/derived/option-order-v1.json',dict(protocol='Derived robustness probe, separate from original benchmark scores',cases=probes,cloud_requests=0))
if __name__=='__main__':main()
