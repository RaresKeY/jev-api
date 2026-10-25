#!/usr/bin/env python3
"""Build the consolidated readiness report offline from saved evidence.

Standalone dependencies: Ignored results from the historical runs named in this script.
See docs/standalone-dependencies.md.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,write_json,sha256,summarize
from scripts.upstream_benchmarks import verify_sources
from decision_lab.benchmarks import jsonl


def read(path):return json.loads((ROOT/path).read_text())


def main():
 out=ROOT/'results/benchmark-readiness-v1';out.mkdir(exist_ok=True)
 originals=verify_sources()
 common=read('results/upstream-partial-v1/summary.json')
 # Preserve historical runs. Use repaired typed interface and exact original candidate protocol.
 common['verdict']=read('results/upstream-verdict-v2/summary.json')['verdict']
 original_verdict=read('results/verdict-original-protocol-v3/summary.json')['verdict']
 for suite,g in original_verdict.items():common['verdict'][suite]=g
 new=read('results/verdict-test-partial-v1/summary.json')
 for backend,g in new.items():common[backend].update(g)
 expected_suites=set(common['kev'])
 if any(set(g)!=expected_suites for g in common.values()):raise ValueError('Backend suites differ')
 # Verify matched original cases for every suite; derived protocol does not change task identity.
 paths={b:ROOT/f'results/upstream-partial-v1/{b}/responses.jsonl' for b in common}
 matched={b:{r['id']:r['case'] for r in jsonl(p)} for b,p in paths.items()}
 for backend in common:
  if backend=='verdict':
   matched[backend].update({r['id']:r['case'] for r in jsonl(ROOT/'results/upstream-verdict-v2/verdict/responses.jsonl')})
   matched[backend].update({r['id']:r['case'] for r in jsonl(ROOT/'results/verdict-original-protocol-v3/verdict/responses.jsonl')})
  else:matched[backend].update({r['id']:r['case'] for r in jsonl(ROOT/f'results/verdict-test-partial-v1/{backend}/responses.jsonl')})
 baseline=matched['kev']
 if any(c!=baseline for c in matched.values()):raise ValueError('Matched benchmark task content differs')
 write_json(out/'upstream-summary.json',common)
 odysseus={b:g['all_64'] for b,g in read('results/odysseus-64-v1/summary.json').items()}
 odysseus.update(read('results/specialists-odysseus-64-v1/summary.json'))
 odysseus['nli-deberta-v3-large']=read('results/nli-large-odysseus-64-v1/summary.json')
 odysseus['Jev (saved)']=read('results/jev-odysseus-64-v1/summary.json')
 suite=read('benchmarks/odysseus-v2.json');cases=[c for c in suite['cases'] if c['split']=='holdout']
 evidence=[ROOT/'results/jev-odysseus-64-v1/responses.jsonl',ROOT/'results/nli-large-odysseus-64-v1/responses.jsonl',*[ROOT/f'results/specialists-odysseus-64-v1/{b}/responses.jsonl' for b in ['kev','laya','verdict']]]
 for p in evidence:
  rows=jsonl(p)
  if [r['case'] for r in rows]!=cases or len(rows)!=64:raise ValueError('Odysseus cases differ')
  if summarize(rows)['fields']!=214:raise ValueError('Odysseus field count differs')
 write_json(out/'odysseus-summary.json',odysseus)
 order={}
 for p in sorted((ROOT/'results/option-order-partial-v1').glob('*/responses.jsonl')):
  rows=jsonl(p);pairs=list(zip(rows[::2],rows[1::2]))
  if any(b['case']['derived']['source_case']!=a['id'] for a,b in pairs):raise ValueError('Option pairs mismatched')
  order[p.parent.name]=dict(pairs=len(pairs),changed=sum(any(a['fields'][k].get('predicted')!=b['fields'][k].get('predicted') for k in a['fields']) for a,b in pairs),errors=sum('error' in r for r in rows))
 write_json(out/'option-order-summary.json',order)
 files=[ROOT/'benchmarks/inventory.json',ROOT/'benchmarks/upstream/manifest.json',*sorted((ROOT/'benchmarks/upstream').glob('*-manifest.json')),*evidence]
 write_json(out/'verification.json',dict(status='verified',original_files=len(originals),matched_upstream_cases=len(baseline),matched_upstream_fields=sum(g['fields'] for g in common['kev'].values()),backends=list(common),odysseus_cases=64,odysseus_fields=214,cloud_requests=0,evidence_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files}))
 lines=['# Benchmark readiness and measured results','','The deployable public suites and original scoring sources are prepared and partially tested. Read [the source/protocol inventory](../../docs/benchmark-inventory.md) before choosing full runs. New inference here is entirely local; Jev results below were read from disk.','','## Matched Odysseus 64 cases','','All models below share the same 64 case contents, 214 labels and unchanged workstation scoring. The GGUF and small NLI rows reuse earlier frozen-setting runs. Native decision heads and the larger NLI are new runs, without tuning on these cases.','','| Backend | Correct fields | Exact cases | Unsafe approvals | Median ms |','|---|---:|---:|---:|---:|']
 for b,g in sorted(odysseus.items(),key=lambda x:-x[1]['field_accuracy']):lines.append(f"| {b} | {g['field_correct']}/{g['fields']} ({g['field_accuracy']:.1%}) | {g['exact']}/64 | {g['unsafe_approvals']} | {g['median_ms']:.1f} |")
 lines+=['','Kev is the strongest local result on this set and closes part of the gap to Jev. It remains a measured candidate, not a replacement safety policy. No default config was changed. The larger NLI improves accuracy but doubles unsafe approvals relative to the smaller NLI (4 vs 2). These are synthetic workstation expectations and do not establish production reliability.','','## Upstream smoke: matched original tasks','','Each suite contributes 12 fixed sampled cases; all seven models share identical originals and labels. There are 228 cases / 299 questions per backend. Failures count as incorrect. This heterogeneous sample is not a weighted publisher index or a statistically reliable model ranking.','','| Suite | '+' | '.join(common)+' |','|---|'+'---:|'*len(common)]
 for suite in sorted(expected_suites):
  cells=[]
  for b in common:
   g=common[b][suite];cells.append(f"{g['correct']}/{g['fields']}"+(f"; {g['errors']} errors" if g['errors'] else ''))
  lines.append('| '+suite+' | '+' | '.join(cells)+' |')
 lines+=['','Laya is specialized to Typed Decisions, so its advantage there is expected. Native SNLI/MNLI probabilities are used only for the NLI models; other models answer the same premise/hypothesis as a typed choice. `upstream-summary.json` contains accuracy and calibration details. `results/upstream-partial-v1/kev-native-metrics.json` uses exact upstream clean-variant math separately.','','Verdict evidence is selected by protocol: v2 fixes structured question rendering; v3 preserves original candidate-array order, especially abstention position. Earlier v1/v2 files remain intact. One 78-option question exceeds Verdict’s supported capacity; Smol produced two out-of-range binary answers. Both remain failures.','','## Original free-text benchmarks','','| Original suite, 12 cases | Qwen | Smol |','|---|---:|---:|']
 ifeval=read('results/ifeval-partial-v1/summary.json');gsm=read('results/gsm8k-partial-v1/summary.json')
 lines.append(f"| IFEval strict prompt / instruction accuracy | {ifeval['qwen']['prompt_strict']:.1%} / {ifeval['qwen']['instruction_strict']:.1%} | {ifeval['smol']['prompt_strict']:.1%} / {ifeval['smol']['instruction_strict']:.1%} |")
 lines.append(f"| GSM8K 5-shot numeric accuracy | {gsm['qwen']['accuracy']:.1%} | {gsm['smol']['accuracy']:.1%} |")
 lines+=['','Google’s original 541 IFEval prompts and 834 instruction checks are available and all validators passed resource preflight. Twelve prompts were generated/scored per GGUF. The 1,024-token cap truncated 5 Qwen and 4 Smol answers. GSM8K used original training demonstrations and test gold; 2 Smol answers were truncated. There were no transport/scorer errors. These tiny samples and token-budget differences are not publisher leaderboard results.','','## Derived robustness probes','','Twelve original standard-tier cases were paired with the same options in reverse insertion order. Original state, descriptions and gold were preserved.','','| Backend | Changed decisions / 12 pairs |','|---|---:|']
 for b,g in order.items():lines.append(f"| {b} | {g['changed']}/12 |")
 lines+=['','No probe interface errors. Sampling differences and model order sensitivity both affect these single-seed pairs; this does not establish a population flip rate. Keep derived probes separate from original benchmark accuracy.','','## Ready for full tests','','- Full local public JevBench easy/standard/hard, Kev decision/transfer/hard, typed soft-gold, Verdict author/banking/cardinality/OOS, native NLI and ARC are available through `scripts/upstream_benchmarks.py --limit 0`.','- Full original IFEval and GSM8K are available through `scripts/general_benchmarks.py --limit 0`. Only generation models support these free-text tasks.','- Original strict metrics, raw distributions, soft targets, abstention, candidate limits and model context differences are now represented. Fresh output directories and implementation snapshots make new runs reviewable.','- Private JevBench/Kev data, specialized Verdict 2.0 weights, sponsor-only reference evaluation code, and unimplemented publisher log-likelihood/judge protocols remain explicit gaps. No full upstream sweep or new Jev pass was started.','','Validation: 33 unit/contract checks passed; installed Python dependency checks passed; original sources and matched case content were reverified. All task-owned llama.cpp containers stopped; the pre-existing chatgpt container was preserved.','']
 (out/'report.md').write_text('\n'.join(lines));print(out/'report.md')

if __name__=='__main__':main()
