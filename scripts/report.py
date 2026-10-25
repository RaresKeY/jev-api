#!/usr/bin/env python3
"""Generate comparison artifacts and selected settings from saved outputs only.

Standalone dependencies: Ignored historical local and Jev result directories; offline.
See docs/standalone-dependencies.md.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,SUITE,write_json,summarize,rank,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',type=Path,default=ROOT/'results/local-ab-v6')
    parser.add_argument('--jev',type=Path,default=ROOT/'results/jev-reference-v1')
    args=parser.parse_args()
    local=json.loads((args.local/'summary.json').read_text())
    jev_rows=[json.loads(line) for line in (args.jev/'responses.jsonl').read_text().splitlines()]
    jev_holdout=[row for row in jev_rows if row['case']['split']=='holdout']
    combined={name:value['holdout'] for name,value in local.items()}
    combined['Jev']=summarize(jev_holdout)
    output=ROOT/'results/comparison'
    output.mkdir(exist_ok=True)
    write_json(output/'summary.json',dict(local_run=str(args.local.relative_to(ROOT)),jev_run=str(args.jev.relative_to(ROOT)),suite_sha256=sha256(SUITE),holdout=combined,jev_all=summarize(jev_rows)))
    with open(output/'metrics.csv','w') as file:
        writer=csv.writer(file);writer.writerow(['backend','correct_fields','total_fields','exact_cases','cases','unsafe_approvals','median_ms','p95_ms'])
        for name,value in combined.items():
            writer.writerow([name,value['field_correct'],value['fields'],value['exact'],value['cases'],value['unsafe_approvals'],value['median_ms'],value['p95_ms']])
    dev=[case for case in json.loads(SUITE.read_text())['cases'] if case['split']=='dev']
    models=[]
    for name,value in local.items():
        if name=='nli-distilroberta-base':continue
        model=next((ROOT/'models'/name).glob('*.gguf'))
        selection=value['selected']
        models.append(dict(id=name,path=str(model.relative_to(ROOT)),filename=model.name,sha256=sha256(model),profile=selection['profile'],cache=selection['cache'],examples=dev if selection['profile'].get('examples') else [],dev=selection['summary'],holdout=value['holdout']))
    best=max(models,key=lambda item:rank(item['dev']))
    write_json(ROOT/'configs/local-best.json',dict(source_run=str(args.local.relative_to(ROOT)),suite_sha256=sha256(SUITE),default_gguf=best['id'],models=models,nli=local['nli-distilroberta-base']['selected']))
    lines=['# Local decision comparison','', 'Settings were selected using 12 development fixtures. The table below uses the same 12 held-out fixtures for every backend (40 scored fields). Jev is a comparison baseline; synthetic human labels remain the reference.','', '| Backend | Correct fields | Exact cases | Unsafe approvals | Median request ms |','|---|---:|---:|---:|---:|']
    for name,value in combined.items():
        median=f"{value['median_ms']:.1f}" if value['median_ms'] is not None else 'unavailable'
        lines.append(f"| {name} | {value['field_correct']}/{value['fields']} ({value['field_accuracy']:.1%}) | {value['exact']}/{value['cases']} | {value['unsafe_approvals']} | {median} |")
    lines += ['', 'The benchmark is small and synthetic. Per-field observations are correlated within a case. Local numbers are not calibrated probabilities. NLI scores are zero-shot adaptations rather than native Jev primitives. A passing JSON schema smoke does not establish semantic accuracy.','', 'Jev was called exactly once per fixture: 24 requests, no retries, all saved. The response model was `jev-1.13.0`. All-case Jev accuracy was 77/80 fields (96.25%); all three mismatches were false positives for GUI requirements.','', 'Latency is end-to-end and includes HTTP transport, prompt processing, output generation, and any split-question requests. Cloud uses HTTPS without connection pooling; NLI is in-process. Concurrent workstation jobs make timings directional. Ranking latency is only a last-resort tie-breaker, not a controlled speed claim.','', '## Selected configurations','']
    for model in models:
        lines += [f"- **{model['id']}**: `{model['profile']['name']}`, temperature `{model['profile']['temperature']}`, top-k `{model['profile']['top_k']}`, top-p `{model['profile']['top_p']}`, repetition penalty `{model['profile']['repeat_penalty']}`, KV cache `{model['cache']}`. Development field accuracy: {model['dev']['field_accuracy']:.1%}."]
    nli=local['nli-distilroberta-base']['selected']
    lines += [f"- **NLI**: premise `{nli['profile']['premise']}`, binary mapping `{nli['profile']['binary']}`, threshold `{nli['threshold']}`; eval/inference mode, local-only files, CUDA when available, batched premise/hypothesis pairs.",'', '## Protocol and evidence','', '- The fixture file and its SHA-256 were frozen before the Jev call. Expected answers were not supplied to inference except through explicitly marked development demonstrations. Demonstrations exclude the case being scored.','- Earlier local passes exposed integration defects and are retained as incomplete diagnostic evidence. Qwen holdout aggregates were seen during preliminary passes, so this is a development comparison, not an untouched external validation study. The final held-out cases did not select profiles, sampling, or cache settings; no labels were rewritten after looking at results.','- Model hashes match publisher LFS metadata. Both served chat templates match official tokenizer templates. No template override is warranted by these checks.','- The pinned llama.cpp README advertises a top-level schema shape that is ignored by this build; its upstream unit tests use `response_format.json_schema.schema`. The enforced-marker smoke proves the latter is active.','- The old Jev SDK field access was wrong: HTTP binary values are `answers[id].noul`. The new adapter normalizes them and preserves original distributions and usage.','- GPU allocation of the actual server PID and server model identity are checked before scoring; startup offload logs are retained when available. Readiness no longer accepts a previous model’s listener. Unique container names and process-group shutdown prevent launch/cleanup races.','- Full raw local requests/responses, Jev responses/usage, selected profiles and per-gate metrics are retained in the source run directories. API keys and headers are never recorded.','', '## Sources','', '- [Pinned llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/c4a278d68efa17811006f2123a84081dac03fac7/tools/server/README.md)','- [Upstream chat/schema smokes](https://github.com/ggml-org/llama.cpp/blob/c4a278d68efa17811006f2123a84081dac03fac7/tools/server/tests/unit/test_chat_completion.py)','- [Qwen model instructions](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)','- [SmolLM model instructions](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)','- [NLI label and usage instructions](https://huggingface.co/cross-encoder/nli-distilroberta-base)','- [Official Jev API](https://docs.typesafe.ai/api)','']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(combined,indent=2))

if __name__=='__main__':main()
