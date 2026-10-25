"""Offline final integrity receipt. No keys, network, or model inference.

Standalone dependencies: Private .work/ cohort, ignored results, ../jev-pipelines, original ledger and Podman.
See docs/standalone-dependencies.md.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab import workspace_replay as replay
from analyze_workspace_replay import build,report_models,capacity_profile
sys.path.insert(0,str(replay.PIPELINES))


def main():
    plan=replay.load_plan()
    runtime_bindings=json.loads((replay.DIRECTORY/'runtime-source-bindings.json').read_text())
    for path,expected in runtime_bindings.items():
        snapshot=replay.DIRECTORY/'runtime-source-snapshot'/path
        if replay.sha256(snapshot)!=expected:raise RuntimeError('Frozen runtime source changed')
    expected_ids={j['job_id'] for j in plan['jobs']}
    if len(expected_ids)!=2942:raise RuntimeError('Prepared membership changed')
    assert len({j['cloud_identity'] for j in plan['jobs']})==2942
    if 'kev-4b-q8-large-batch' in report_models():
        profile=capacity_profile()
        if profile['source_plan_sha256']!=replay.sha256(replay.PLAN) or profile['settings']['physical_batch']!=4096:
            raise RuntimeError('Supplemental capacity protocol changed')
        if (replay.DIRECTORY/'capacity-source-snapshot/replay_kev_capacity4096.py').read_bytes()!=(replay.ROOT/'scripts/replay_kev_capacity4096.py').read_bytes():
            raise RuntimeError('Frozen capacity runner changed')
    receipt={}
    for model in report_models():
        out=replay.DIRECTORY/model
        manifest=json.loads((out/'execution.json').read_text())
        if not manifest['complete'] or manifest['plan_sha256']!=replay.sha256(replay.PLAN):
            raise RuntimeError('Incomplete model run')
        paths={p.stem:p for p in out.glob('*--*.json')}
        if set(paths)!=expected_ids:raise RuntimeError('Model coverage differs from frozen plan')
        rows={k:json.loads(p.read_text()) for k,p in paths.items()}
        if any(r.get('status')!='complete' for r in rows.values()):raise RuntimeError('Unresolved model attempt')
        reused=sum(bool(r.get('reused')) for r in rows.values())
        if reused!=64:raise RuntimeError('Completed Odysseus rows were not reused exactly')
        if model!='jev-1.14.0':
            old_name='kev-4b-q8-control' if model=='kev-4b-q8-large-batch' else replay.OLD_LOCAL[model]
            reference=json.loads((replay.OLD_RUNS/old_name/'manifest.json').read_text())['weight']
            weight=replay.ROOT/reference['path']
            if weight.stat().st_size!=reference['bytes'] or replay.sha256(weight)!=reference['sha256']:
                raise RuntimeError('Retained model weights changed')
            memory=json.loads((out/'memory.json').read_text())
            if not memory['samples'] or not memory['peak_process_mib']:
                raise RuntimeError('Actual GPU process allocation missing')
            target=memory.get('container',memory.get('server_name'))
            if not target or subprocess.run(['podman','container','exists',target]).returncode!=1:
                raise RuntimeError('Owned model container remains')
            if model!='mica-01-q8':
                runtime=json.loads((out/'runtime/runtime.json').read_text())
                gpu=json.loads((out/'runtime/gpu.json').read_text())
                if not gpu['allocations'] or gpu['server_pid'] not in {s['pid'] for s in memory['samples']}:
                    raise RuntimeError('GPU samples do not identify the actual native server')
                settings=runtime['settings']
                expected_context=12288 if model=='clef-flash-q4' else 16384
                if int(settings['LLM_CTX'])!=expected_context or settings['LLM_N_GPU_LAYERS']!='999':
                    raise RuntimeError('Native runtime differs from frozen capacity configuration')
                if model=='kev-4b-q8-large-batch' and any(int(settings[k])!=4096 for k in ('LLM_BATCH','LLM_UBATCH')):
                    raise RuntimeError('Supplemental physical batch differs from its frozen protocol')
                if not any(Path(reference['path']).name in v['id'] for v in runtime['models']['data']):
                    raise RuntimeError('Actual server model differs from expected weights')
        receipt[model]=dict(requests=len(rows),reused_odysseus=reused,
            errors=dict(Counter(r['error'] for r in rows.values() if r.get('error'))),
            execution_sha256=replay.sha256(out/'execution.json'))
        if model!='jev-1.14.0':receipt[model]['verified_weight']=reference
    canonical=Path.home()/'.local/state/jev-campaigns'/replay.CAMPAIGN/'session.json'
    ledger_before=canonical.read_bytes()
    value=json.loads(ledger_before)
    if value['halted'] or any(e['status']!='settled' for e in value['entries'].values()):
        raise RuntimeError('Unknown or unresolved paid charge')
    expected_attempts=292+(2942-64)
    if len(value['entries'])!=expected_attempts:
        raise RuntimeError('Unexpected extra/missing paid attempt or repeat')
    total=sum(e['total_charge'] for e in value['entries'].values())
    if total>5_000_000:raise RuntimeError('Total allowance exceeded')
    for identity,entry in value['entries'].items():
        call=json.loads((canonical.parent/'calls'/(identity+'.json')).read_text())
        raw=call['response'];counts=raw['usage']
        if (entry['request_sha256']!=replay.digest(call['request']) or
            raw.get('model')!='jev-1.14.0' or
            any(entry[k]!=counts[k] for k in ('input_tokens','output_tokens')) or
            entry['total_charge']!=counts['input_tokens']+counts['output_tokens']):
            raise RuntimeError('Canonical campaign charge differs from its raw exchange')
    for dataset in {e['dataset'] for e in value['entries'].values()}:
        if sum(e['input_tokens'] for e in value['entries'].values() if e['dataset']==dataset)>2_000_000:
            raise RuntimeError('Per-dataset input allowance exceeded')
    outputs=('report.md','summary.json','analysis.json','paired-metrics.json','budget.json','verification.json',
             'interpretation-validation.json')
    def forbidden(*args,**kwargs):raise AssertionError('Offline verification attempted network or credentials')
    with patch('urllib.request.urlopen',forbidden),patch('socket.socket.connect',forbidden), \
         patch('jev_pipelines.current_client.read_key',forbidden):
        build()
        first={name:replay.sha256(replay.PUBLIC/name) for name in outputs}
        build()
        second={name:replay.sha256(replay.PUBLIC/name) for name in outputs}
    if first!=second or canonical.read_bytes()!=ledger_before:
        raise RuntimeError('Offline report is nondeterministic or modified paid accounting')
    interpretation=json.loads((replay.PUBLIC/'interpretation-validation.json').read_text())
    for model in receipt:
        receipt[model]['owner_interpretation_errors']=dict(Counter(
            r['error'] for r in interpretation['failures'][model]))
    report_sources={}
    for name in ('decision_lab/workspace_report.py','scripts/analyze_workspace_replay.py',
                 'scripts/verify_workspace_replay.py','tests/test_workspace_analysis.py'):
        path=replay.ROOT/name
        target=replay.DIRECTORY/'report-source-snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=path.read_bytes():raise RuntimeError('Frozen final report source changed')
        if not target.exists():target.write_bytes(path.read_bytes())
        report_sources[name]=replay.sha256(path)
    replay.save(replay.PUBLIC/'final-verification.json',dict(plan_sha256=replay.sha256(replay.PLAN),
        source_files=len(plan['sources']),runtime_sources=len(runtime_bindings),models=receipt,
        exactly_3170_settled_paid_attempts=True,no_retries=True,total_tokens=total,
        all_canonical_raw_calls_verified=len(value['entries']),
        report_deterministic=True,network_and_credentials_forbidden=True,
        ledger_bytes_unchanged=True,report_sha256=first,report_sources=report_sources))
    print('Final offline integrity, billing, no-retry, GPU cleanup and deterministic-report checks passed.')


if __name__=='__main__':main()
