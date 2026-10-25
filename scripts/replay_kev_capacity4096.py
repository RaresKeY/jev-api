"""Separate local capacity profile; original small-batch failures stay immutable.

Standalone dependencies: Private frozen .work/ cohort, ignored prior evidence, model weights and shared GPU runtime.
See docs/standalone-dependencies.md.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab import workspace_replay as replay
from decision_lab.runtime import ManagedServer
from decision_lab.llama_decision import LlamaDecisionClient
from compare_new_decisions import IMAGE,MODELS,gpu_rows

MODEL='kev-4b-q8-large-batch'
MANIFEST=replay.PUBLIC/'kev-capacity-profile-4096.json'


def prepare(plan):
    value=dict(schema='jev-local-capacity-profile/v1',model=MODEL,weights='kev-4b-q8',
        authorization='Explicit local capacity comparison over the frozen prepared cohort.',
        reason='Inherited physical batch 512 rejects long option lists. Separate mechanical capacity profile; original failures retained.',
        source_plan_sha256=replay.sha256(replay.PLAN),requests=len(plan['jobs']),
        settings=dict(context=16384,batch=4096,physical_batch=4096,parallel=1,cache='f16'),
        no_prompt_or_gold_changes=True,no_paid_calls=True,automatic_retries=False,
        reused_odysseus=64,selection='Runtime capacity only, no choices or confidence fitted on labels.')
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=value:raise RuntimeError('Frozen capacity profile changed')
    if not MANIFEST.exists():replay.save(MANIFEST,value)
    source=replay.DIRECTORY/'capacity-source-snapshot'/Path(__file__).name
    source.parent.mkdir(parents=True,exist_ok=True)
    if source.exists() and source.read_bytes()!=Path(__file__).read_bytes():raise RuntimeError('Capacity runner changed')
    if not source.exists():source.write_bytes(Path(__file__).read_bytes())
    return value


def run(plan):
    profile=prepare(plan)
    # Preserve GPU ownership of the complete primary pass before this follow-up.
    while True:
        path=replay.DIRECTORY/'mica-01-q8/execution.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):break
        time.sleep(10)
    with (replay.DIRECTORY/'local-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for model in replay.MODELS:
            if not json.loads((replay.DIRECTORY/model/'execution.json').read_text()).get('complete'):
                raise RuntimeError('Primary comparison is incomplete')
        output=replay.DIRECTORY/MODEL;output.mkdir(parents=True,exist_ok=True);output.chmod(0o700)
        weight=MODELS['kev-4b-q8']
        reference=json.loads((replay.OLD_RUNS/'kev-4b-q8-control/manifest.json').read_text())['weight']
        if weight.stat().st_size!=reference['bytes'] or replay.sha256(weight)!=reference['sha256']:
            raise RuntimeError('Retained Kev weights changed')
        samples=[]
        with ManagedServer(weight,output/'runtime',cache='f16',image=IMAGE,batch=4096,ubatch=4096,
                           parallel=1,ctx=16384) as server:
            client=LlamaDecisionClient(server.base_url)
            short={'q':dict(type='choice',instructions='What does the customer want?',
                criteria={'cancel':'Cancel order','track':'Track order'})}
            if client.evaluate('Cancel my order.',short)!={'q':'cancel'}:raise RuntimeError('Semantic smoke failed')
            # Prove long-option capacity before scoring, without using benchmark gold.
            filler='Unrelated descriptive filler. '*350
            large={'q':dict(type='choice',instructions='What does the customer want?',
                criteria={'cancel':'Cancel the order. '+filler,'track':'Track the order. '+filler})}
            client.distributions('Cancel my order.',large)
            replay.save(output/'preflight.json',dict(short_semantic='passed',
                long_options='accepted without truncation',raw=client.last_raw,settings=profile['settings']))
            pid=json.loads((server.output/'gpu.json').read_text())['server_pid']
            stop=threading.Event()
            def monitor():
                while not stop.is_set():
                    try:
                        used=gpu_rows().get(pid)
                        if used is not None:samples.append(dict(pid=pid,used_mib=used,time=time.time()))
                    except (subprocess.CalledProcessError,ValueError):pass
                    stop.wait(1)
            worker=threading.Thread(target=monitor,daemon=True);worker.start()
            replay.OLD_LOCAL[MODEL]='kev-4b-q8-control'
            try:replay.execute(client,MODEL,plan)
            finally:
                stop.set();worker.join()
                replay.save(output/'memory.json',dict(samples=samples,server_name=server.name,
                    peak_process_mib=max((r['used_mib'] for r in samples),default=None),settings=profile['settings']))
        if subprocess.run(['podman','container','exists',server.name]).returncode!=1:
            raise RuntimeError('Owned capacity-profile container remains')
        print('Kev larger-batch profile completed; original failures retained; owned container removed.',flush=True)


if __name__=='__main__':
    plan=replay.load_plan()
    if '--run' in sys.argv:run(plan)
    else:
        prepare(plan)
        print('Separate Kev capacity profile frozen. No inference or API calls.')
