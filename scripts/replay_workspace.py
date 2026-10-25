"""Execute all prepared workspace decision replays, without any actions.

Standalone dependencies: Private frozen .work/ cohort, ignored results, sibling projects and shared model runtimes.
See docs/standalone-dependencies.md.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab import workspace_replay as replay
from compare_new_decisions import IMAGE, MODELS, gpu_rows
from decision_lab.evaluation import write_json


def native(model, plan):
    from decision_lab.runtime import ManagedServer
    from decision_lab.llama_decision import LlamaDecisionClient
    output=replay.DIRECTORY/model
    output.mkdir(parents=True,exist_ok=True)
    clef=model=='clef-flash-q4'
    settings=dict(cache='f16',image=IMAGE,batch=plan['clef_context'] if clef else 1024,
        ubatch=plan['clef_context'] if clef else 512,parallel=1,
        ctx=plan['clef_context'] if clef else plan['context'])
    # Verify each retained weight against its earlier scored acquisition manifest.
    reference=json.loads((replay.OLD_RUNS/replay.OLD_LOCAL[model]/'manifest.json').read_text())['weight']
    if MODELS[model].stat().st_size!=reference['bytes'] or replay.sha256(MODELS[model])!=reference['sha256']:
        raise RuntimeError('Model weight identity changed')
    samples=[]
    with ManagedServer(MODELS[model],output/'runtime',**settings) as server:
        client=LlamaDecisionClient(server.base_url)
        smoke=client.evaluate('Cancel my order.',{'q':dict(type='choice',instructions='What does the customer want?',
            criteria={'cancel':'Cancel order','track':'Track order'})})
        if smoke!={'q':'cancel'}:raise RuntimeError('Semantic smoke failed')
        pid=json.loads((server.output/'gpu.json').read_text())['server_pid']
        stop=threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    used=gpu_rows().get(pid)
                    if used is not None:samples.append(dict(pid=pid,used_mib=used,time=time.time()))
                except (subprocess.CalledProcessError,ValueError):pass
                stop.wait(1)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:replay.execute(client,model,plan)
        finally:
            stop.set();thread.join()
            replay.save(output/'memory.json',dict(samples=samples,settings=settings,
                peak_process_mib=max((s['used_mib'] for s in samples),default=None),server_name=server.name))
    if subprocess.run(['podman','container','exists',server.name]).returncode!=1:
        raise RuntimeError('Owned native container remains')


def mica(plan):
    root=replay.ROOT
    reference=json.loads((replay.OLD_RUNS/replay.OLD_LOCAL['mica-01-q8']/'manifest.json').read_text())['weight']
    weight=MODELS['mica-01-q8']
    if weight.stat().st_size!=reference['bytes'] or replay.sha256(weight)!=reference['sha256']:
        raise RuntimeError('Mica weight identity changed')
    purpose='workspace-mica-'+uuid.uuid4().hex[:8]
    command=[str(root.parent/'tooling/godot-podman/bin/godot-podman'),'run','--project',str(root),
        '--purpose',purpose,'--image','localhost/asset-pipeline:local','--display','none',
        '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','USE_TF=0',
        '--env','LD_LIBRARY_PATH=/workspace/.local/llama-df18bee-runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64',
        '--','python3','scripts/replay_workspace.py','mica-worker']
    # The private frozen bundle is copied under this owning project. No sibling mounts or credentials.
    output=replay.DIRECTORY/'mica-01-q8';output.mkdir(parents=True,exist_ok=True)
    cid=None;samples=[]
    with (output/'runtime.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                found=subprocess.check_output(['podman','ps','-q','--no-trunc','--filter',
                    'label=io.smallloopworks.godot-podman.purpose='+purpose],text=True).strip()
                if found:
                    cid=found
                    try:
                        pids=subprocess.check_output(['podman','top',cid,'hpid'],text=True).splitlines()[1:]
                        memory=gpu_rows()
                        samples.extend(dict(pid=p.strip(),used_mib=memory[p.strip()],time=time.time())
                                       for p in pids if p.strip() in memory)
                    except subprocess.CalledProcessError:pass
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=120)
    if cid and subprocess.run(['podman','container','exists',cid]).returncode!=1:
        raise RuntimeError('Owned Mica container remains')
    replay.save(output/'memory.json',dict(container=cid,removed=True,samples=samples,
        peak_process_mib=max((s['used_mib'] for s in samples),default=None)))
    if process.returncode:raise RuntimeError('Mica replay worker failed; inspect private runtime log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','report','native','mica','mica-worker','jev'))
    parser.add_argument('--model',choices=('clef-flash-q4','kev-4b-q8'))
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        plan=replay.prepare()
        print(json.dumps(dict(jobs=len(plan['jobs']),models=plan['models'],groups=len({j['group'] for j in plan['jobs']}))))
        return
    if args.command=='report':replay.report();return
    if args.command=='mica-worker':
        # Container has the verified private bundle and source snapshots only.
        replay.DIRECTORY=Path('/workspace/.work/workspace-replay-20261015-v1')
        replay.OLD_RUNS=Path('/workspace/results/new-decisions-20261015-v1')
        plan=json.loads((replay.DIRECTORY/'plan.json').read_text())
        from decision_lab.mica_native import MicaClient
        client=MicaClient('/workspace/.local/mica-abi.json',context=plan['context'])
        try:replay.execute(client,'mica-01-q8',plan)
        finally:client.close()
        return
    plan=replay.load_plan()
    lockname='cloud-dispatch' if args.command=='jev' else 'local-gpu'
    with (replay.DIRECTORY/(lockname+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='native':
            if not args.model:parser.error('native requires --model')
            native(args.model,plan)
        elif args.command=='mica':mica(plan)
        else:
            if not args.live:parser.error('Jev requires explicit --live')
            from decision_lab.jev_campaign import JevComparator
            client=JevComparator(live=True)
            replay.execute(client,'jev-1.14.0',plan,live=True)


if __name__=='__main__':main()
