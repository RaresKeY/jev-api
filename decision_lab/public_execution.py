"""Admitted local runtimes and separately gated paid transport for public trials."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .evaluation import ROOT, sha256, write_json
from .public_trials import run_client

IMAGE = 'localhost/llama-server-local:llama-cpp-df18bee-cuda12.6.0-ubuntu22.04'
WEIGHTS = {
    'clef-flash-q4': 'models/clef-flash-q4/Clef-Flash-Q4_K_M.gguf',
    'mica-01-q8': 'models/mica-01-q8/mica-v0.1-4b-Q8_0.gguf',
    'kev-4b-q8': 'models/kev-4b-gguf/Kev-4B-Q8_0.gguf',
}


def verified_weight(model):
    path = ROOT/WEIGHTS[model]
    if model == 'kev-4b-q8':
        entries = json.loads((ROOT/'docs/downloads/kev-4b-q8.json').read_text())['files']
    else:
        entries = json.loads((ROOT/'docs/downloads/new-decisions-20261015.json').read_text())['models'][model]['files']
    expected = next(e for e in entries if e['path'] == WEIGHTS[model])
    if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError('Weight differs from its verified acquisition')
    return path, expected


def verify_image():
    metadata = json.loads(subprocess.check_output(['podman', 'image', 'inspect', IMAGE], text=True))[0]
    expected = json.loads((ROOT/'results/new-decisions-20261015-v1/clef-flash-q4-v2/manifest.json').read_text())['image_identity']
    if metadata['Id'] != expected:
        raise ValueError('Pinned runtime image identity changed')


def execute(plan_path, plan, output, model, approved, max_requests, max_new_tokens=None, worker=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use a fresh output directory inside the owning project')
    if model == 'jev-1.14.0':
        from .jev_campaign import JevComparator
        client = JevComparator(live=True)
        result = run_client(client, plan, output, model, max_new_tokens)
    elif model == 'mica-01-q8' and not worker:
        return launch_mica(Path(plan_path).resolve(), output, approved, max_requests)
    elif model == 'mica-01-q8':
        if os.environ.get('JEV_PUBLIC_MICA_WORKER') != approved or not Path('/run/.containerenv').exists():
            raise ValueError('Mica worker requires the managed container launcher')
        weight, identity = verified_weight(model)
        if not (ROOT/'.local/mica-abi.json').is_file():
            raise ValueError('Recreate the validated cache first: python scripts/prepare_mica_runtime.py')
        from .mica_native import MicaClient
        client = MicaClient(ROOT/'.local/mica-abi.json', seqs=1, context=16384)
        try:
            result = run_client(client, plan, output, model)
        finally:
            client.close()
        write_json(output/'model.json', dict(weight=identity, context=16384, sequences=1,
            physical_batch=512, calibration='Unchanged shipped temperature', generation=False))
    else:
        from .runtime import ManagedServer
        from .llama_decision import LlamaDecisionClient
        weight, identity = verified_weight(model)
        verify_image()
        clef = model == 'clef-flash-q4'
        settings = dict(cache='f16', image=IMAGE, batch=12288 if clef else 4096,
            ubatch=12288 if clef else 4096, ctx=12288 if clef else 16384, parallel=1)
        runtime = output.parent/(output.name+'-runtime')
        if runtime.exists():
            raise ValueError('Preserve existing runtime evidence; select a fresh output')
        with ManagedServer(weight, runtime, **settings) as server:
            result = run_client(LlamaDecisionClient(server.base_url), plan, output, model)
        write_json(output/'model.json', dict(weight=identity, settings=settings))
    shutil.copyfile(plan_path, output/'plan.json')
    return result


def launch_mica(plan_path, output, approved, max_requests):
    if not plan_path.is_relative_to(ROOT):
        raise ValueError('Managed Mica requires a plan under the project mount')
    purpose = 'public-mica-'+uuid.uuid4().hex[:8]
    runner = ROOT.parent/'tooling/godot-podman/bin/godot-podman'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(runner), 'run', '--project', str(ROOT), '--purpose', purpose,
        '--image', 'localhost/asset-pipeline:local', '--display', 'none',
        '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1', '--env', 'USE_TF=0',
        '--env', 'JEV_PUBLIC_MICA_WORKER='+approved,
        '--env', 'LD_LIBRARY_PATH=/workspace/.local/llama-df18bee-runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64',
        '--', 'python3', 'scripts/public_trials.py', '_mica-worker',
        '--plan', '/workspace/'+str(plan_path.relative_to(ROOT)),
        '--output', '/workspace/'+str(output.relative_to(ROOT)), '--model', 'mica-01-q8',
        '--execute', '--approve-sha256', approved, '--max-requests', str(max_requests)]
    from scripts.compare_new_decisions import gpu_rows
    samples, cid = [], None
    with (output.parent/(output.name+'-runtime.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                found = subprocess.check_output(['podman', 'ps', '-q', '--no-trunc', '--filter',
                    'label=io.smallloopworks.godot-podman.purpose='+purpose], text=True).strip()
                if found:
                    if '\n' in found:
                        raise RuntimeError('Ambiguous task-owned container identity')
                    cid = found
                    try:
                        pids = subprocess.check_output(['podman', 'top', cid, 'hpid'], text=True).splitlines()[1:]
                        memory = gpu_rows()
                        samples.extend(dict(pid=p.strip(), mib=memory[p.strip()], time=time.time())
                            for p in pids if p.strip() in memory)
                    except subprocess.CalledProcessError:
                        pass  # Container can exit between the listing and its PID sample.
                time.sleep(.25)
        except BaseException:
            process.terminate()
            process.wait(timeout=150)
            raise
    if cid and subprocess.run(['podman', 'container', 'exists', cid], stdout=subprocess.DEVNULL).returncode != 1:
        raise RuntimeError('Owned Mica container remains')
    if process.returncode or not samples:
        raise RuntimeError('Mica worker failed or had no verified CUDA process; inspect its runtime log')
    write_json(output/'memory.json', dict(container=cid, removed=True, samples=samples,
        peak_process_mib=max(v['mib'] for v in samples)))
    return json.loads((output/'summary.json').read_text())
