"""Matched 64/228 comparison. Local inference or separately guarded --live Jev.

Standalone dependencies: Ignored 228-case cohort and prior evidence; models and shared GPU runtimes; Jev adds ../jev-pipelines and the original ledger.
See docs/standalone-dependencies.md.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, evaluate_values, summarize, write_json, sha256
from decision_lab.benchmarks import score_record, summarize_rows
from decision_lab.runtime import ManagedServer
from decision_lab.llama_decision import LlamaDecisionClient
from decision_lab.errors import FatalCloudError

IMAGE = 'localhost/llama-server-local:llama-cpp-df18bee-cuda12.6.0-ubuntu22.04'
UPSTREAM = ROOT / 'results/kev-cpp-input-v1/upstream-cases.json'
ODYSSEUS = ROOT / 'benchmarks/odysseus-v2.json'
MODELS = {
    'clef-flash-q4': ROOT / 'models/clef-flash-q4/Clef-Flash-Q4_K_M.gguf',
    'kev-4b-q8': ROOT / 'models/kev-4b-gguf/Kev-4B-Q8_0.gguf',
    'mica-01-q8': ROOT / 'models/mica-01-q8/mica-v0.1-4b-Q8_0.gguf',
}


def freeze(output, model):
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(ODYSSEUS.read_text())['cases']
    upstream = json.loads(UPSTREAM.read_text())
    for case in upstream:
        if sha256(ROOT / case['source']['path']) != case['source']['sha256']:
            raise RuntimeError('Original benchmark source changed')
    if len(cases) != 64 or len(upstream) != 228:
        raise RuntimeError('Matched cohort changed')
    write_json(output / 'cases.json', dict(odysseus=cases, upstream=upstream))
    sources = [Path(__file__), *sorted((ROOT/'decision_lab').glob('*.py')),
               *sorted((ROOT/'vendored/mica-v01').rglob('*.py'))]
    for source in sources:
        target = output / 'source-snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = dict(model=model, status='frozen', odysseus_sha256=sha256(ODYSSEUS),
        upstream_sha256=sha256(UPSTREAM), sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        no_retuning=True, no_actions=True, automatic_retries=False,
        scalar_scoring='Existing Odysseus threshold .5 and score tolerance .75.',
        upstream_scoring='Unchanged original distribution scorer, including lexical ties and ordinal winning level.')
    if model in MODELS:
        path = MODELS[model]
        manifest['weight'] = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path))
        if model == 'kev-4b-q8':
            reference = json.loads((ROOT/'docs/downloads/kev-4b-q8.json').read_text())['files']
        else:
            reference = json.loads((ROOT/'docs/downloads/new-decisions-20261015.json').read_text())['models'][model]['files']
        expected = next(row for row in reference if row['path'] == manifest['weight']['path'])
        if any(manifest['weight'][key] != expected[key] for key in ('bytes', 'sha256')):
            raise RuntimeError('Weight does not match verified publisher acquisition')
    write_json(output / 'manifest.json', manifest)
    return cases, upstream, manifest


def run_cases(client, output, cases, upstream, cloud=False):
    rows = []
    with (output/'odysseus-responses.jsonl').open('x') as handle:
        for index, case in enumerate(cases):
            row = dict(id=case['id'], case=case)
            start = time.perf_counter()
            try:
                if cloud:
                    client.dataset = 'odysseus-gates'
                values = client.evaluate(case['state'], case['questions'])
                row.update(values=values, fields=evaluate_values(values, case), raw=client.last_raw)
            except FatalCloudError:
                raise
            except Exception as error:
                row.update(error=type(error).__name__, fields=evaluate_values({}, case), raw=client.last_raw)
            row['latency_ms'] = (time.perf_counter()-start)*1000
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n'); handle.flush()
            if (index+1) % 16 == 0:
                print(output.name, 'Odysseus', index+1, flush=True)
    workstation = summarize(rows)
    write_json(output/'odysseus-summary.json', workstation)
    groups = defaultdict(list)
    with (output/'upstream-responses.jsonl').open('x') as handle:
        for index, case in enumerate(upstream):
            row = dict(id=case['id'], suite=case['suite'], case=case)
            start = time.perf_counter()
            try:
                if cloud:
                    client.dataset = case['suite']
                distributions = client.distributions(case['state'], case['questions'])
                row.update(fields=score_record(case, distributions), raw=client.last_raw)
                if cloud:
                    # Preserve explicit provider decisions as a separate diagnostic.
                    row['explicit_choice_mismatches'] = [name for name, q in case['questions'].items()
                        if q['type']=='choice' and client.last_raw['answers'][name]['choice'] !=
                        max(distributions[name], key=distributions[name].get)]
            except FatalCloudError:
                raise
            except Exception as error:
                row.update(error=type(error).__name__, fields=score_record(case, predictions={}), raw=client.last_raw)
            row['elapsed_ms'] = (time.perf_counter()-start)*1000
            groups[case['suite']].append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n'); handle.flush()
            if (index+1) % 36 == 0:
                print(output.name, 'upstream', index+1, flush=True)
    upstream_summary = {name:dict(summarize_rows(items), errors=sum('error' in r for r in items))
                        for name, items in groups.items()}
    result = dict(odysseus=workstation, upstream=upstream_summary,
        upstream_correct=sum(g['correct'] for g in upstream_summary.values()),
        upstream_fields=sum(g['fields'] for g in upstream_summary.values()),
        upstream_errors=sum(g['errors'] for g in upstream_summary.values()))
    write_json(output/'summary.json', result)
    print(output.name, 'complete', workstation['field_correct'], '/', workstation['fields'],
          'upstream', result['upstream_correct'], '/', result['upstream_fields'], flush=True)
    return result


def gpu_rows():
    listing = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    return {pid.strip(): int(memory.strip()) for line in listing.splitlines()
            for pid, memory in [line.split(',')]}


def native(model, output):
    cases, upstream, manifest = freeze(output, model)
    clef = model == 'clef-flash-q4'
    settings = dict(cache='f16', image=IMAGE, batch=4096 if clef else 1024,
                    ubatch=4096 if clef else 512, parallel=1, ctx=4096)
    manifest.update(settings=settings, image_identity=json.loads(subprocess.check_output(
        ['podman','image','inspect',IMAGE], text=True))[0]['Id'])
    write_json(output/'manifest.json', manifest)
    samples = []
    with ManagedServer(MODELS[model], output/'runtime', **settings) as server:
        client = LlamaDecisionClient(server.base_url)
        smoke = client.evaluate('Cancel my order.', {'q':dict(type='choice', instructions='What does the customer want?',
                     criteria={'cancel':'Cancel order','track':'Track order'})})
        if smoke != {'q':'cancel'}:
            raise RuntimeError('Semantic smoke failed')
        write_json(output/'smoke.json', client.last_raw)
        pid = str(server.props and json.loads((server.output/'gpu.json').read_text())['server_pid'])
        stop = threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    memory = gpu_rows().get(pid)
                    if memory is not None:
                        samples.append(dict(pid=pid, used_mib=memory, time=time.time()))
                except (subprocess.CalledProcessError, ValueError):
                    pass
                stop.wait(.25)
        worker = threading.Thread(target=monitor, daemon=True); worker.start()
        try:
            run_cases(client, output, cases, upstream)
        finally:
            stop.set(); worker.join()
            write_json(output/'memory.json', dict(server_pid=pid, samples=samples,
                peak_process_mib=max((r['used_mib'] for r in samples), default=None)))
    manifest['status'] = 'complete'; write_json(output/'manifest.json', manifest)


def mica_worker(output, seqs=1):
    cases, upstream, manifest = freeze(output, 'mica-01-q8')
    from decision_lab.mica_native import MicaClient
    client = MicaClient(ROOT/'.local/mica-abi.json', seqs=seqs)
    try:
        smoke = client.evaluate('Cancel my order.', {'q':dict(type='choice', instructions='What does the customer want?',
                      criteria={'cancel':'Cancel order','track':'Track order'})})
        if smoke != {'q':'cancel'}:
            raise RuntimeError('Semantic smoke failed')
        write_json(output/'smoke.json', dict(raw=client.last_raw, abi=client.abi, codebook=client.codebook))
        run_cases(client, output, cases, upstream)
    finally:
        client.close()
    manifest.update(status='complete', runtime='Pinned author direct logits readout, df18bee shared library ABI checked in C',
        settings=dict(context_per_question=4096, question_sequences=seqs, shared_prefix=seqs > 1,
                      ubatch=512, flash_attention=True, generation=False,
                      calibration='Shipped temperature, not fitted on test data'))
    write_json(output/'manifest.json', manifest)


def mica_launch(output, seqs=1):
    purpose = 'decision-mica-' + uuid.uuid4().hex[:8]
    runner = ROOT.parent/'tooling/godot-podman/bin/godot-podman'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(runner), 'run', '--project', str(ROOT), '--purpose', purpose,
        '--image', 'localhost/asset-pipeline:local', '--display', 'none',
        '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','USE_TF=0',
        '--env','LD_LIBRARY_PATH=/workspace/.local/llama-df18bee-runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64',
        '--','python3','scripts/compare_new_decisions.py','mica-worker',
        '--output','/workspace/'+str(output.relative_to(ROOT)), '--mica-seqs',str(seqs)]
    cid = None
    samples = []
    with (output.parent/(output.name+'-runtime.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            found = subprocess.check_output(['podman','ps','-q','--no-trunc','--filter',
                'label=io.smallloopworks.godot-podman.purpose='+purpose], text=True).strip()
            if found:
                cid = found
                try:
                    pids = subprocess.check_output(['podman','top',cid,'hpid'], text=True).splitlines()[1:]
                    memory = gpu_rows()
                    for pid in map(str.strip, pids):
                        if pid in memory:
                            samples.append(dict(pid=pid, used_mib=memory[pid], time=time.time()))
                except subprocess.CalledProcessError:
                    pass
            time.sleep(.25)
        if cid and subprocess.run(['podman','container','exists',cid]).returncode != 1:
            raise RuntimeError('Owned Mica container remains after runner exit')
        if process.returncode:
            raise RuntimeError('Mica worker failed; inspect its retained runtime log')
    if not samples:
        raise RuntimeError('No NVIDIA allocation verified for the actual Mica container processes')
    write_json(output/'memory.json', dict(container=cid, removed=True, samples=samples,
               peak_process_mib=max(r['used_mib'] for r in samples)))
    print('Mica complete; owned container removal verified.', flush=True)


def cloud(output, live):
    if not live:
        raise RuntimeError('Jev dispatch requires --live')
    from decision_lab.jev_campaign import JevComparator
    cases, upstream, manifest = freeze(output, 'jev-1.14.0')
    client = JevComparator(live=True)
    try:
        run_cases(client, output, cases, upstream, cloud=True)
        manifest['status'] = 'complete'
    finally:
        manifest['budget'] = client.ledger.summary()
        write_json(output/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('native','mica','mica-worker','jev'))
    parser.add_argument('--model', choices=('clef-flash-q4','kev-4b-q8'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--mica-seqs', type=int, choices=(1,8), default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'native':
        if not args.model:
            parser.error('native requires --model')
        native(args.model, output)
    elif args.command == 'mica':
        mica_launch(output, args.mica_seqs)
    elif args.command == 'mica-worker':
        mica_worker(output, args.mica_seqs)
    else:
        cloud(output, args.live)


if __name__ == '__main__':
    main()
