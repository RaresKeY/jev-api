#!/usr/bin/env python3
"""Test native Kev GGUFs against frozen local cases, never the cloud API.

Standalone dependencies: Ignored upstream cases/prior summaries and pinned weights; shared manager, Podman/CUDA image and NVIDIA tools.
See docs/standalone-dependencies.md.
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, evaluate_values, sha256, summarize, write_json
from decision_lab.runtime import ManagedServer
from decision_lab.llama_decision import LlamaDecisionClient
from decision_lab.kev_models import MODELS, verified_model
from scripts.upstream_benchmarks import run as run_upstream


def report(output, results):
    settings = json.loads((output/'manifest.json').read_text())
    baseline = json.loads((ROOT/'results/benchmark-readiness-v1/odysseus-summary.json').read_text())
    lines = ['# Native llama.cpp Kev results', '',
             'One local pass per quantization on frozen cases. No Jev API requests. Warmup and loading are excluded from latency.', '',
             '| Model | Correct fields / 214 | Exact cases / 64 | Unsafe approvals | Median ms | P95 ms | Process VRAM GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        s = result['odysseus']
        memory = json.loads((output/name/'memory.json').read_text())['peak_process_mib']
        lines.append(f"| {MODELS[name]['label']} | {s['field_correct']} ({s['field_accuracy']:.1%}) | {s['exact']} | {s['unsafe_approvals']} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} | {memory/1024:.2f} |")
    if '9b-q4' in results:
        saved = json.loads((ROOT/'results/kev-cpp-ab-v2/summary.json').read_text())
        for name, result in saved.items():
            s = result['odysseus']
            memory = json.loads((ROOT/f'results/kev-cpp-ab-v2/{name}/memory.json').read_text())['peak_process_mib']
            lines.append(f"| {MODELS[name]['label']} (saved) | {s['field_correct']} ({s['field_accuracy']:.1%}) | {s['exact']} | {s['unsafe_approvals']} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} | {memory/1024:.2f} |")
    for name in ['kev', 'Jev (saved)']:
        s = baseline[name]
        label = 'Kev 0.8B FP32 (saved)' if name == 'kev' else 'Jev cloud (saved)'
        lines.append(f"| {label} | {s['field_correct']} ({s['field_accuracy']:.1%}) | {s['exact']} | {s['unsafe_approvals']} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} | {'3.26' if name == 'kev' else '—'} |")
    old = {'Kev 0.8B (saved)': json.loads((ROOT/'results/benchmark-readiness-v1/upstream-summary.json').read_text())['kev']}
    if '9b-q4' in results:
        saved = json.loads((ROOT/'results/kev-cpp-ab-v2/summary.json').read_text())
        old = {**{MODELS[name]['label']+' (saved)': result['upstream'] for name, result in saved.items()}, **old}
    lines += ['', 'VRAM is sampled NVIDIA memory for the actual server PID, including CUDA/cache/workspace; sampling may miss short peaks. The older 0.8B memory run is separate. Latency includes local HTTP for GGUFs; saved native/cloud runs have different execution paths and host load, so timing comparisons are directional.', '',
              '## Original upstream smoke cases', '',
              'Each original suite contributes the same 12 fixed cases used earlier. This small heterogeneous sample is not a publisher leaderboard or a statistically reliable overall ranking.', '',
              '| Suite | '+ ' | '.join([*results, *old])+' |',
              '|---|'+'---:|'*(len(results)+len(old))]
    for suite in next(iter(results.values()))['upstream']:
        cells = []
        for result in results.values():
            s = result['upstream'][suite]
            cells.append(f"{s['correct']}/{s['fields']}"+(f"; {s['errors']} errors" if s['errors'] else ''))
        for groups in old.values():
            s = groups[suite]; cells.append(f"{s['correct']}/{s['fields']}")
        lines.append('| '+suite+' | '+' | '.join(cells)+' |')
    lines += ['', 'Synthetic Odysseus expectations, original benchmark labels, thresholds and score tolerances were unchanged. Responses retain distributions, native model identity and usage. Per-suite Brier/ECE/KL and ordinal metrics are in the upstream summaries. Failed requests remain incorrect; no retries. The historical default configuration was preserved.', '',
              f"Reproduce with `scripts/test_kev_cpp.py --models {' '.join(results)} --batch {settings['batch']} --parallel {settings.get('parallel', 1)} --ctx {settings.get('ctx', 4096)} --upstream-cases results/kev-cpp-input-v1/upstream-cases.json --output NEW_DIRECTORY` using the project virtual environment. Full original suite runs are available through `scripts/upstream_benchmarks.py run --backends {'kev-9b-q4' if '9b-q4' in results else 'kev-4b-q8'} --limit 0 --output NEW_DIRECTORY`.", '']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['q8'])
    parser.add_argument('--image', default='localhost/jev-kev-server:llama-cpp-a4cb4c6-cuda12.6.0-ubuntu22.04')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1024, help='Native decision option spans must fit within one logical batch')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--ctx', type=int, default=4096, help='Total context across slots; preserve enough context per question')
    parser.add_argument('--upstream-cases', type=Path, default=ROOT/'results/upstream-partial-v1/cases.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    suite_path = ROOT/'benchmarks/odysseus-v2.json'
    cases = json.loads(suite_path.read_text())['cases']
    upstream = json.loads(args.upstream_cases.read_text())
    for case in upstream:
        if sha256(ROOT/case['source']['path']) != case['source']['sha256']:
            raise ValueError('Original upstream benchmark source changed')
    manifest = dict(status='started', models=args.models, image=args.image, batch=args.batch, parallel=args.parallel, ctx=args.ctx,
                    suite_sha256=sha256(suite_path), upstream_case_sha256=sha256(args.upstream_cases),
                    cloud_requests=0, protocol='Native calibrated decision probabilities; 64 synthetic cases plus fixed upstream smoke cases; warmup excluded')
    image = json.loads(subprocess.check_output(['podman', 'image', 'inspect', args.image], text=True))[0]
    manifest['image_identity'] = dict(id=image['Id'], labels=image['Config'].get('Labels', {}))
    write_json(args.output/'manifest.json', manifest)
    write_json(args.output/'cases.json', dict(odysseus=cases, upstream=upstream))
    for source in [Path(__file__), ROOT/'scripts/upstream_benchmarks.py', *sorted((ROOT/'decision_lab').glob('*.py')), *sorted((ROOT/'vendored').rglob('*.py')), ROOT/'vendored/manifest.json']:
        target = args.output/'source-snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    results = {}
    for name in args.models:
        out = args.output/name
        out.mkdir()
        model, verified = verified_model(name)
        write_json(out/'model.json', dict(path=str(model.relative_to(ROOT)), sha256=verified['sha256'], bytes=model.stat().st_size))
        with ManagedServer(model, out/'runtime', cache='f16', image=args.image, batch=args.batch, parallel=args.parallel, ctx=args.ctx) as server:
            client = LlamaDecisionClient(server.base_url)
            warm = {'decision': dict(type='choice', instructions='What does the customer want?', criteria={'cancel': 'Cancel order', 'track': 'Track order'})}
            client.distributions('Cancel my order.', warm)
            write_json(out/'smoke.json', client.last_raw)
            stop = threading.Event()
            samples = []
            # PID is verified against NVIDIA allocation by ManagedServer.
            pid = json.loads((server.output/'gpu.json').read_text())['server_pid']
            def monitor():
                while not stop.is_set():
                    try:
                        listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
                        for line in listing.splitlines():
                            process, memory = map(str.strip, line.split(','))
                            if process == pid:
                                samples.append(dict(time=time.time(), used_mib=int(memory)))
                    except (subprocess.CalledProcessError, ValueError):
                        pass
                    stop.wait(.25)
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            try:
                rows = []
                with open(out/'odysseus-responses.jsonl', 'x') as file_out:
                    for case in cases:
                        t = time.perf_counter()
                        row = dict(id=case['id'], case=case)
                        try:
                            values = client.evaluate(case['state'], case['questions'])
                            row.update(values=values, fields=evaluate_values(values, case), raw=client.last_raw)
                        except Exception as e:
                            row.update(error=f'{type(e).__name__}: {str(e)[:500]}', fields=evaluate_values({}, case), raw=client.last_raw)
                        row['latency_ms'] = (time.perf_counter()-t)*1000
                        rows.append(row)
                        file_out.write(json.dumps(row)+'\n'); file_out.flush()
                        if len(rows) % 16 == 0:
                            print(name, 'Odysseus', len(rows), flush=True)
                workstation = summarize(rows)
                write_json(out/'odysseus-summary.json', workstation)
                upstream_out = out/'upstream'; upstream_out.mkdir()
                upstream_summary = run_upstream(client, upstream, upstream_out)
                results[name] = dict(odysseus=workstation, upstream=upstream_summary)
            finally:
                stop.set(); monitor_thread.join(timeout=5)
                write_json(out/'memory.json', dict(server_pid=pid, samples=samples,
                          peak_process_mib=max((s['used_mib'] for s in samples), default=None),
                          note='NVIDIA process VRAM sampled every 250ms; includes context/cache/workspace. Sampling can miss transient peaks. Startup is excluded.'))
            print(name, workstation, flush=True)
        write_json(args.output/'summary.json', results)
    manifest['status'] = 'complete'
    write_json(args.output/'manifest.json', manifest)
    report(args.output, results)


if __name__ == '__main__':
    main()
