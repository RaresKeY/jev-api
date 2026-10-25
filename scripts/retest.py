#!/usr/bin/env python3
"""Evaluate frozen local settings on a new suite; never call the Jev API.

Standalone dependencies: Ignored models and optional packages/shared runtime; historical comparisons read ignored results.
See docs/standalone-dependencies.md.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.contracts import validate
from decision_lab.evaluation import ROOT, SUITE, evaluate_values, sha256, summarize, write_json
from decision_lab.llama import LlamaServerClient
from decision_lab.runtime import ManagedServer
from scripts.evaluate import run_cases, smoke, subprocess_image


def validate_suite(suite, original):
    cases = suite['cases']
    old = [case for case in original['cases'] if case['split'] == 'holdout']
    if len(cases) != 64 or cases[:len(old)] != old:
        raise ValueError('Suite must preserve the 12 original evaluation cases and contain 64 cases')
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case ID')
    templates = {case['gate']: case['questions'] for case in original['cases']}
    for case in cases:
        if case['split'] != 'holdout' or case['questions'] != templates[case['gate']]:
            raise ValueError('Evaluation split or gate descriptors changed')
        expected = {key: float(value) if isinstance(value, bool) else value
                    for key, value in case['expected'].items()}
        validate(expected, case['questions'])
        if not isinstance(case['state'], dict) or not case['state']:
            raise ValueError('Missing case state')
        if case['id'].startswith('odysseus-') and not case.get('label_reason'):
            raise ValueError('New case lacks an auditable label rationale')
    return cases


def grouped(rows, original_ids):
    return dict(all_64=summarize(rows),
                original_12=summarize([row for row in rows if row['id'] in original_ids]),
                new_52=summarize([row for row in rows if row['id'] not in original_ids]))


def report(output, results, cloud):
    lines = ['# Frozen-settings Odysseus retest', '',
             '64 scored cases: 12 original evaluation cases plus 52 new synthetic cases. '
             'The original 12 development examples remain separate. No settings were selected using these results.', '',
             '| Backend | All 64 fields | Exact cases | New 52 fields | New exact cases | Unsafe approvals (all/new) | Median ms |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, groups in results.items():
        all_rows, new = groups['all_64'], groups['new_52']
        lines.append(f"| {name} | {all_rows['field_correct']}/{all_rows['fields']} ({all_rows['field_accuracy']:.1%}) | {all_rows['exact']}/64 | {new['field_correct']}/{new['fields']} ({new['field_accuracy']:.1%}) | {new['exact']}/52 | {all_rows['unsafe_approvals']}/{new['unsafe_approvals']} | {all_rows['median_ms']:.1f} |")
    lines += ['', '## Matched original 12 cases', '',
              'Saved Jev responses cover these 12 cases only. There is no Jev score for the new 52 or the full 64.', '',
              '| Backend | Correct fields | Exact cases | Unsafe approvals |', '|---|---:|---:|---:|']
    for name, groups in results.items():
        summary = groups['original_12']
        lines.append(f"| {name} | {summary['field_correct']}/{summary['fields']} ({summary['field_accuracy']:.1%}) | {summary['exact']}/12 | {summary['unsafe_approvals']} |")
    if cloud:
        summary = cloud['summary']
        lines.append(f"| Jev (saved) | {summary['field_correct']}/{summary['fields']} ({summary['field_accuracy']:.1%}) | {summary['exact']}/12 | {summary['unsafe_approvals']} |")
    lines += ['', '## Per-gate accuracy on the 52 new cases', '', '| Gate | ' + ' | '.join(results) + ' |',
              '|---|' + '---:|' * len(results)]
    for gate in sorted(next(iter(results.values()))['new_52']['by_gate']):
        cells = []
        for groups in results.values():
            metric = groups['new_52']['by_gate'][gate]
            cells.append(f"{metric['correct']}/{metric['total']} ({metric['correct']/metric['total']:.1%})")
        lines.append('| ' + gate + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Interpretation', '',
              '- Labels were authored and frozen before inference; no label or threshold changed after seeing outputs.',
              '- These are synthetic rubric labels assigned by the suite author, not independent human annotations or production traffic.',
              '- Earlier debugging exposed the original cases. The new-52 column is the useful generalization check.',
              '- Overall accuracy weights fields, not gates; exact cases require every requested field to pass.',
              '- Binary threshold and score tolerance are unchanged. NLI scores remain uncalibrated adaptations.',
              '- Unsafe approvals count safety-policy errors only; this is not a complete authorization audit.',
              '- Timing includes differing HTTP/batching paths and current host load; it is directional.',
              '- Jev was read from disk. No cloud client was constructed and no API requests were made.',
              '- Raw responses, per-field failures, runtime/GPU proof, frozen config and file hashes are saved beside this report.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    with open(output / 'metrics.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['backend', 'group', 'cases', 'valid', 'field_correct', 'fields', 'field_accuracy', 'exact', 'unsafe_approvals', 'median_ms', 'p95_ms'])
        writer.writeheader()
        for name, groups in results.items():
            for group, summary in groups.items():
                writer.writerow(dict(backend=name, group=group, **{key: summary[key] for key in writer.fieldnames if key not in ('backend', 'group')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT / 'benchmarks/odysseus-v2.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local-best.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    original = json.loads(SUITE.read_text())
    if suite['source_sha256'] != sha256(SUITE):
        raise ValueError('Original suite provenance hash changed')
    cases = validate_suite(suite, original)
    config = json.loads(args.config.read_text())
    original_ids = {case['id'] for case in original['cases'] if case['split'] == 'holdout'}
    for model in config['models']:
        if sha256(ROOT / model['path']) != model['sha256']:
            raise ValueError('Model weight hash changed')
        if any(example in cases or example['id'] in {case['id'] for case in cases} for example in model['examples']):
            raise ValueError('Scored case leaked into development examples')
    # A new output directory is required, so no old predictions are silently reused.
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / 'scripts/evaluate.py', *sorted((ROOT / 'decision_lab').glob('*.py'))]
    manifest = dict(status='started', suite_sha256=sha256(args.suite), config_sha256=sha256(args.config),
                    source_hashes={str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                    image=subprocess_image(), selection='Frozen previous settings; no tuning on these cases',
                    cases=64, original_cases=12, new_cases=52, cloud_requests=0)
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'suite.json', suite)
    write_json(args.output / 'frozen-config.json', config)
    results = {}
    failure_rows = []

    def save(name, rows):
        results[name] = grouped(rows, original_ids)
        for row in rows:
            for field, value in row.get('fields', {}).items():
                if not value['passed']:
                    failure_rows.append(dict(backend=name, case=row['id'], gate=row['case']['gate'], field=field,
                                             **value, label_reason=row['case'].get('label_reason', 'Original fixture label')))
        write_json(args.output / 'summary.json', results)
        print(name, json.dumps(results[name]), flush=True)

    for model in config['models']:
        model_out = args.output / model['id']
        model_out.mkdir()
        with ManagedServer(ROOT / model['path'], model_out / 'runtime', cache=model['cache']) as server:
            smoke(server, model['id'], model_out / 'smoke.json')
            client = LlamaServerClient(server.base_url, profile=model['profile'])
            rows = run_cases(client, cases, model_out / 'responses.jsonl', examples=model['examples'])
        save(model['id'], rows)

    from decision_lab.nli import NLIClient
    nli_out = args.output / 'nli-distilroberta-base'
    nli_out.mkdir()
    nli_path = ROOT / 'models/nli-distilroberta-base'
    write_json(nli_out / 'weight-hashes.json', {path.name: sha256(path) for path in sorted(nli_path.iterdir()) if path.is_file()})
    client = NLIClient(nli_path)
    rows = run_cases(client, cases, nli_out / 'responses.jsonl', threshold=config['nli']['threshold'], nli_profile=config['nli']['profile'])
    save('nli-distilroberta-base', rows)

    # Match case content, not just IDs, before presenting the old cloud comparison.
    cloud_path = ROOT / 'results/jev-reference-v1/responses.jsonl'
    cloud_rows = [json.loads(line) for line in cloud_path.read_text().splitlines()]
    matched = [row for row in cloud_rows if row['id'] in original_ids]
    if len(matched) != 12 or any(row['case'] != next(case for case in cases if case['id'] == row['id']) for row in matched):
        raise ValueError('Saved cloud cases do not match the preserved evaluation cases')
    rescored = [dict(row, fields=evaluate_values(row.get('values', {}), row['case'])) for row in matched]
    cloud = dict(source=str(cloud_path.relative_to(ROOT)), source_sha256=sha256(cloud_path),
                 coverage='Original 12 only; no new requests', summary=summarize(rescored))
    write_json(args.output / 'jev-saved-comparison.json', cloud)
    write_json(args.output / 'failures.json', failure_rows)
    report(args.output, results, cloud)
    manifest.update(status='complete', inference_errors=sum(group['all_64']['cases'] - group['all_64']['valid'] for group in results.values()))
    write_json(args.output / 'manifest.json', manifest)
    if manifest['inference_errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
