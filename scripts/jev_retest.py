#!/usr/bin/env python3
"""One explicit 64-case Jev run, followed by an offline comparison.

Standalone dependencies: Explicit Jev key for run; ignored matching local result evidence for report.
See docs/standalone-dependencies.md.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.env import load_env
from decision_lab.evaluation import ROOT, SUITE, evaluate_values, sha256, summarize, write_json
from decision_lab.jev import JevClient
from scripts.retest import grouped, validate_suite


def run(output, suite_path=ROOT / 'benchmarks/odysseus-v2.json'):
    output = Path(output)
    suite = json.loads(suite_path.read_text())
    if suite['source_sha256'] != sha256(SUITE):
        raise ValueError('Original suite provenance changed')
    cases = validate_suite(suite, json.loads(SUITE.read_text()))
    output.mkdir(parents=True, exist_ok=True)
    attempt = dict(status='started', suite_sha256=sha256(suite_path), maximum_requests=64,
                   requests=0, no_retries=True, authorization='Explicit run command for one full 64-case Jev evaluation.')
    # Existing attempts, including interrupted/failed runs, forbid new requests.
    with open(output / 'attempt.json', 'x') as file:
        json.dump(attempt, file, indent=2)
    write_json(output / 'suite.json', suite)
    write_json(output / 'source-hashes.json', {
        str(path.relative_to(ROOT)): sha256(path) for path in
        [Path(__file__), ROOT / 'decision_lab/jev.py', ROOT / 'decision_lab/http.py', ROOT / 'decision_lab/contracts.py']})
    rows = []
    try:
        load_env(ROOT / '.env')
        client = JevClient()
        with open(output / 'responses.jsonl', 'x') as file:
            for case in cases:
                start = time.perf_counter()
                row = dict(id=case['id'], case=case,
                           request=dict(model=client.model, state=case['state'], questions=case['questions']))
                # Reserve this request before transport so an interrupted call is counted.
                attempt['requests'] += 1
                write_json(output / 'attempt.json', attempt)
                try:
                    values = client.evaluate(case['state'], case['questions'])
                    row.update(values=values, fields=evaluate_values(values, case))
                except Exception as error:
                    row['error'] = str(error)
                row.update(raw=client.last_raw, latency_ms=(time.perf_counter() - start) * 1000)
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
                file.flush()
                rows.append(row)
                print(f"Jev {len(rows)}/64 {case['id']}: {'error; stopping without retry' if row.get('error') else 'saved'}", flush=True)
                if row.get('error'):
                    break
        attempt['status'] = 'complete' if len(rows) == 64 and all(not row.get('error') for row in rows) else 'stopped'
    except Exception:
        attempt['status'] = 'stopped'
        raise
    finally:
        write_json(output / 'summary.json', summarize(rows))
        write_json(output / 'attempt.json', attempt)
    return attempt


def compare(output, local=ROOT / 'results/odysseus-64-v1'):
    """Read saved evidence only; never initialize a client."""
    output, local = Path(output), Path(local)
    attempt = json.loads((output / 'attempt.json').read_text())
    if attempt['status'] != 'complete':
        raise ValueError('Full comparison requires a complete Jev pass')
    suite = json.loads((output / 'suite.json').read_text())
    if suite != json.loads((local / 'suite.json').read_text()):
        raise ValueError('Local and cloud suites differ')
    cases = suite['cases']
    original_ids = {case['id'] for case in cases[:12]}
    results, failures = {}, []
    models, token_totals = set(), {}
    for name, path in [
        ('Jev', output / 'responses.jsonl'),
        *[(name, local / name / 'responses.jsonl') for name in json.loads((local / 'summary.json').read_text())],
    ]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 64 or len({row['id'] for row in rows}) != 64 or any(row['case'] != case for row, case in zip(rows, cases)):
            raise ValueError('Saved predictions do not cover the same ordered 64 cases')
        rescored = [dict(row, fields=evaluate_values(row.get('values', {}), row['case'])) for row in rows]
        results[name] = grouped(rescored, original_ids)
        for row in rescored:
            for field, result in row['fields'].items():
                if not result['passed']:
                    failures.append(dict(backend=name, case=row['id'], gate=row['case']['gate'],
                                         field=field, **result,
                                         label_reason=row['case'].get('label_reason', 'Original fixture label')))
            if name == 'Jev':
                models.add(row['raw']['model'])
                for key, value in row['raw'].get('usage', {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        token_totals[key] = token_totals.get(key, 0) + value
    write_json(output / 'comparison.json', dict(results=results, model_ids=sorted(models), usage_totals=token_totals,
                                               requests=attempt['requests'], local_source=str(local.relative_to(ROOT))))
    write_json(output / 'failures.json', failures)
    lines = ['# Full 64-case Jev comparison', '',
             'One newly authorized Jev pass across all 64 frozen cases; 64 requests, no retries. '
             'Existing local predictions are reused without new local inference. Human expectations remain the reference.', '',
             '| Backend | Correct fields (all 64) | Fully correct cases | New 52 field accuracy | Unsafe approvals | Median ms |',
             '|---|---:|---:|---:|---:|---:|']
    for name, groups in results.items():
        all_rows, new = groups['all_64'], groups['new_52']
        lines.append(f"| {name} | {all_rows['field_correct']}/{all_rows['fields']} ({all_rows['field_accuracy']:.1%}) | {all_rows['exact']}/64 | {new['field_accuracy']:.1%} | {all_rows['unsafe_approvals']} | {all_rows['median_ms']:.1f} |")
    lines += ['', '## All 64 cases by gate', '', '| Gate | ' + ' | '.join(results) + ' |',
              '|---|' + '---:|' * len(results)]
    for gate in sorted(results['Jev']['all_64']['by_gate']):
        cells = []
        for groups in results.values():
            metric = groups['all_64']['by_gate'][gate]
            cells.append(f"{metric['correct']}/{metric['total']} ({metric['correct']/metric['total']:.1%})")
        lines.append('| ' + gate + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Evidence and limits', '',
              f"- Resolved Jev model(s): {', '.join(sorted(models))}.",
              f'- Reported usage totals: `{json.dumps(token_totals, sort_keys=True)}`.',
              '- Labels, thresholds and score tolerance were unchanged after inference. Audit rationales and expected labels were excluded from API requests.',
              '- Synthetic author-assigned labels are not independent annotations or production traffic; the original 12 cases were exposed during earlier debugging.',
              '- Unsafe approvals count the safety-policy field only, not every possible authorization error.',
              '- Local and cloud runs occurred at different times with different transports; latency is directional.',
              '- The earlier 24-request Jev reference and original local reports remain unchanged.',
              '- This report is generated offline from saved responses; rerendering it makes no API calls.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    with open(output / 'metrics.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=['backend', 'group', 'cases', 'valid', 'field_correct', 'fields', 'field_accuracy', 'exact', 'unsafe_approvals', 'median_ms', 'p95_ms'])
        writer.writeheader()
        for name, groups in results.items():
            for group, summary in groups.items():
                writer.writerow(dict(backend=name, group=group, **{key: summary[key] for key in writer.fieldnames if key not in ('backend', 'group')}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'run':
        attempt = run(args.output)
        if attempt['status'] != 'complete':
            raise SystemExit(2)
    results = compare(args.output)
    print(json.dumps(results['Jev'], indent=2))


if __name__ == '__main__':
    main()
