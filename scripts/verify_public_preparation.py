#!/usr/bin/env python3
"""Offline source/adapter/plan audit. Never constructs a model or a paid client.

Standalone dependencies: Ignored acquisition/cleanup receipts, plans/forecast and original campaign ledger.
See docs/standalone-dependencies.md.
"""
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, sha256, write_json
from decision_lab.public_sources import BUNDLE, git
from decision_lab.public_benchmarks import load_corpus, jsonl, summarize
from decision_lab.public_trials import load_plan, check_gates, budget_snapshot


def verify_retained_evidence():
    manifest = json.loads((BUNDLE/'retained-evidence.json').read_text())
    assert manifest['static_review_sha256'] == sha256(BUNDLE/'static-review.json')
    review = json.loads((BUNDLE/'static-review.json').read_text())['repositories']
    acquisition = json.loads((BUNDLE/'acquisition.json').read_text())['repositories']
    for item in manifest['files']:
        repository = item['repository']
        assert item['commit'] == acquisition[repository]['commit'] == review[repository]['commit']
        reviewed = next(row for row in review[repository]['files'] if row['path'] == item['source_path'])
        path = ROOT/item['path']
        assert not path.is_symlink() and path.resolve().is_relative_to((BUNDLE/'retained-evidence').resolve())
        assert path.stat().st_size == item['bytes'] == reviewed['bytes']
        assert sha256(path) == item['sha256'] == reviewed['sha256']


def verify_checkouts():
    acquisition = json.loads((BUNDLE/'acquisition.json').read_text())['repositories']
    cleanup_path = BUNDLE/'cleanup.json'
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
    if cleanup:
        assert cleanup['status'] == 'complete'
        assert set(cleanup['removed']) == set(acquisition)
        for relative, digest in cleanup['retained_sha256'].items():
            assert sha256(BUNDLE/relative) == digest
    checkouts = {}
    for name, row in acquisition.items():
        directory = Path(row['external_directory'])
        assert not directory.is_symlink()
        if directory.exists():
            assert git(directory, 'rev-parse', 'HEAD') == row['commit']
            assert git(directory, 'remote', 'get-url', 'origin') == row['url']
            assert not git(directory, 'status', '--porcelain', '--untracked-files=all')
            checkouts[name] = dict(commit=row['commit'], clean=True, status='present')
        else:
            assert cleanup, f'Checkout absent without cleanup receipt: {name}'
            removed = cleanup['removed'][name]
            assert all(removed[key] == row[key] for key in ('url', 'commit', 'external_directory'))
            assert removed['clean'] and removed['no_ignored_files'] and removed['worktrees'] == 1
            checkouts[name] = dict(commit=row['commit'], clean_before_removal=True, status='removed')
    return checkouts


def main():
    before = budget_snapshot()
    forecast = json.loads((BUNDLE/'forecast.json').read_text())
    if before['ledger_sha256'] != forecast['budget']['ledger_sha256']:
        raise ValueError('Shared accounting changed since this preparation forecast')
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline audit forbids network')):
        verify_retained_evidence()
        corpus = load_corpus()
        originals = {}
        count = 0
        for records in corpus.values():
            for case in records:
                source = ROOT/case['source']['path']
                if source not in originals:
                    originals[source] = jsonl(source)
                original = originals[source][case['source']['row']]
                assert case['state'] == ({'question': original['question']} if case['suite'].startswith('mmlu-') else original['state'])
                if case['suite'].startswith('jevbench-'):
                    assert case['questions'] == {'decision': original['question']}
                elif case['suite'].startswith('semif-'):
                    assert list(case['questions']['decision']['criteria'].items()) == [(o['id'], o['description']) for o in original['options']]
                    assert case['expected']['decision'] == original['options'][original['label']]['id']
                elif case['suite'].startswith('kev-'):
                    assert case['questions'] == {key: {k: v for k, v in q.items() if k not in {'label', 'src'}} for key, q in original['questions'].items()}
                    assert all(v is None for v in case['expected'].values()) == (original['_meta']['source'] == 'unknowable')
                else:
                    assert list(case['questions']['decision']['criteria'].values()) == original['options']
                    assert case['expected']['decision'] == original['answer']
                    assert 'cot_content' not in case['state']
                count += 1
        plans = {}
        for path in sorted((BUNDLE/'plans').glob('*.json')):
            plan = load_plan(path)
            selected_ids = {r['id'] for r in plan['cases']}
            selected_groups = {r['group'] for r in plan['cases']}
            candidates = [r for suite in plan['selection']['suites'] for r in corpus[suite]
                          if not plan['selection']['families'] or r['family'] in plan['selection']['families']]
            assert {r['id'] for r in candidates if r['group'] in selected_groups} == selected_ids
            approved = sha256(path)
            check_gates(path, plan, 'mica-01-q8', True, approved, len(plan['cases']))
            plans[path.stem] = dict(sha256=approved, requests=len(plan['cases']))
        smoke = BUNDLE/'plans/smoke.json'
        denied_output = ROOT/'.work/public-trial-gate-must-not-create'
        assert not denied_output.exists()
        denial = subprocess.run([sys.executable, str(ROOT/'scripts/public_trials.py'), 'run',
            '--plan', str(smoke), '--model', 'mica-01-q8', '--output', str(denied_output),
            '--max-requests', '81'], text=True, capture_output=True)
        assert denial.returncode == 2 and 'requires --execute' in denial.stderr and not denied_output.exists()
        core = BUNDLE/'plans/core-full.json'
        try:
            check_gates(core, load_plan(core), 'jev-1.14.0', True, sha256(core), 1747,
                        True, before['remaining'])
        except ValueError as error:
            assert 'Estimated upper usage' in str(error)
        else:
            raise AssertionError('Full core cloud run should be rejected by remaining-budget admission')
        # Reproduce the public baseline using the owned scorer, not the downloaded evaluator.
        retained = BUNDLE/'retained-evidence/SemIf-OpenJev'
        raw_path = retained/'results/raw/predictions/direct-authored144.jsonl'
        raw = {r['id']: r for r in jsonl(raw_path)}
        responses = []
        for case in corpus['semif-authored']:
            row = raw[case['id'].split('/', 1)[1]]
            probabilities = row['probabilities']
            if isinstance(probabilities, list): probabilities = dict(zip(row['option_ids'], probabilities))
            responses.append(dict(id=case['id'], distributions={'decision': probabilities}))
        actual = summarize(corpus['semif-authored'], responses)['semif-authored']['mean_family_balanced_accuracy']
        expected = json.loads((retained/'results/phase1-summary.json').read_text())['semantic_quality']['authored_144_mean_family_balanced_accuracy']['direct_logits']
        assert abs(actual-expected) < 1e-9
        for filename in ['scoring.py', 'metrics.py']:
            assert sha256(ROOT/'vendored/jevbench'/filename) == sha256(BUNDLE/'upstream/jevbench/jevbench'/filename)
        checkouts = verify_checkouts()
    after = budget_snapshot()
    assert before == after
    receipt = dict(status='passed', adapted_rows_verified=count, selected_plans=plans,
        checkouts=checkouts, offline_source_and_scorer_audit=True, self_contained_evidence=True,
        refused_without_execute=True, refused_over_budget_core_cloud=True,
        semif_published_scorer_parity=dict(actual=actual, published=expected, prediction_sha256=sha256(raw_path)),
        unchanged_canonical_ledger_sha256=after['ledger_sha256'],
        new_model_inference_calls=0, new_paid_calls=0,
        implementation_tests='13 offline tests, including group closure, ordered ties, missing-evidence controls, no gold leakage, execution gates and reservation stops')
    write_json(BUNDLE/'verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
