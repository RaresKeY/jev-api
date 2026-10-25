import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_lab.public_benchmarks import select, score_case, summarize, jsonl, link_kev_groups
from decision_lab.public_trials import check_gates, run_client, wire, estimate, campaign_identity
from decision_lab.evaluation import sha256


def case(i='a', group='g', suite='semif-authored', target='z'):
    return dict(id=i, suite=suite, group=group, family='f', stratum='f',
        state='Visible input', questions={'decision': dict(type='choice', instructions='Select',
            criteria={'z': 'First option', 'a': 'Second option'})},
        expected={'decision': target}, metric='semif-ordered-family-balanced',
        provenance={'rationale': 'Never send this'}, source={'row': 0})


class PublicTrialTests(unittest.TestCase):
    def test_campaign_identity_preserves_original_ascii_wire_encoding(self):
        # Pinned output from the campaign transport, including Unicode escaping.
        request = dict(model='jev-1.14.0', state='café', questions={})
        self.assertEqual(campaign_identity(request),
                         'primary--b23d500300d76143ae043ab4232b4986583b750404281a90162821f4e8fd6134')

    def test_kev_controls_and_buried_parents_remain_in_same_selection_unit(self):
        rows = [case('base', 'base', 'kev-test'), case('buried', 'buried', 'kev-test'),
                case('unknown', 'unknown', 'kev-test'), case('control', 'control', 'kev-test')]
        for row in rows: row['provenance']['upstream_provenance'] = {'id': row['id']}
        rows[1]['provenance']['upstream_provenance']['parent_id'] = 'base'
        rows[2]['provenance']['upstream_provenance']['control_id'] = 'control'
        suites = {'kev-test': rows}; link_kev_groups(suites)
        self.assertEqual(rows[0]['group'], rows[1]['group'])
        self.assertEqual(rows[2]['group'], rows[3]['group'])
        self.assertEqual(len(select(suites, ['kev-test'], 2)), 2)

    def test_jsonl_preserves_unicode_line_separators_inside_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            original = {'text': 'first\u2028second\x85third'}
            path.write_text(json.dumps(original, ensure_ascii=False)+'\n')
            self.assertEqual(jsonl(path), [original])

    def test_related_groups_are_indivisible_across_suites(self):
        corpus = {'semif-authored': [case(f'a{i}', f'g{i//2}') for i in range(8)],
            'semif-perturbations': [case(f'p{i}', f'g{i}', 'semif-perturbations') for i in range(4)]}
        rows = select(corpus, list(corpus), 4)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(sum(r['group'] == g for r in rows) == 3 for g in {r['group'] for r in rows}))

    def test_subset_is_order_and_label_independent(self):
        corpus = {'s': [case(str(i), str(i), 's') for i in range(30)]}
        changed = copy.deepcopy(corpus)
        changed['s'].reverse()
        for row in changed['s']:
            row['expected']['decision'] = 'a'
        self.assertEqual([r['id'] for r in select(corpus, ['s'], 9)],
                         [r['id'] for r in select(changed, ['s'], 9)])

    def test_full_and_invalid_subset_controls(self):
        corpus = {'s': [case(str(i), 'same', 's') for i in range(3)]}
        with self.assertRaises(ValueError): select(corpus, ['s'], 2)
        self.assertEqual(len(select(corpus, ['s'], 0)), 3)
        with self.assertRaises(ValueError): select(corpus, ['s'], -1)

    def test_original_tie_rules_remain_distinct(self):
        row = case()
        values = {'decision': {'z': .5, 'a': .5}}
        self.assertTrue(score_case(row, values)['decision']['correct'])
        row['metric'] = 'jevbench-exact'
        self.assertFalse(score_case(row, values)['decision']['correct'])

    def test_invalid_probability_and_missing_outputs_count_as_errors(self):
        rows = [case('a'), case('b', target='a')]
        result = summarize(rows, [{'id': 'a', 'distributions': {'decision': {'z': .9, 'a': .1}}}])['semif-authored']
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['missing_requests'], 1)
        self.assertFalse(score_case(rows[0], {'decision': {'z': float('nan'), 'a': 0}})['decision']['valid'])

    def test_removed_evidence_has_no_invented_correctness(self):
        row = case(suite='kev-transfer-v9-test', target=None)
        row['metric'] = 'kev-ordered-native'
        row['provenance'].update(upstream_provenance={'source': 'unknowable', 'variant': 'clean', 'id': 'a'}, evidence_removed=True)
        result = summarize([row], [{'id': 'a', 'distributions': {'decision': {'z': .99, 'a': .01}}}])[row['suite']]
        self.assertEqual(result['fields'], 0)
        self.assertIsNone(result['accuracy'])
        self.assertEqual(result['unlabelled_diagnostics']['high_confidence_share'], 1)

    def test_budget_reservation_stops_before_another_paid_request(self):
        class Client:
            last_raw = None
            calls = 0
            def distributions(self, state, questions):
                self.calls += 1
                self.last_raw = {'usage': {'input_tokens': 100, 'output_tokens': 10}}
                return {'decision': {'z': .9, 'a': .1}}
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)/'ledger.json';ledger.write_text('{"entries":{}}')
            plan = {'selection': {}, 'cases_sha256': 'fixture', 'cases': [case('a'), case('b')]}
            client = Client()
            with patch('decision_lab.public_trials.LEDGER', ledger), patch('decision_lab.public_trials.budget_snapshot', return_value={}):
                result = run_client(client, plan, Path(directory)/'run', 'jev-1.14.0', 131072)
            self.assertEqual(client.calls, 1)
            manifest = json.loads((Path(directory)/'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertEqual(manifest['pending_requests'], 1)
            self.assertEqual(result['semif-authored']['accuracy'], .5)

    def test_gold_never_enters_model_wire(self):
        row = case()
        self.assertEqual(set(wire(row)), {'state', 'questions'})
        self.assertNotIn('expected', json.dumps(wire(row)))
        self.assertNotIn('rationale', json.dumps(wire(row)))

    def test_no_implicit_execution_or_implicit_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'; path.write_text('{}')
            plan = {'models': ['mica-01-q8', 'jev-1.14.0'], 'cases': [case()],
                    'estimate': {'upper_total_tokens_with_headroom': 1000}}
            good = sha256(path)
            for execute, approved, cap in [(False, good, 1), (True, 'wrong', 1), (True, good, None)]:
                with self.assertRaises(ValueError):
                    check_gates(path, plan, 'mica-01-q8', execute, approved, cap)
            check_gates(path, plan, 'mica-01-q8', True, good, 1)
            budget = {'remaining': 200000, 'halted': False, 'unresolved': 0}
            with patch('decision_lab.public_trials.budget_snapshot', return_value=budget):
                with self.assertRaises(ValueError):
                    check_gates(path, plan, 'jev-1.14.0', True, good, 1, False, 150000)
                with self.assertRaises(ValueError):
                    check_gates(path, plan, 'jev-1.14.0', True, good, 1, True, 250000)
                check_gates(path, plan, 'jev-1.14.0', True, good, 1, True, 150000)

    def test_runner_never_retries_and_retains_failure_denominator(self):
        class Client:
            last_raw = None
            calls = 0
            def distributions(self, state, questions):
                self.calls += 1
                if self.calls == 1: raise RuntimeError('fixture failure')
                self.last_raw = {'usage': {'output_tokens': 0}}
                return {'decision': {'z': .9, 'a': .1}}
        plan = {'selection': {}, 'cases_sha256': 'fixture', 'cases': [case('a'), case('b')]}
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            result = run_client(client, plan, Path(directory)/'run', 'mica-01-q8')
        self.assertEqual(client.calls, 2)
        self.assertEqual(result['semif-authored']['accuracy'], .5)
        self.assertEqual(result['semif-authored']['request_errors'], 1)

    def test_plan_estimates_include_output_and_headroom(self):
        # A source checkout has no ignored historical timing results.
        with tempfile.TemporaryDirectory() as directory, patch('decision_lab.public_trials.ROOT', Path(directory)):
            result = estimate([case()])
        self.assertGreater(result['provider_output_tokens_planning'], 0)
        self.assertGreater(result['total_tokens_with_25pct_headroom'], result['provider_input_tokens']['planning'])
        self.assertIsNone(result['runtime_minutes'])
        self.assertIsNone(result['native_three_models_minutes'])


if __name__ == '__main__': unittest.main()
