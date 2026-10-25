"""Protect the evaluation population and prevent accidental label/demo leakage."""
import copy
import json
import unittest

from decision_lab.evaluation import ROOT, SUITE
from scripts.build_odysseus_suite import build
from scripts.retest import validate_suite


class OdysseusSuiteTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(SUITE.read_text())
        self.suite = json.loads((ROOT / 'benchmarks/odysseus-v2.json').read_text())

    def test_preserved_population_and_reproducible_labels(self):
        cases = validate_suite(self.suite, self.original)
        self.assertEqual(self.suite, build())
        self.assertEqual(len(cases), 64)
        self.assertEqual(sum(case['id'].startswith('odysseus-') for case in cases), 52)
        self.assertEqual(sum(case['split'] == 'dev' for case in self.original['cases']), 12)

    def test_rejects_relabeling_original_case(self):
        changed = copy.deepcopy(self.suite)
        changed['cases'][0]['expected']['compact'] = not changed['cases'][0]['expected']['compact']
        with self.assertRaises(ValueError):
            validate_suite(changed, self.original)

    def test_rejects_duplicate_and_invalid_new_label(self):
        changed = copy.deepcopy(self.suite)
        changed['cases'][-1]['id'] = changed['cases'][-2]['id']
        with self.assertRaises(ValueError):
            validate_suite(changed, self.original)
        changed = copy.deepcopy(self.suite)
        changed['cases'][-1]['expected']['strategy'] = 'invented-option'
        with self.assertRaises(ValueError):
            validate_suite(changed, self.original)

    def test_saved_examples_exclude_all_scored_cases(self):
        ids = {case['id'] for case in self.suite['cases']}
        config = json.loads((ROOT / 'configs/local-best.json').read_text())
        for model in config['models']:
            self.assertTrue(all(example['split'] == 'dev' and example['id'] not in ids
                                for example in model['examples']))


if __name__ == '__main__':
    unittest.main()
