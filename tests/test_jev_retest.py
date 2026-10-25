"""Request-budget and interruption guards for the one-pass cloud runner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_lab.evaluation import ROOT
from scripts.jev_retest import run


class JevRetestTests(unittest.TestCase):
    def test_existing_attempt_blocks_client_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'attempt.json').write_text('{"status":"stopped"}')
            with patch('scripts.jev_retest.JevClient') as client:
                with self.assertRaises(FileExistsError):
                    run(output)
                client.assert_not_called()

    def test_transport_failure_stops_after_one_call_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.jev_retest.load_env'), patch('scripts.jev_retest.JevClient') as factory, patch('builtins.print'):
                client = factory.return_value
                client.model = 'jev-latest'
                client.last_raw = None
                client.evaluate.side_effect = RuntimeError('Synthetic transport failure')
                attempt = run(Path(directory))
            self.assertEqual(client.evaluate.call_count, 1)
            self.assertEqual(attempt['requests'], 1)
            self.assertEqual(attempt['status'], 'stopped')

    def test_full_pass_is_exactly_64_and_sends_no_audit_labels(self):
        cases = json.loads((ROOT / 'benchmarks/odysseus-v2.json').read_text())['cases']
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.jev_retest.load_env'), patch('scripts.jev_retest.JevClient') as factory, patch('builtins.print'):
                client = factory.return_value
                client.model = 'jev-latest'
                client.last_raw = {'model': 'synthetic-test-model'}
                client.evaluate.side_effect = [dict(case['expected']) for case in cases]
                attempt = run(Path(directory))
            self.assertEqual(client.evaluate.call_count, 64)
            self.assertEqual(attempt['status'], 'complete')
            for call, case in zip(client.evaluate.call_args_list, cases):
                self.assertEqual(call.args, (case['state'], case['questions']))
            rows = [json.loads(line) for line in (Path(directory) / 'responses.jsonl').read_text().splitlines()]
            self.assertTrue(all(set(row['request']) == {'model', 'state', 'questions'} for row in rows))


if __name__ == '__main__':
    unittest.main()
