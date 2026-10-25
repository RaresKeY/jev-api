import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if not (Path(__file__).resolve().parents[2]/'jev-pipelines/jev_pipelines').is_dir():
    raise unittest.SkipTest('Workspace integration requires the unbundled sibling jev-pipelines project')

from decision_lab.jev_campaign import DatasetLedger, FatalCloudError, JevComparator


class CampaignGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = dict(id='test', total_token_limit=5_000_000,
                           ledger_directory=str(self.directory))
        self.ledger = DatasetLedger(self.directory, self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def test_unknown_charge_is_persistent_and_blocks_new_dispatch(self):
        self.ledger.reserve('first','suite','hash')
        self.ledger.halt()
        other = DatasetLedger(self.directory, self.config)
        self.assertEqual(other.summary()['charged_total'], 131072)
        with self.assertRaises(FatalCloudError):
            other.reserve('second','other','hash2')

    def test_global_cap_reserves_both_input_and_output(self):
        with self.ledger.locked() as value:
            value['entries']['old'] = dict(status='settled', dataset='other',
                input_tokens=4_900_000, output_tokens=0, total_charge=4_900_000)
        with self.assertRaises(FatalCloudError):
            self.ledger.reserve('new','suite','hash')
        self.assertEqual(self.ledger.summary()['attempts'], 1)

    def test_dataset_input_cap_is_cumulative(self):
        with self.ledger.locked() as value:
            value['entries']['old'] = dict(status='settled', dataset='suite',
                input_tokens=1_950_000, output_tokens=0, total_charge=1_950_000)
        with self.assertRaises(FatalCloudError):
            self.ledger.reserve('new','suite','hash')
        self.ledger.reserve('other','other-suite','hash2')

    def test_live_opt_in_precedes_key_loading(self):
        with patch('decision_lab.jev_campaign.authorize') as authorize:
            with self.assertRaises(RuntimeError):
                JevComparator()
        authorize.assert_not_called()

    def test_saved_request_is_reused_without_new_charge_or_http(self):
        questions = {'q':dict(type='noul', instructions='Is it true?')}
        raw = dict(model='jev-1.14.0', answers={'q':dict(type='noul', noul=.8)},
                   usage=dict(input_tokens=100, output_tokens=10))
        with patch('decision_lab.jev_campaign.authorize', return_value=self.config), \
             patch('jev_pipelines.current_client.read_key', return_value='test-only'), \
             patch('jev_pipelines.current_client.urllib.request.urlopen',
                   return_value=io.BytesIO(json.dumps(raw).encode())) as send:
            client = JevComparator(live=True)
            self.assertEqual(client.evaluate('state', questions), {'q':.8})
            self.assertEqual(client.evaluate('state', questions), {'q':.8})
            self.assertEqual(client.ledger.summary()['charged_total'], 110)
        self.assertEqual(send.call_count, 1)


if __name__ == '__main__':
    unittest.main()
