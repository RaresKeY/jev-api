import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from decision_lab.jev import JevClient
from decision_lab.env import load_env
from decision_lab.evaluation import evaluate_values
from decision_lab.contracts import validate
from decision_lab.llama import LlamaServerClient


class Adapters(unittest.TestCase):
    def test_jev_noul_and_fractional_score(self):
        questions={'ok':{'type':'noul','instructions':'ok?'},'risk':{'type':'score','instructions':'risk','criteria':['low','high']}}
        raw={'model':'jev-1.14.0','usage':{'input_tokens':12},'answers':{'ok':{'type':'noul','noul':.8},'risk':{'type':'score','score':.7,'confidence':.3,'legend':{'0':'low','1':'high'},'probabilities':{'0':.3,'1':.7}}}}
        with patch('decision_lab.jev.request_json',return_value=raw) as request:
            response=JevClient(key='dummy').system_one({},questions)
        self.assertEqual(request.call_count,1)
        self.assertEqual(response.nouls['ok'].probability,.8)
        self.assertEqual(response.scores['risk'].score,.7)
        self.assertEqual(response.scores['risk'].confidence,.3)

    def test_unexpected_provider_model_is_not_accepted(self):
        with patch('decision_lab.jev.request_json', return_value={'model':'jev-1.13.0','answers':{}}) as request:
            with self.assertRaisesRegex(ValueError, 'model mismatch'):
                JevClient(key='dummy').evaluate({}, {})
        self.assertEqual(request.call_count, 1)

    def test_no_retry_on_cloud_error(self):
        with patch('decision_lab.jev.request_json',side_effect=RuntimeError('HTTP 429')) as request:
            with self.assertRaises(RuntimeError):JevClient(key='dummy').evaluate({}, {})
        self.assertEqual(request.call_count,1)

    def test_environment_read_is_data_only(self):
        import os
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=True):
            path=Path(directory)/'.env'
            path.write_text('JEV_API="dummy$(false)"\nUNKNOWN=x\n')
            load_env(path)
            self.assertEqual(os.environ['JEV_API'],'dummy$(false)')
            self.assertNotIn('UNKNOWN',os.environ)

    def test_numeric_bool_rejected(self):
        with self.assertRaises(ValueError):validate({'ok':True},{'ok':{'type':'noul'}})

    def test_schema_request_matches_pinned_server_docs(self):
        client=LlamaServerClient(model='test',profile={'split':False})
        calls=[]
        def request(path,payload=None):
            calls.append(payload)
            return {'choices':[{'finish_reason':'stop','message':{'content':'{"x":"yes"}'}}]}
        client._request=request
        client.chat_json([],{'type':'object'})
        self.assertEqual(calls[0]['response_format'],{'type':'json_schema','json_schema':{'name':'decision','strict':True,'schema':{'type':'object'}}})
        self.assertFalse(calls[0]['cache_prompt'])

    def test_jev_attempt_guard_prevents_calls(self):
        from scripts.evaluate import jev
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'attempt.json').write_text('{}')
            with patch('decision_lab.jev.request_json') as request:
                with self.assertRaises(FileExistsError):jev(path)
            request.assert_not_called()

if __name__=='__main__':unittest.main()
