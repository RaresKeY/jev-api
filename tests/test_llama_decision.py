import unittest
from unittest.mock import patch
from decision_lab.llama_decision import LlamaDecisionClient


class NativeDecisionTests(unittest.TestCase):
    def test_native_probabilities_and_fractional_score(self):
        q = {'c': {'type': 'choice', 'instructions': '', 'criteria': {'a': 'A', 'b': 'B'}},
             's': {'type': 'score', 'instructions': '', 'criteria': ['low', 'high']},
             'n': {'type': 'noul', 'instructions': ''}}
        raw = {'answers': {'c': {'type': 'choice', 'probabilities': {'a': .3, 'b': .7}},
                           's': {'type': 'score', 'probabilities': {'0': .2, '1': .8}},
                           'n': {'type': 'noul', 'noul': .9}}, 'usage': {'output_tokens': 0}}
        client = LlamaDecisionClient('http://127.0.0.1:8080')
        with patch('decision_lab.llama_decision.request_json', return_value=raw) as send:
            self.assertEqual(client.evaluate({'text': 'test'}, q), {'c': 'b', 's': .8, 'n': .9})
            self.assertEqual(send.call_args.args[1], {'state': {'text': 'test'}, 'questions': q})
        raw['answers']['n']['noul'] = float('nan')
        with patch('decision_lab.llama_decision.request_json', return_value=raw), self.assertRaises(ValueError):
            client.evaluate({}, q)

    def test_cloud_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            LlamaDecisionClient('https://api.typesafe.ai')


if __name__ == '__main__':
    unittest.main()
