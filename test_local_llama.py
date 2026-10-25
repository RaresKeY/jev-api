"""HTTP contract and failure behavior, independent of model quality."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from local_llama import LlamaServerClient
from legacy.odysseus_workstation_sim import MockTypeSafeClient, Choice, Score, Noul


class ContractTests(unittest.TestCase):
    def test_http_payload_and_answer_mapping(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, value):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())
            def do_GET(self):
                self.reply({'data': [{'id': 'test-model'}]})
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.reply({'choices': [{'finish_reason': 'stop', 'message': {'content': '{"intent":"yes","risk":3,"destructive":0.9}'}}]})
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = LlamaServerClient(f'http://127.0.0.1:{server.server_port}/v1', profile={'split':False})
            result = client.system_one({'command': 'test'}, {
                'intent': Choice(instructions='Choose', criteria={'yes': 'Yes', 'no': 'No'}),
                'risk': Score(instructions='Risk', criteria=['safe', 'low', 'medium', 'high']),
                'destructive': Noul(instructions='Destructive?'),
            })
            self.assertEqual(result.choices['intent'].choice, 'yes')
            self.assertIsNone(result.choices['intent'].confidence)
            self.assertEqual(result.scores['risk'].score, 3)
            self.assertEqual(result.nouls['destructive'].probability, .9)
            self.assertEqual(requests[0]['model'], 'test-model')
            self.assertEqual(requests[0]['response_format']['json_schema']['schema']['properties']['risk']['maximum'], 3)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_reject_invalid_answers(self):
        question = {'value': SimpleNamespace(type='noul', instructions='test')}
        for answers in ({}, {'value': True}, {'value': 2}, {'value': float('nan')}, {'value': .5, 'extra': 1}):
            client = LlamaServerClient(model='test')
            client.chat_json = lambda *args, result=answers: result
            with self.assertRaises(ValueError):
                client.system_one({}, question)

    def test_reject_invalid_choice_score_and_truncation(self):
        for kind, criteria, value in [('choice', {'yes': 'Yes'}, []), ('score', ['safe', 'high'], 1.5), ('score', ['safe', 'high'], 2)]:
            client = LlamaServerClient(model='test')
            client.chat_json = lambda *args, value=value: {'value': value}
            question = SimpleNamespace(type=kind, instructions='test', criteria=criteria)
            with self.assertRaises(ValueError):
                client.system_one({}, {'value': question})
        client = LlamaServerClient(model='test')
        client._request = lambda *args: {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}
        with self.assertRaises(ValueError):
            client.chat_json([], {})

    def test_mock_score_contract(self):
        response = MockTypeSafeClient().system_one({'command': 'rm -rf', 'task': 'race condition'}, {
            'risk': Score(instructions='Risk', criteria=['safe', 'low', 'medium', 'high', 'critical']),
            'complexity': Score(instructions='Difficulty', criteria=['zero', 'one', 'two', 'three', 'four']),
        })
        for answer in response.scores.values():
            self.assertLessEqual(answer.score, 4)
            self.assertAlmostEqual(sum(answer.probabilities), 1)


if __name__ == '__main__':
    unittest.main()
