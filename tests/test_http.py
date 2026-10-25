"""Credential boundaries tested against a loopback HTTP fixture, never a provider."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

from decision_lab.http import request_json


@contextmanager
def endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get('Authorization')))
            if self.path == '/redirect':
                self.send_response(302)
                self.send_header('Location', f'http://localhost:{self.server.server_port}/leak')
                self.end_headers()
            elif self.path == '/error':
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'private-response-body test-only-credential')
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class CredentialTransport(unittest.TestCase):
    def test_redirect_cannot_forward_credentials_or_make_another_request(self):
        with endpoint() as (base, seen):
            with self.assertRaisesRegex(RuntimeError, '^HTTP 302 from inference endpoint$'):
                request_json(base+'/redirect', key='test-only-credential')
        self.assertEqual(seen, [('/redirect', 'Bearer test-only-credential')])

    def test_http_error_does_not_expose_body_key_or_endpoint(self):
        with endpoint() as (base, seen):
            with self.assertRaises(RuntimeError) as caught:
                request_json(base+'/error', key='test-only-credential')
        self.assertEqual(str(caught.exception), 'HTTP 500 from inference endpoint')
        self.assertEqual(len(seen), 1)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_invalid_or_credential_bearing_urls_fail_before_transport(self):
        for url, key in [('http://example.invalid/path', 'test-only-credential'),
                         ('https://user:password@example.invalid/', None),
                         ('file:///tmp/fixture', None), ('http://[broken', None)]:
            with self.subTest(url=url), patch('decision_lab.http.build_opener') as send:
                with self.assertRaises(ValueError):
                    request_json(url, key=key)
                send.assert_not_called()

    def test_loopback_http_still_accepts_explicit_test_credentials(self):
        with endpoint() as (base, seen):
            self.assertEqual(request_json(base, key='test-only-credential'), {'ok': True})
        self.assertEqual(seen, [('/', 'Bearer test-only-credential')])


if __name__ == '__main__':
    unittest.main()
