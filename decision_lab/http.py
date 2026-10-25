"""JSON HTTP without retries, credential-bearing error bodies or header logging."""
import json
from ipaddress import ip_address
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # urllib can forward Authorization to a different host on redirects.
        return None


def request_json(url, payload=None, key=None, timeout=120):
    try:
        target = urlsplit(url)
        if target.scheme not in {'http', 'https'} or not target.hostname or target.username or target.password:
            raise ValueError
        loopback = target.hostname == 'localhost'
        try:
            loopback = loopback or ip_address(target.hostname).is_loopback
        except ValueError:
            pass
        if key and target.scheme != 'https' and not loopback:
            raise ValueError
    except ValueError:
        raise ValueError('Use an HTTP(S) endpoint without URL credentials; keys require HTTPS or loopback HTTP') from None
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        code = error.code
        error.close()
        raise RuntimeError(f'HTTP {code} from inference endpoint') from None
    except (URLError, OSError):
        raise RuntimeError('Inference endpoint unavailable or timed out') from None
