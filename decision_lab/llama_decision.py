"""Native llama.cpp decision probabilities; local endpoint only, no generation."""
import math
from urllib.parse import urlparse
from .http import request_json
from .specialists import TypedDistributionClient, labels


class LlamaDecisionClient(TypedDistributionClient):
    def __init__(self, base_url):
        url = urlparse(base_url)
        if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password:
            raise ValueError('Native decision tests require a local HTTP server')
        self.base_url = base_url.rstrip('/')
        self.last_raw = None

    def distributions(self, state, questions):
        self.last_raw = None
        raw = request_json(self.base_url+'/v1/systemone', dict(state=state, questions=questions))
        self.last_raw = raw
        answers = raw.get('answers', {})
        if set(answers) != set(questions):
            raise ValueError('Native decision answer IDs differ from the request')
        distributions = {}
        for name, question in questions.items():
            answer = answers[name]
            if answer.get('type') != question['type']:
                raise ValueError('Native decision answer type mismatch')
            if question['type'] == 'noul':
                p = answer.get('noul')
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('Invalid native noul probability')
                probabilities = {'false': 1-p, 'true': p}
            else:
                probabilities = answer.get('probabilities')
                if not isinstance(probabilities, dict) or set(probabilities) != set(labels(question)):
                    raise ValueError('Native decision probability keys differ from criteria')
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or abs(sum(probabilities.values())-1) > 1e-5:
                raise ValueError('Invalid native decision distribution')
            distributions[name] = probabilities
        if raw.get('usage', {}).get('output_tokens') != 0:
            raise ValueError('Native decision endpoint unexpectedly generated tokens')
        return distributions
