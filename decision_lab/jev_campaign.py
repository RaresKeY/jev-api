"""Current Jev comparator under the workspace's one cumulative 5M campaign."""
import json
from pathlib import Path
import sys

from .evaluation import ROOT, write_json
from .contracts import validate as scalar_validate
from .errors import FatalCloudError

PIPELINES = ROOT.parent / 'jev-pipelines'
sys.path.insert(0, str(PIPELINES))
from jev_pipelines import current_client as transport
from jev_pipelines.bounded import TotalLedger, RESERVE_INPUT, RESERVE_OUTPUT

CAMPAIGN = 'workspace-jev114-20261015'
AUTHORIZATION = PIPELINES / 'experiments' / CAMPAIGN / 'execution-authorization.json'


class DatasetLedger(TotalLedger):
    def reserve(self, identity, dataset, request_hash):
        with self.locked() as value:
            if value['halted']:
                raise FatalCloudError('Unknown usage previously halted this campaign')
            if identity in value['entries']:
                raise FatalCloudError('An existing attempt must never be resent')
            entries = list(value['entries'].values())
            if sum(e['total_charge'] for e in entries) + RESERVE_INPUT + RESERVE_OUTPUT > value['limit']:
                raise FatalCloudError('Shared 5M input plus output reservation ceiling reached')
            dataset_input = sum(e['input_tokens'] if e['status'] == 'settled' else RESERVE_INPUT
                                for e in entries if e['dataset'] == dataset)
            if dataset_input + RESERVE_INPUT > 2_000_000:
                raise FatalCloudError('Dataset cumulative 2M input reservation ceiling reached')
            value['entries'][identity] = dict(dataset=dataset, request_sha256=request_hash,
                status='reserved', input_tokens=0, output_tokens=0,
                total_charge=RESERVE_INPUT + RESERVE_OUTPUT)


def authorize():
    preparation = json.loads((AUTHORIZATION.parent / 'preparation.json').read_text())
    config = dict(schema='jev-bounded-campaign/v1', id=CAMPAIGN, total_token_limit=5_000_000,
        ledger_directory=preparation['ledger_directory'], model=transport.MODEL,
        per_dataset_input_limit=2_000_000,
        authorization='Explicit live comparison of the matched decision cohorts within the bounded campaign.',
        scope='Matched benchmark and workstation requests plus prepared workspace replays within one cumulative campaign.',
        automatic_retries=False, no_actions=True)
    if AUTHORIZATION.exists() and json.loads(AUTHORIZATION.read_text()) != config:
        raise RuntimeError('Existing campaign authorization differs; preserve it')
    if not AUTHORIZATION.exists():
        write_json(AUTHORIZATION, config)
    return config


class JevComparator:
    def __init__(self, live=False):
        if not live:
            raise RuntimeError('Paid comparator requires explicit --live')
        config = authorize()
        directory = Path(config['ledger_directory']).expanduser().resolve()
        self.ledger = DatasetLedger(directory, config)
        self.client = transport.Client(directory, self.ledger, ROOT / '.env', live=True)
        self.last_raw = None
        self.dataset = 'odysseus-gates'

    def send(self, state, questions, identity=None):
        self.last_raw = None
        request = dict(model=transport.MODEL, state=state, questions=questions)
        identity = identity or 'primary--' + transport.digest(request)
        try:
            saved = self.client.ask(identity, self.dataset, request)
            self.ledger.verify(identity, saved['dataset'], transport.digest(request),
                               transport.usage(saved['response']))
        except Exception:
            self.ledger.halt()
            raise FatalCloudError('Jev stopped; exact sanitized attempt and charge retained, no retry') from None
        self.last_raw = saved['response']
        if not saved['valid']:
            raise ValueError('Billed typed response failed validation')
        transport.validate(self.last_raw, request)
        return self.last_raw

    def evaluate(self, state, questions):
        raw = self.send(state, questions)
        values = {name: raw['answers'][name][{'noul':'noul','choice':'choice','score':'score'}[q['type']]]
                  for name, q in questions.items()}
        return scalar_validate(values, questions, fractional_scores=True)

    def distributions(self, state, questions):
        raw = self.send(state, questions)
        result = {}
        for name, question in questions.items():
            answer = raw['answers'][name]
            result[name] = ({'false':1-answer['noul'], 'true':answer['noul']}
                            if question['type'] == 'noul' else answer['probabilities'])
        return result
