"""Official Jev HTTP adapter. Exactly one request per evaluate, no retries."""
import os
import math
from .http import request_json
from .contracts import descriptors, validate, response_object


class JevClient:
    def __init__(self, key=None, model='jev-1.14.0'):
        self.key = key or os.getenv('JEV_API') or os.getenv('TYPESAFE_API_KEY')
        if not self.key:
            raise RuntimeError('JEV_API or TYPESAFE_API_KEY is required')
        self.model = model
        self.last_raw = None

    def evaluate(self,state,questions):
        self.last_raw = None
        payload = dict(model=self.model,state=state,questions=questions)
        raw = request_json('https://api.typesafe.ai/v1/systemone',payload,self.key,timeout=60)
        self.last_raw = raw
        if raw.get('model') != self.model:
            raise ValueError('Jev response model mismatch')
        if set(raw.get('answers',{})) != set(questions):
            raise ValueError('Jev response has unexpected answers')
        values = {}
        for name, question in questions.items():
            answer = raw['answers'][name]
            kind = question['type']
            if answer.get('type') != kind:
                raise ValueError('Jev answer type mismatch')
            if kind in ('choice','score'):
                keys=set(question['criteria']) if kind=='choice' else {str(i) for i in range(len(question['criteria']))}
                probabilities=answer.get('probabilities')
                if not isinstance(probabilities,dict) or set(probabilities)!=keys:
                    raise ValueError('Invalid Jev probability keys')
                confidence=answer.get('confidence')
                numbers=[confidence,*probabilities.values()]
                if any(isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not 0<=n<=1 for n in numbers):
                    raise ValueError('Invalid Jev probability or confidence')
                if abs(sum(probabilities.values())-1)>.01:
                    raise ValueError('Jev probabilities do not sum to one')
                if kind=='score' and set(answer.get('legend',{}))!=keys:
                    raise ValueError('Invalid Jev score legend')
            values[name] = answer['noul' if kind=='noul' else 'choice' if kind=='choice' else 'score']
        return validate(values,questions,fractional_scores=True)

    def system_one(self,state,questions):
        questions = descriptors(questions)
        values = self.evaluate(state,questions)
        return response_object(values,questions,self.last_raw['model'],self.last_raw)
