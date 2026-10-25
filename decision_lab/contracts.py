"""Backend-independent descriptors, normalized answers and strict validation."""
import math
from types import SimpleNamespace


def descriptors(questions):
    return {name: value if isinstance(value, dict) else
            value.model_dump(exclude_none=True) if hasattr(value, 'model_dump') else
            dict(vars(value), type=value.type) for name, value in questions.items()}


def schema_for(questions):
    properties = {}
    for name, question in questions.items():
        kind = question['type']
        if kind == 'choice':
            properties[name] = {'type': 'string', 'enum': list(question['criteria'])}
        elif kind == 'score':
            properties[name] = {'type': 'integer', 'minimum': 0, 'maximum': len(question['criteria']) - 1}
        elif kind == 'noul':
            properties[name] = {'type': 'number', 'minimum': 0, 'maximum': 1}
        else:
            raise ValueError('Unsupported question type')
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def validate(values, questions, fractional_scores=False):
    if not isinstance(values, dict) or set(values) != set(questions):
        raise ValueError('Response keys do not match requested questions')
    for name, question in questions.items():
        value = values[name]
        kind = question['type']
        if kind == 'choice':
            if not isinstance(value, str) or value not in question['criteria']:
                raise ValueError('Invalid choice answer')
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid numeric answer')
            maximum = len(question['criteria']) - 1 if kind == 'score' else 1
            if not 0 <= value <= maximum:
                raise ValueError('Answer out of range')
            if kind == 'score' and not fractional_scores and not isinstance(value, int):
                raise ValueError('Local score must be integer')
    return values


def response_object(values, questions, model=None, raw=None):
    choices, scores, nouls = {}, {}, {}
    for name, question in questions.items():
        kind = question['type']
        extra = (raw or {}).get('answers', {}).get(name, {})
        if kind == 'choice':
            choices[name] = SimpleNamespace(choice=values[name], confidence=extra.get('confidence'), probabilities=extra.get('probabilities'))
        elif kind == 'score':
            scores[name] = SimpleNamespace(score=values[name], confidence=extra.get('confidence'), probabilities=extra.get('probabilities'), legend=extra.get('legend', question['criteria']))
        else:
            nouls[name] = SimpleNamespace(probability=values[name])
    return SimpleNamespace(choices=choices, scores=scores, nouls=nouls, model=model, raw=raw)
