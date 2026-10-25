"""Resolve persisted model-specific settings without silently changing providers."""
import json
from pathlib import Path
from .llama import LlamaServerClient
from .evaluation import ROOT


def configured_llama(base_url=None):
    client=LlamaServerClient(base_url=base_url)
    if not client.model:
        models=client._request('/v1/models')['data']
        if len(models)!=1:raise ValueError('Set LLAMA_MODEL for a multi-model server')
        client.model=models[0]['id']
    path=ROOT/'configs/local-best.json'
    if not path.exists():return client,[]
    data=json.loads(path.read_text())
    for model in data['models']:
        if Path(client.model).name==model['filename']:
            client.profile=model['profile']
            return client,model.get('examples',[])
    return client,[]
