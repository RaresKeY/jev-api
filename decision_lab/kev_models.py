"""Pinned native Kev GGUF candidates and verified local asset lookup."""
import json
from .evaluation import ROOT, sha256

MODELS = {
    'q8': dict(label='Kev 4B Q8', repository='ggml-org/Kev-4B-GGUF',
               revision='d924f2e2c3872da8b8aaf3eb4453b4126deceb79', directory='kev-4b-gguf',
               filename='Kev-4B-Q8_0.gguf', manifest='kev-4b-q8.json'),
    'q4': dict(label='Kev 4B Q4', repository='ggml-org/Kev-4B-GGUF',
               revision='d924f2e2c3872da8b8aaf3eb4453b4126deceb79', directory='kev-4b-gguf',
               filename='Kev-4B-Q4_K_M.gguf', manifest='kev-4b-gguf.json'),
    '9b-q4': dict(label='Kev 9B Q4', repository='ggml-org/Kev-9B-GGUF',
                  revision='ec2bbfe6620218aee2e01cc93bb78dee2a96ed58', directory='kev-9b-gguf',
                  filename='Kev-9B-Q4_K_M.gguf', manifest='kev-9b-q4.json'),
}


def verified_model(name):
    spec = MODELS[name]
    download = json.loads((ROOT/'docs/downloads'/spec['manifest']).read_text())
    path = ROOT/'models'/spec['directory']/spec['filename']
    item = next(f for f in download['files'] if ROOT/f['path'] == path)
    if download['status'] != 'verified' or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
        raise ValueError('Kev GGUF does not match its verified download')
    return path, item
