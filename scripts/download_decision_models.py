#!/usr/bin/env python3
"""Fetch pinned public decision assets without credentials or executing upstream code.

Standalone dependencies: Network/TLS access to pinned public sources and local storage; no inference.
See docs/standalone-dependencies.md.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, sha256, write_json

VERDICT_REV = 'bff28567cff463b833bf044f351a8b7945d53e07'


def read(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'decision-lab-public-assets'}), timeout=60)


def hub(repo, revision, directory, selected=None):
    with read(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
        metadata = json.load(response)
    files = []
    for file in metadata['siblings']:
        name = file['rfilename']
        if name.startswith('.') or (selected and name not in selected) or name.endswith('.log'):
            continue
        files.append(dict(path=f'models/{directory}/{name}',
                          url=f"https://huggingface.co/{repo}/resolve/{metadata['sha']}/{name}",
                          bytes=file['size'], expected_sha256=file.get('lfs', {}).get('sha256'),
                          git_blob_sha1=None if file.get('lfs') else file['blobId']))
    return dict(repository=repo, revision=metadata['sha'], declared_license=metadata.get('cardData', {}).get('license'), files=files)


def plan():
    bundles = {
        'kev-0.8b': hub('jaredpalmer/kev-0.8b', 'bf75a6a8848ea6960ff2ed108d9ed44c2941174f', 'kev-0.8b'),
        'kev-base': hub('Qwen/Qwen3.5-0.8B-Base', 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68', 'qwen3.5-0.8b-base'),
        'laya-typed-decisions': hub('convaiinnovations/laya-typed-decisions', '1a793eb568e6718f15941d08f85432581df534e3', 'laya-typed-decisions'),
    }
    repo = 'Heman10x-NGU/openJev-verdict-2.0'
    prefix = f'https://raw.githubusercontent.com/{repo}/{VERDICT_REV}'
    with read(f'{prefix}/artifacts/verdict2-base/model.pt') as response:
        pointer = response.read().decode()
    digest = next(line.removeprefix('oid sha256:') for line in pointer.splitlines() if line.startswith('oid sha256:'))
    size = int(next(line.removeprefix('size ') for line in pointer.splitlines() if line.startswith('size ')))
    # The GitHub LFS batch service confirms this object does not exist on the
    # server. Preserve the pointer as evidence; it is not a checkpoint.
    files = [dict(path='models/openjev-verdict2/model.pt.lfs-pointer',
                  url=f'{prefix}/artifacts/verdict2-base/model.pt')]
    names = {'LICENSE': 'LICENSE', 'README.md': 'README.md',
             'artifacts/verdict2-base/tokenizer.json': 'tokenizer.json',
             'artifacts/verdict2-base/tokenizer_config.json': 'tokenizer_config.json',
             'artifacts/verdict2-base/dev_metrics.json': 'dev_metrics.json',
             'verdict2/model.py': 'upstream/model.py', 'verdict2/data.py': 'upstream/data.py',
             'verdict2/evaluate.py': 'upstream/evaluate.py'}
    for source, dest in names.items():
        files.append(dict(path=f'models/openjev-verdict2/{dest}', url=f'{prefix}/{source}'))
    # Encoder configuration only: the trained Verdict checkpoint contains the encoder tensors.
    bundles['verdict-encoder-config'] = hub('answerdotai/ModernBERT-base', 'main', 'openjev-verdict2/encoder', {'config.json', 'LICENSE'})
    bundles['openjev-verdict2-reference'] = dict(repository=repo, revision=VERDICT_REV,
        declared_license='Apache-2.0 (repository declaration)', files=files,
        checkpoint_status='unavailable; only metadata/source reference downloaded',
        missing_checkpoint=dict(expected_sha256=digest, bytes=size,
            reason='GitHub LFS download API returned 404: Object does not exist on the server; advertised Hugging Face repo unavailable.'))
    bundles['openjev-verdict'] = hub('heman10x/rlcd-modernbert-151m',
        '8af2496eb63c7fa66d7d234e1f62629380030eb4', 'openjev-verdict',
        {'README.md', 'model.safetensors', 'config.json', 'tokenizer.json',
         'tokenizer_config.json', 'calibrator.json', 'bundle_manifest.json', 'train_manifest.json'})
    # Record exact code revisions for the license and runtime references; no code is installed or executed.
    for name, repo in [('kev', 'jaredpalmer/kev'), ('laya', 'haddock-development/laya_new')]:
        with read(f'https://api.github.com/repos/{repo}/commits/main') as response:
            revision = json.load(response)['sha']
        folder = 'kev-0.8b' if name == 'kev' else 'laya-typed-decisions'
        bundles[name + '-runtime-reference'] = dict(repository=repo, revision=revision, files=[
            dict(path=f'models/{folder}/upstream/{file}', url=f'https://raw.githubusercontent.com/{repo}/{revision}/{file}')
            for file in ('LICENSE', 'README.md', 'pyproject.toml')])
    return bundles


def download(file):
    path = ROOT / file['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + '.download-part')
        with read(file['url']) as response, open(temporary, 'wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        target = temporary
    else:
        target = path
    size, digest = target.stat().st_size, sha256(target)
    if file.get('bytes') is not None and file['bytes'] != size:
        raise ValueError(f"Size mismatch: {file['path']}")
    if file.get('expected_sha256') and digest != file['expected_sha256']:
        raise ValueError(f"LFS SHA-256 mismatch: {file['path']}")
    if file.get('git_blob_sha1'):
        data = target.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != file['git_blob_sha1']:
            raise ValueError(f"Git blob mismatch: {file['path']}")
    if target != path:
        target.replace(path)
    print(f"Verified {file['path']}: {size:,} bytes", flush=True)
    return dict(file, bytes=size, sha256=digest, verified=True)


def main():
    bundles = plan()
    destination = ROOT / 'docs/downloads/decision-models.json'
    write_json(destination, dict(status='downloading', bundles=bundles))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [(bundle, [pool.submit(download, file) for file in bundle['files']])
                   for bundle in bundles.values()]
        for bundle, futures in pending:
            bundle['files'] = [future.result() for future in futures]
    total = sum(file['bytes'] for bundle in bundles.values() for file in bundle['files'])
    write_json(destination, dict(status='complete', total_bytes=total,
                                unavailable_checkpoints=['specialized Verdict 2.0'],
                                verdict_selection='heman10x/rlcd-modernbert-151m (general-purpose; distinct from unavailable Verdict 2.0)',
                                scope='Downloaded available assets only; no upstream code executed, dependencies installed, inference or Jev API calls.', bundles=bundles))
    print(f'Total verified download size: {total:,} bytes', flush=True)


if __name__ == '__main__':
    main()
