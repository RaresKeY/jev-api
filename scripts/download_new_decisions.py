"""Acquire pinned public Clef/Mica test weights and author source, without execution.

Standalone dependencies: Network/TLS access to pinned public sources and local storage; no inference.
See docs/standalone-dependencies.md.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'docs/downloads/new-decisions-20261015.json'
MODELS = {
    'clef-flash-q4': ('ggml-org/Clef-Flash-GGUF', '6b4fe4c59098479652d94f0fe11a40facbebd1ec',
                      ['Clef-Flash-Q4_K_M.gguf', 'README.md', '.gitattributes']),
    'mica-01-q8': ('sky7350/Mica-v0.1-4B', '9b6428b272f4708ef54a4702b52bb453a4059363',
                   ['mica-v0.1-4b-Q8_0.gguf', 'tokenizer.json', 'tokenizer_config.json',
                    'chat_template.jinja', 'config.json', 'calibration.json', 'README.md', 'NOTICE', 'LICENSE']),
}
MICA_SOURCE = '4f47a21bfc3ca9d1fd26d789c485ea24d7c04e2e'


def acquire(name):
    repo, revision, names = MODELS[name]
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
        metadata = json.load(response)
    if metadata['sha'] != revision:
        raise ValueError('Revision changed')
    siblings = {f['rfilename']: f for f in metadata['siblings']}
    files = []
    directory = ROOT / 'models' / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename in names:
        if filename not in siblings:
            continue
        record = siblings[filename]
        path = directory / filename
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            if partial.exists():
                raise RuntimeError('Partial download retained; inspect before another attempt')
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            print('Downloading', name, filename, flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            partial.replace(path)
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        expected = record.get('lfs', {}).get('sha256')
        if path.stat().st_size != record['size'] or expected and digest != expected:
            raise ValueError('Downloaded asset differs from publisher identity')
        if not expected:
            blob = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0' + path.read_bytes()).hexdigest()
            if blob != record['blobId']:
                raise ValueError('Small asset Git identity mismatch')
        files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest,
                          publisher_sha256=expected, publisher_git_blob=record.get('blobId')))
        print('Verified', name, filename, flush=True)
    return name, dict(repository=repo, revision=revision, status='verified', files=files)


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    acquired = dict(ThreadPoolExecutor(max_workers=2).map(acquire, MODELS))
    source_dir = ROOT / 'vendored/mica-v01'
    source_dir.mkdir(parents=True, exist_ok=True)
    source_files = []
    for name in ['mica/__init__.py', 'mica/direct_backend.py', 'mica/native.py', 'mica/codebook.py',
                 'mica/codebook_255.json', 'mica/calibration.py', 'mica/typesafe_server.py', 'LICENSE', 'NOTICE', 'calibration.json']:
        url = f'https://raw.githubusercontent.com/akivet/Mica/{MICA_SOURCE}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        path = source_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Existing pinned source changed')
        path.write_bytes(data)
        source_files.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), url=url))
    result = dict(status='verified', models=acquired,
                  mica_source=dict(repository='akivet/Mica', revision=MICA_SOURCE, files=source_files),
                  unavailable={'mica-0.5': 'Public model API returned 401; author public inventory only lists v0.1.'},
                  inference_calls=0)
    TARGET.write_text(json.dumps(result, indent=2) + '\n')
    print('Acquisition verified:', TARGET, flush=True)


if __name__ == '__main__':
    main()
