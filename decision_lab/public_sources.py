"""Acquire and statically inventory external sources without importing their code."""
import ast
import concurrent.futures
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import urllib.parse

from .evaluation import ROOT, sha256, write_json

EXTERNAL = Path.home() / 'external-repos'
BUNDLE = ROOT / 'benchmarks/public-20261015'
REPOSITORIES = {
    'jevbench': 'https://github.com/fstandhartinger/jevbench.git',
    'SemIf-OpenJev': 'https://github.com/TheoLeeCJ/SemIf-OpenJev.git',
    'kev': 'https://github.com/jaredpalmer/kev.git',
    'MMLU-Pro': 'https://github.com/TIGER-AI-Lab/MMLU-Pro.git',
    'MMLU-Pro-data': 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro',
}
PATTERNS = {
    'process_execution': r'\bsubprocess\.|\bos\.system\(',
    'dynamic_code_or_pickle': r'\b(?:exec|eval)\(|\bpickle\.load|\btorch\.load',
    'remote_model_code': r'trust_remote_code\s*=\s*True',
    'network_or_download': r'https?://|\b(?:curl|wget)\b|snapshot_download|load_dataset\(',
    'retry_or_answer_selection': r'retry_wrong|retry_count|args\.retry',
    'package_lifecycle': r'"(?:preinstall|postinstall|prepare)"\s*:',
}


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def acquire():
    """Explicit network operation. Existing dirty or unexpected repos are preserved and rejected."""
    EXTERNAL.mkdir(exist_ok=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    identities = {}
    for name, url in REPOSITORIES.items():
        path = EXTERNAL / name
        if not path.exists():
            subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'filter.lfs.smudge=',
                '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false', 'clone',
                '--depth', '1', '--single-branch', url, str(path)], env=env, check=True)
        if path.is_symlink() or git(path, 'remote', 'get-url', 'origin') != url:
            raise ValueError(f'Unexpected external checkout: {name}')
        if git(path, 'status', '--porcelain'):
            raise ValueError(f'Preserve dirty external checkout: {name}')
        revision = git(path, 'rev-parse', 'HEAD')
        subprocess.run(['git', '-C', str(path), '-c', 'core.hooksPath=/dev/null',
                        'checkout', '--detach', revision], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        identities[name] = dict(url=url, commit=revision, external_directory=str(path))
    destination = BUNDLE / 'acquisition.json'
    if destination.exists() and json.loads(destination.read_text())['repositories'] != identities:
        raise ValueError('Pinned acquisition differs; use a new versioned bundle')
    write_json(destination, dict(schema='public-benchmark-sources/v1', repositories=identities,
        acquisition='Shallow Git clones, detached commits, LFS smudge and hooks disabled',
        upstream_code_executed=False))
    return identities


def scan():
    identities = json.loads((BUNDLE / 'acquisition.json').read_text())['repositories']
    repositories = {}
    for name, identity in identities.items():
        path = Path(identity['external_directory'])
        if git(path, 'rev-parse', 'HEAD') != identity['commit'] or git(path, 'status', '--porcelain'):
            raise ValueError(f'Checkout changed before review: {name}')
        entries, findings, errors = [], [], []
        for raw in subprocess.check_output(['git', '-C', str(path), 'ls-files', '-z']).split(b'\0'):
            if not raw:
                continue
            relative = raw.decode()
            file = path / relative
            if file.is_symlink():
                entries.append(dict(path=relative, kind='symlink', target=os.readlink(file)))
                continue
            payload = file.read_bytes()
            entries.append(dict(path=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            if file.suffix not in {'.py', '.sh', '.js', '.ts', '.json', '.toml', '.yaml', '.yml'} and file.name != 'Dockerfile':
                continue
            text = payload.decode('utf-8', errors='replace')
            if file.suffix == '.py':
                try:
                    ast.parse(text, filename=relative)
                except SyntaxError as error:
                    errors.append(dict(path=relative, line=error.lineno, error=error.msg))
            for kind, pattern in PATTERNS.items():
                lines = [i for i, line in enumerate(text.splitlines(), 1) if re.search(pattern, line)]
                if lines:
                    findings.append(dict(path=relative, category=kind, lines=lines))
        repositories[name] = dict(commit=identity['commit'], files=entries,
            extensions=dict(Counter(Path(e['path']).suffix or '(none)' for e in entries)),
            heuristic_findings=findings, python_parse_errors=errors)
    report = dict(schema='external-static-inventory/v1', repositories=repositories,
        upstream_code_executed=False, scope='Tracked-file hashes, symlinks, Python AST syntax, static capability searches',
        limitation='A static triage inventory, not a security certification. Data text remains untrusted.',
        disposition={
            'jevbench': 'Import public data only; compare scorer bytes with already reviewed vendored scorer.',
            'SemIf-OpenJev': 'Import authored/perturbation data; preserve ordered argmax and family-balanced reporting.',
            'kev': 'Import v9 split data/manifests only; preserve original labels and unknown-evidence controls.',
            'MMLU-Pro': 'Reference code only. Do not run remote-code loaders, provider clients, installers or retry-on-wrong paths.',
            'MMLU-Pro-data': 'Resolve only two pinned Parquet LFS objects; use a retained offline reader for conversion.',
        })
    write_json(BUNDLE / 'static-review.json', report)
    return {name: dict(files=len(v['files']), findings=len(v['heuristic_findings']),
                       parse_errors=len(v['python_parse_errors'])) for name, v in repositories.items()}


def export_sources():
    """Copy only reviewed data/scorer references. No imports from external checkouts."""
    review = json.loads((BUNDLE / 'static-review.json').read_text())
    acquisition = json.loads((BUNDLE / 'acquisition.json').read_text())
    selected = {
        'jevbench': ['LICENSE', 'datasets/public/easy.jsonl', 'datasets/public/original.jsonl',
                     'datasets/public/hard.jsonl', 'jevbench/scoring.py', 'jevbench/metrics.py'],
        'SemIf-OpenJev': ['LICENSE', 'benchmarks/data/authored144.jsonl',
            'benchmarks/data/perturbations108.jsonl', 'benchmarks/data/perturbations108-manifest.json',
            'benchmarks/evaluate.py', 'benchmarks/evaluate_perturbations.py'],
        'kev': ['LICENSE', 'evals/v9/transfer-v9/manifest.json',
                'evals/v9/transfer-v9/development.jsonl', 'evals/v9/transfer-v9/test.jsonl', 'kev/metrics.py'],
        'MMLU-Pro': ['LICENSE', 'README.md'],
        'MMLU-Pro-data': ['README.md', 'data/test-00000-of-00001.parquet',
                           'data/validation-00000-of-00001.parquet'],
    }
    manifest = []
    for name, paths in selected.items():
        identity = acquisition['repositories'][name]
        index = {v['path']: v for v in review['repositories'][name]['files']}
        for relative in paths:
            source = Path(identity['external_directory']) / relative
            expected = index[relative]
            if source.is_symlink() or sha256(source) != expected['sha256']:
                raise ValueError(f'Reviewed source changed: {name}/{relative}')
            destination = BUNDLE / 'upstream' / name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != expected['sha256']:
                raise ValueError('Preserve changed existing export')
            shutil.copyfile(source, destination)
            manifest.append(dict(repository=name, commit=identity['commit'], source_path=relative,
                path=str(destination.relative_to(ROOT)), bytes=destination.stat().st_size,
                sha256=sha256(destination), source_url=identity['url'].removesuffix('.git'),
                role='reference-only, never imported' if destination.suffix == '.py' else 'original data/notice'))
    write_json(BUNDLE / 'source-manifest.json', dict(files=manifest, static_review_sha256=sha256(BUNDLE/'static-review.json')))
    return manifest


def resolve_parquet():
    """Resolve explicit LFS objects into our bundle, leaving the clean external Git tree alone."""
    identity = json.loads((BUNDLE/'acquisition.json').read_text())['repositories']['MMLU-Pro-data']
    records = []
    for split in ['test', 'validation']:
        relative = f'data/{split}-00000-of-00001.parquet'
        pointer = BUNDLE/'upstream/MMLU-Pro-data'/relative
        text = pointer.read_text()
        checksum = re.search(r'^oid sha256:([a-f0-9]{64})$', text, re.M).group(1)
        size = int(re.search(r'^size ([0-9]+)$', text, re.M).group(1))
        destination = BUNDLE/'objects'/f'{split}.parquet'
        destination.parent.mkdir(exist_ok=True)
        url = f"{identity['url']}/resolve/{identity['commit']}/{relative}"
        if not destination.exists():
            temporary = destination.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size != size or sha256(temporary) != checksum:
                raise ValueError('LFS object hash/size mismatch')
            temporary.rename(destination)
        if destination.stat().st_size != size or sha256(destination) != checksum:
            raise ValueError('Existing LFS object differs from pinned pointer')
        records.append(dict(split=split, path=str(destination.relative_to(ROOT)), bytes=size,
            sha256=checksum, url=url, repository_commit=identity['commit'], pointer_sha256=sha256(pointer)))
    write_json(BUNDLE/'parquet-manifest.json', dict(files=records))
    return records


def export_mmlu_rows():
    """HF's read-only Parquet viewer export, checking X-Revision on every page.

    No new reader dependency is installed. Original LFS objects remain retained.
    The export's own hashes and page receipts distinguish it from Parquet bytes.
    """
    pinned = json.loads((BUNDLE/'acquisition.json').read_text())['repositories']['MMLU-Pro-data']['commit']
    objects = json.loads((BUNDLE/'parquet-manifest.json').read_text())['files']
    output = []
    for obj in objects:
        split = obj['split']
        def page(offset):
            query = urllib.parse.urlencode(dict(dataset='TIGER-Lab/MMLU-Pro', config='default',
                                                split=split, offset=offset, length=100))
            url = 'https://datasets-server.huggingface.co/rows?' + query
            with urllib.request.urlopen(url, timeout=45) as response:
                revision = response.headers.get('X-Revision')
                raw = response.read()
            if revision != pinned:
                raise ValueError('Dataset viewer revision differs from the pinned Parquet repository')
            data = json.loads(raw)
            if data['partial']:
                raise ValueError('Partial viewer export is not an original complete split')
            return offset, data, dict(url=url, revision=revision, response_sha256=hashlib.sha256(raw).hexdigest())
        first = page(0)
        count = first[1]['num_rows_total']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pages = [first, *pool.map(page, range(100, count, 100))]
        rows, receipts = [], []
        for offset, data, receipt in sorted(pages):
            if data['num_rows_total'] != count:
                raise ValueError('Viewer split size changed')
            for item in data['rows']:
                if item['row_idx'] != len(rows) or item.get('truncated_cells'):
                    raise ValueError('Viewer omitted, reordered or truncated an original row')
                rows.append(item['row'])
            receipts.append(dict(offset=offset, rows=len(data['rows']), **receipt))
        if len(rows) != count:
            raise ValueError('Incomplete split')
        destination = BUNDLE/'objects'/f'mmlu-{split}.jsonl'
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
        if destination.exists() and destination.read_text() != payload:
            raise ValueError('Preserve changed existing viewer export')
        destination.write_text(payload)
        output.append(dict(split=split, path=str(destination.relative_to(ROOT)), sha256=sha256(destination),
            rows=count, parquet_sha256=obj['sha256'], repository_commit=pinned, viewer_pages=receipts))
    write_json(BUNDLE/'mmlu-export-manifest.json', dict(method='HF rows API with exact X-Revision checks', files=output))
    return [{k: v for k, v in row.items() if k != 'viewer_pages'} for row in output]
