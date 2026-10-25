"""Owned adapters and frozen, group-preserving subsets of reviewed public data."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .evaluation import ROOT, sha256, write_json
from .public_sources import BUNDLE
from .benchmarks import record, validate_record, score_record, summarize_rows
from .specialists import labels

DEFAULT_SUITES = ['jevbench-easy', 'jevbench-standard', 'jevbench-hard',
    'semif-authored', 'semif-perturbations', 'kev-transfer-v9-test', 'mmlu-pro-test']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().split('\n') if line.strip()]


def verify_bundle():
    manifest = json.loads((BUNDLE/'source-manifest.json').read_text())
    if sha256(BUNDLE/'static-review.json') != manifest['static_review_sha256']:
        raise ValueError('Static review identity changed')
    files = list(manifest['files'])
    files += json.loads((BUNDLE/'parquet-manifest.json').read_text())['files']
    files += json.loads((BUNDLE/'mmlu-export-manifest.json').read_text())['files']
    if (BUNDLE/'catalog.json').exists():
        catalog = json.loads((BUNDLE/'catalog.json').read_text())
        if catalog['source_manifest_sha256'] != sha256(BUNDLE/'source-manifest.json'):
            raise ValueError('Catalog/source manifest binding changed')
    for item in files:
        path = ROOT/item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(BUNDLE.resolve()) or sha256(path) != item['sha256']:
            raise ValueError(f'Imported source hash changed: {item["path"]}')
    return files


def annotate(row, repository, relative, split, family, group, metric, **extras):
    acquisition = json.loads((BUNDLE/'acquisition.json').read_text())['repositories'][repository]
    url = acquisition['url'].removesuffix('.git')
    row.update(split=split, family=family, group=group, stratum=family, metric=metric,
        provenance=dict(repository=url, revision=acquisition['commit'], upstream_path=relative,
            original_row=row['source']['row'], original_file_sha256=row['source']['sha256'],
            license='MIT' if repository != 'kev' else 'Apache-2.0 wrapper; underlying source terms retained',
            author_selection_exposure=(
                'Mica author reports SemIf used in recipe selection' if repository == 'SemIf-OpenJev' else
                'Mica author reports Kev v9 used in recipe selection; exact split exposure unestablished' if repository == 'kev' else
                'Mica author reports roughly half of public hard used in recipe selection; item membership unavailable' if 'hard.jsonl' in relative else
                'No selection exposure established by inspected author disclosure; pretraining overlap unverified'),
            **extras))
    validate_record(row)
    return row


def link_kev_groups(suites):
    """Keep original groups plus explicit intact-control/parent links together."""
    rows = [r for name, records in suites.items() if name.startswith('kev-') for r in records]
    parent = {r['group']: r['group'] for r in rows}
    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    by_id = {}
    for row in rows:
        meta = row['provenance']['upstream_provenance']
        by_id[meta['id']] = row['group']
    for row in rows:
        meta = row['provenance']['upstream_provenance']
        for key in ['parent_id', 'control_id']:
            if meta.get(key) in by_id:
                a, b = root(row['group']), root(by_id[meta[key]])
                parent[max(a, b)] = min(a, b)
    for row in rows:
        row['provenance']['original_group_id'] = row['group']
        row['group'] = root(row['group'])


def extract():
    verify_bundle()
    suites = {}
    upstream = BUNDLE/'upstream'
    for tier, filename in [('easy', 'easy'), ('standard', 'original'), ('hard', 'hard')]:
        suite = 'jevbench-'+tier
        rel = f'datasets/public/{filename}.jsonl'
        path = upstream/'jevbench'/rel
        rows = []
        for i, original in enumerate(jsonl(path)):
            expected = original['expected']
            if original['question']['type'] == 'noul':
                expected = {'yes': 'true', 'no': 'false'}.get(expected, expected)
            row = record(suite, original['id'], original['state'], {'decision': original['question']},
                {'decision': str(expected) if expected is not None else None}, path, i)
            rows.append(annotate(row, 'jevbench', rel, 'public', original['family'],
                'jevbench/'+str(original.get('group') or original['id']), 'jevbench-exact', upstream_provenance=original.get('provenance')))
        suites[suite] = rows
    for part, filename in [('authored', 'authored144'), ('perturbations', 'perturbations108')]:
        suite = 'semif-'+part
        rel = f'benchmarks/data/{filename}.jsonl'
        path = upstream/'SemIf-OpenJev'/rel
        rows = []
        for i, original in enumerate(jsonl(path)):
            options = original['options']
            q = dict(type='choice', instructions=original['question'],
                     criteria={o['id']: o['description'] for o in options})
            row = record(suite, original['id'], original['state'], {'decision': q},
                {'decision': options[original['label']]['id']}, path, i)
            if original.get('target_distribution') is not None:
                target = original['target_distribution']
                row['gold'] = {'decision': target if isinstance(target, dict) else dict(zip(q['criteria'], target))}
            group = original['provenance'].get('source_group_id', original['group_id'])
            rows.append(annotate(row, 'SemIf-OpenJev', rel, original['split'], original['family'],
                'semif/'+group, 'semif-ordered-family-balanced', upstream_provenance=original['provenance'],
                original_group_id=original['group_id']))
        suites[suite] = rows
    for split in ['test', 'development']:
        suite = 'kev-transfer-v9-'+split
        rel = f'evals/v9/transfer-v9/{split}.jsonl'
        path = upstream/'kev'/rel
        rows = []
        for i, original in enumerate(jsonl(path)):
            questions, expected = {}, {}
            meta = original['_meta']
            for key, question in original['questions'].items():
                questions[key] = {k: v for k, v in question.items() if k not in {'label', 'src'}}
                target = question['label']
                expected[key] = None if meta['source'] == 'unknowable' else (
                    str(target).lower() if question['type'] == 'noul' else str(target))
            row = record(suite, f'{i}-{meta.get("id", i)}', original['state'], questions, expected, path, i)
            rows.append(annotate(row, 'kev', rel, split, meta['source'],
                'kev/'+str(meta.get('group_id', meta.get('id', i))), 'kev-ordered-native',
                upstream_provenance=meta, evidence_removed=meta['source'] == 'unknowable'))
        suites[suite] = rows
    for item in json.loads((BUNDLE/'mmlu-export-manifest.json').read_text())['files']:
        split = item['split']
        suite = 'mmlu-pro-'+split
        path = ROOT/item['path']
        rows = []
        for i, original in enumerate(jsonl(path)):
            option_ids = [chr(65+k) for k in range(len(original['options']))]
            q = dict(type='choice', instructions='Select the correct answer to the question in the state.',
                     criteria=dict(zip(option_ids, original['options'])))
            if option_ids[original['answer_index']] != original['answer']:
                raise ValueError('MMLU answer index/letter mismatch')
            row = record(suite, original['question_id'], {'question': original['question']},
                         {'decision': q}, {'decision': original['answer']}, path, i)
            rows.append(annotate(row, 'MMLU-Pro-data', f'data/{split}-00000-of-00001.parquet',
                split, original['category'], f'mmlu-pro/{split}/{original["question_id"]}',
                'direct-choice-adaptation', parquet_sha256=item['parquet_sha256'],
                original_source=original['src'], adaptation='Direct typed choice; CoT gold excluded, no few-shot examples'))
        suites[suite] = rows
    link_kev_groups(suites)
    catalog = {}
    for name, rows in suites.items():
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate adapted IDs')
        destination = BUNDLE/'cases'/f'{name}.jsonl'
        destination.parent.mkdir(exist_ok=True)
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
        if destination.exists() and destination.read_text() != payload:
            raise ValueError('Existing adapted corpus changed; preserve it and create a new version')
        destination.write_text(payload)
        catalog[name] = dict(path=str(destination.relative_to(ROOT)), sha256=sha256(destination),
            requests=len(rows), questions=sum(len(r['questions']) for r in rows),
            labelled_questions=sum(v is not None for r in rows for v in r['expected'].values()),
            groups=len({r['group'] for r in rows}), families=dict(Counter(r['family'] for r in rows)),
            original_split=rows[0]['split'], metric=rows[0]['metric'])
    write_json(BUNDLE/'catalog.json', dict(schema='owned-public-benchmarks/v1', suites=catalog,
        default_suites=DEFAULT_SUITES, source_manifest_sha256=sha256(BUNDLE/'source-manifest.json'),
        adapter_sha256=sha256(Path(__file__)), no_inference=True))
    return catalog


def load_corpus():
    verify_bundle()
    catalog = json.loads((BUNDLE/'catalog.json').read_text())
    result = {}
    for name, info in catalog['suites'].items():
        path = ROOT/info['path']
        if sha256(path) != info['sha256']:
            raise ValueError('Adapted data changed')
        result[name] = jsonl(path)
    return result


def select(corpus, suites, limit=12, seed='public-trials-20261015', families=None):
    """Select whole related groups without using labels or predictions; per-suite upper bound."""
    if type(limit) is not int or limit < 0:
        raise ValueError('Subset limit must be nonnegative; zero explicitly means full')
    if not suites or len(set(suites)) != len(suites) or set(suites)-set(corpus):
        raise ValueError('Select distinct known suites explicitly')
    groups = defaultdict(list)
    for suite in suites:
        for row in corpus[suite]:
            if families and row['family'] not in families:
                continue
            groups[row['group']].append(row)
    strata = defaultdict(list)
    for group, rows in groups.items():
        key = tuple(sorted({(r['suite'], r['family']) for r in rows}))
        strata[key].append((digest([seed, group]), group, rows))
    for key in strata:
        strata[key].sort(reverse=True)
    counts, selected = Counter(), []
    while any(strata.values()):
        for key in sorted(strata):
            if not strata[key]:
                continue
            _, _, rows = strata[key].pop()
            sizes = Counter(r['suite'] for r in rows)
            if limit and any(counts[s]+n > limit for s, n in sizes.items()):
                continue
            selected.extend(sorted(rows, key=lambda r: r['id']))
            counts.update(sizes)
    if not selected or any(not counts[s] for s in suites):
        raise ValueError('No complete group fits in at least one selected suite; increase the limit or adjust families')
    return selected


def score_case(case, distributions=None):
    fields = score_record(case, distributions=distributions) if distributions is not None else score_record(case, predictions={})
    if case['metric'] != 'jevbench-exact' and distributions is not None:
        for name, question in case['questions'].items():
            values = distributions.get(name)
            keys = labels(question)
            tolerance = 1e-4
            if (not isinstance(values, dict) or set(values) != set(keys) or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    or not 0 <= v <= 1 for v in values.values()) or abs(sum(values.values())-1) > tolerance):
                fields[name] = dict(valid=False, correct=False if case['expected'][name] is not None else None,
                                    error='Original ordered probability validation failed')
                continue
            chosen = max(keys, key=values.get)
            fields[name].update(valid=True, predicted=chosen, probs=values,
                correct=chosen == case['expected'][name] if case['expected'][name] is not None else None)
    for name, field in fields.items():
        if field.get('valid') and field.get('probs') and case['expected'][name] is not None:
            floor = 1e-9 if case['metric'].startswith('kev-') else 1e-12
            field['nll'] = -math.log(max(field['probs'][case['expected'][name]], floor))
    return fields


def summarize(cases, responses):
    by_id = {r['id']: r for r in responses}
    if len(by_id) != len(responses) or set(by_id)-{r['id'] for r in cases}:
        raise ValueError('Duplicate or unexpected response IDs')
    groups = defaultdict(list)
    for case in cases:
        response = by_id.get(case['id'], {})
        groups[case['suite']].append(dict(case=case, fields=score_case(case, response.get('distributions')),
            error=response.get('error') or ('missing response' if not response else None)))
    results = {}
    for suite, rows in groups.items():
        out = summarize_rows(rows)
        out.update(requests=len(rows), request_errors=sum(bool(r['error']) for r in rows),
                   missing_requests=sum(r['case']['id'] not in by_id for r in rows),
                   source_groups=len({r['case']['group'] for r in rows}))
        labelled = [f for row in rows for f in row['fields'].values() if f.get('correct') is not None]
        nll = [f['nll'] for f in labelled if 'nll' in f]
        out['nll_valid_distributions_only'] = sum(nll)/len(nll) if nll else None
        out['nll'] = out['nll_valid_distributions_only'] if len(nll) == len(labelled) else None
        out['invalid_labelled_questions'] = sum(not f.get('valid') for f in labelled)
        if suite.startswith('semif-'):
            recalls = defaultdict(list)
            for row in rows:
                for key, value in row['fields'].items():
                    recalls[(row['case']['family'], row['case']['expected'][key])].append(bool(value.get('correct')))
            family = defaultdict(list)
            for (name, _), values in recalls.items():
                family[name].append(sum(values)/len(values))
            out['mean_family_balanced_accuracy'] = sum(sum(v)/len(v) for v in family.values())/len(family)
        diagnostic = [f for row in rows for name, f in row['fields'].items()
                      if row['case']['expected'][name] is None and f.get('valid') and f.get('probs')]
        out['unlabelled_diagnostics'] = dict(valid=len(diagnostic),
            mean_max_probability=sum(max(f['probs'].values()) for f in diagnostic)/len(diagnostic) if diagnostic else None,
            high_confidence_share=sum(max(f['probs'].values()) >= .9 for f in diagnostic)/len(diagnostic) if diagnostic else None)
        if suite.startswith('kev-'):
            clean = [r for r in rows if r['case']['provenance']['upstream_provenance'].get('variant') == 'clean'
                     and not r['case']['provenance'].get('evidence_removed')]
            out['kev_clean_knowable'] = summarize_rows(clean)
            out['kev_clean_knowable']['scope'] = 'Selected clean knowable rows; separate from all-selected common metrics'
            controls = {}
            missing = []
            for row in rows:
                meta = row['case']['provenance']['upstream_provenance']
                fields = list(row['fields'].values())
                if not fields or not all(f.get('valid') and f.get('probs') for f in fields):
                    continue
                conf = sum(max(f['probs'].values()) for f in fields)/len(fields)
                if meta['source'] == 'unknowable_control': controls[meta['id']] = conf
                if meta['source'] == 'unknowable': missing.append((meta.get('control_id'), conf))
            drops = [controls[key]-conf for key, conf in missing if key in controls]
            out['unlabelled_diagnostics'].update(paired_controls=len(drops),
                paired_mean_confidence_drop=sum(drops)/len(drops) if drops else None)
        results[suite] = out
    return results
