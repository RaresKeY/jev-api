"""Native Kev metrics and paired option-order probes, computed without cloud calls."""
import copy
from .benchmarks import labels
from vendored.kev.metrics import metrics,grouped_metrics,scored_rows


def kev_native(rows):
    converted=[];missing=0
    for row in rows:
        case=row['case']
        if not case['suite'].startswith('kev-'):continue
        meta=case['original'].get('_meta',{})
        for name,q in case['questions'].items():
            field=row['fields'][name];p=field.get('probs')
            if not field['valid'] or not p:
                missing+=1;continue
            keys=labels(q)
            converted.append(dict(p=[p[k] for k in keys],label=keys.index(case['expected'][name]),type=q['type'],keys=keys,
                 task=case['stratum']+'/'+name,source=case['stratum'],variant=meta.get('variant','clean'),id=case['id'],question=name,group=meta.get('group_id',case['id'])))
    population=scored_rows(converted)
    # Upstream refuses partial metrics after rejected examples. Preserve that boundary.
    if missing:return dict(status='incomplete-native-population',missing_fields=missing,available_fields=len(population))
    if not population:return dict(status='no-distribution-population')
    return dict(status='scored',clean_fields=len(population),all_variant_fields=len(converted),micro=metrics(population),per_task=grouped_metrics(population,'task'),note='Exact upstream metrics on clean knowable fields; served checkpoint temperature already applied. No temperature refit or synthesized distributions.')


def option_order_probe(case):
    """Reverse only choice option order; preserve text, labels, gold and semantics."""
    changed=copy.deepcopy(case);changed['id']=case['id']+'/reverse-options'
    for q in changed['questions'].values():
        if q['type']=='choice':q['criteria']=dict(reversed(list(q['criteria'].items())))
    changed['derived']=dict(transform='reverse choice criteria insertion order',source_case=case['id'],expectation='Original semantic target preserved; ties may expose original scorer ordering')
    return changed
