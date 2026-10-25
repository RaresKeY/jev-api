#!/usr/bin/env python3
"""Extract reviewed data, freeze subsets and estimate offline; explicit gates for inference.

Standalone dependencies: Planning/execution needs local campaign/results and ../jev-pipelines; inference also needs explicit model runtimes/keys.
See docs/standalone-dependencies.md.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import write_json
from decision_lab.public_benchmarks import extract, load_corpus, select, DEFAULT_SUITES, summarize
from decision_lab.public_trials import make_plan, load_plan, estimate, check_gates, budget_snapshot, MODEL_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['extract', 'plan', 'estimate', 'forecast', 'run', 'report', '_mica-worker'])
    parser.add_argument('--suites', nargs='+')
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--limit-per-suite', type=int, default=12)
    parser.add_argument('--seed', default='public-trials-20261015')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--include-development', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=MODEL_IDS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approve-sha256')
    parser.add_argument('--max-requests', type=int)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--max-new-tokens', type=int)
    args = parser.parse_args()
    if args.action == 'extract':
        result = extract()
    elif args.action == 'plan':
        if args.output is None:
            parser.error('plan requires a fresh --output file')
        result = make_plan(args.output, args.suites, args.limit_per_suite, args.seed, args.families,
                           args.full, args.include_development)
    elif args.action == 'forecast':
        corpus = load_corpus()
        scenarios = {}
        for name, suites, limit in [('smoke', DEFAULT_SUITES, 12), ('pilot', DEFAULT_SUITES, 64),
            ('full-first-three', DEFAULT_SUITES[:-1], 0), ('full-with-mmlu', DEFAULT_SUITES, 0)]:
            scenarios[name] = estimate(select(corpus, suites, limit))
        result = dict(scenarios=scenarios, budget=budget_snapshot(), inference_executed=False)
        if args.output:
            if args.output.exists():
                parser.error('Preserve existing forecast; use a fresh output')
            write_json(args.output, result)
    else:
        if args.plan is None:
            parser.error('This action requires --plan')
        plan = load_plan(args.plan)
        if args.action == 'estimate':
            result = dict(estimate=estimate(plan['cases']), current_budget=budget_snapshot())
        elif args.action == 'report':
            if args.output is None:
                parser.error('report requires --output pointing to an existing run')
            path = args.output/'responses.jsonl'
            rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
            result = summarize(plan['cases'], rows)
            write_json(args.output/'summary.json', result)
        else:
            if args.output is None:
                parser.error('run requires a fresh --output directory')
            check_gates(args.plan, plan, args.model, args.execute, args.approve_sha256,
                        args.max_requests, args.live, args.max_new_tokens)
            from decision_lab.public_execution import execute
            result = execute(args.plan, plan, args.output, args.model, args.approve_sha256,
                args.max_requests, args.max_new_tokens, worker=args.action == '_mica-worker')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileExistsError) as error:
        print(f'Refused: {error}', file=sys.stderr)
        sys.exit(2)
