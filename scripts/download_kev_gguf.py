#!/usr/bin/env python3
"""Download selected, pinned native Kev GGUFs and verify published hashes.

Standalone dependencies: Network/TLS access to pinned public sources and local storage; no inference.
See docs/standalone-dependencies.md.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, write_json
from decision_lab.kev_models import MODELS
from scripts.download_decision_models import hub, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['9b-q4'])
    args = parser.parse_args()
    for name in args.models:
        spec = MODELS[name]
        bundle = hub(spec['repository'], spec['revision'], spec['directory'], {spec['filename'], 'README.md'})
        manifest = ROOT/'docs/downloads'/spec['manifest']
        write_json(manifest, dict(status='downloading', bundle=bundle))
        files = [download(f) for f in bundle['files']]
        write_json(manifest, dict(status='verified', bundle=bundle, files=files))
        print(name, 'verified', flush=True)


if __name__ == '__main__':
    main()
