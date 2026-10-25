#!/usr/bin/env python3
"""Run the Mica semantic smoke explicitly; importing never runs inference.

Standalone dependencies: Verified ignored Mica assets,
.local/mica-abi.json, pinned native libraries and optional Python packages.
See docs/standalone-dependencies.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    abi = ROOT/'.local/mica-abi.json'
    if not abi.is_file():
        parser.exit(2, 'Missing prepared Mica runtime; see docs/standalone-dependencies.md\n')
    from decision_lab.mica_native import MicaClient
    client = MicaClient(abi)
    try:
        result = client.evaluate('Cancel my order.', {
            'q': dict(type='choice', instructions='What does the customer want?',
                      criteria={'cancel':'Cancel order', 'track':'Track order'})})
        if result != {'q':'cancel'}:
            raise RuntimeError('Mica semantic smoke failed')
        print(json.dumps(dict(result=result, abi=client.abi, codebook=client.codebook,
                              raw=client.last_raw)), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
