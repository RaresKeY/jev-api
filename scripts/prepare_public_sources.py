#!/usr/bin/env python3
"""Explicit acquisition, static scan and data-only export, in separate stages.

Standalone dependencies: Git/network for acquisition; original checkouts and local acquisition metadata for later stages.
See docs/standalone-dependencies.md.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.public_sources import acquire, scan, export_sources, resolve_parquet, export_mmlu_rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['acquire', 'scan', 'export', 'parquet', 'rows'])
    stage = parser.parse_args().stage
    value = {'acquire': acquire, 'scan': scan, 'export': export_sources, 'parquet': resolve_parquet,
             'rows': export_mmlu_rows}[stage]()
    print(json.dumps(value, indent=2))
