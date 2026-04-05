"""Shared result loading utilities for benchmark analysis scripts."""

import json
import sys
from pathlib import Path


def load_benchmark_results(results_dir):
    """Load all per-seed JSON result files from a benchmark results directory.

    Walks ``results_dir/<pmid_dir>/*.json``, parsing each file into a dict.
    Skips files that fail to parse and prints a warning to stderr.

    Returns a list of result dicts.
    """
    results_dir = Path(results_dir)
    records = []
    for pmid_dir in sorted(results_dir.iterdir()):
        if not pmid_dir.is_dir() or not pmid_dir.name[:1].isdigit():
            continue
        for jf in sorted(pmid_dir.glob("*.json")):
            try:
                d = json.loads(jf.read_text())
                records.append(d)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"WARNING: failed to load {jf}: {e}", file=sys.stderr)
    return records
