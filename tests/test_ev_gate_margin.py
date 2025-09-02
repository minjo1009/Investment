import json, subprocess, sys, os
from pathlib import Path

import pandas as pd
import yaml

from test_strategy_v2 import _make_dummy


def _run(tmp: Path, delta: float) -> Path:
    csv_path = _make_dummy(tmp)
    outdir = tmp / 'out'
    outdir.mkdir()
    params = yaml.safe_load(open('conf/params_champion.yml'))
    params.setdefault('gating', {}).setdefault('conviction', {}).setdefault('ev_gate', {})['delta_p_min'] = delta
    yaml.safe_dump(params, open(tmp / 'params.yml', 'w'))
    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(tmp / 'params.yml'),
        '--outdir', str(outdir)
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '.')
    subprocess.check_call(cmd, env=env)
    return outdir


def test_ev_gate_margin(tmp_path: Path):
    deltas = [0.0, 0.1]
    counts = []
    for d in deltas:
        outdir = _run(tmp_path / f'd{int(d*100)}', d)
        summary = json.load(open(outdir / 'summary.json'))
        counts.append(summary['n_trades'])
    assert counts[0] >= counts[1]
