import json, subprocess, sys, os
from pathlib import Path

import pandas as pd
import yaml

from test_strategy_v2 import _make_dummy


def _run_runner(tmp: Path, args: list[str]) -> Path:
    outdir = tmp / f"out_{len(list(tmp.iterdir()))}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        *args,
        '--outdir', str(outdir)
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '.')
    subprocess.check_call(cmd, env=env)
    return outdir


def test_runs_fast_smoke(tmp_path: Path):
    outdir = _run_runner(tmp_path, [
        '--data', 'ETHUSDT_1min_2020_2025.zip',
        '--params', 'conf/params_champion.yml',
        '--limit-bars', '100000',
        '--debug-level', 'none',
        '--no-preds'
    ])
    assert (outdir / 'summary.json').exists()


def test_gate_monotonic(tmp_path: Path):
    csv_path = _make_dummy(tmp_path)
    thrs = [0.60, 0.70, 0.80, 0.95]
    counts = []
    for thr in thrs:
        params = yaml.safe_load(open('conf/params_champion.yml'))
        params['entry']['p_thr']['trend'] = thr
        params['entry']['p_thr']['range'] = thr
        yaml.safe_dump(params, open(tmp_path / 'params.yml', 'w'))
        outdir = _run_runner(tmp_path, [
            '--data-root', str(csv_path.parent),
            '--csv-glob', csv_path.name,
            '--params', str(tmp_path / 'params.yml')
        ])
        summary = json.load(open(outdir / 'summary.json'))
        counts.append(summary['n_trades'])
    assert counts == sorted(counts, reverse=True)


def test_artifacts_schema(tmp_path: Path):
    csv_path = _make_dummy(tmp_path)
    outdir = _run_runner(tmp_path, [
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', 'conf/params_champion.yml'
    ])
    trades = pd.read_csv(outdir / 'trades.csv')
    assert set(['entry_idx','exit_idx','side','entry_px','exit_px','pnl_bps']).issubset(trades.columns)
    summary = json.load(open(outdir / 'summary.json'))
    for k in ['n_trades','hit_rate','mcc','cum_pnl_bps']:
        assert k in summary
    gd = pd.read_csv(outdir / 'gating_debug.csv')
    req = {'i','side','pop','p_ev_req','ev_bps','tp_bps_i','sl_bps_i','regime','be_armed','decision'}
    assert req.issubset(gd.columns)
