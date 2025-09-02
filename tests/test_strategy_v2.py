import json, subprocess, sys
from pathlib import Path

import pandas as pd
import yaml


def _make_dummy(tmp: Path) -> Path:
    n = 120
    ts = pd.date_range('2020-01-01', periods=n, freq='1min', tz='UTC')
    price = 100.0
    rows = []
    for i in range(n):
        open_ = price
        if i < 60:
            high = open_ + 0.1
        else:
            high = open_ + 0.5
        low = open_
        close = high
        rows.append({
            'timestamp': ts[i],
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1.0,
            'p_hat': 0.9
        })
        price = close
    df = pd.DataFrame(rows)
    csv_path = tmp / 'sample.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def _run(tmp: Path, thr: float | None = None) -> Path:
    csv_path = _make_dummy(tmp)
    outdir = tmp / 'out'
    outdir.mkdir()
    params = yaml.safe_load(open('conf/params_champion.yml'))
    if thr is not None:
        params.setdefault('entry', {}).setdefault('p_thr', {})
        params['entry']['p_thr']['trend'] = thr
        params['entry']['p_thr']['range'] = thr
    yaml.safe_dump(params, open(tmp / 'params.yml', 'w'))
    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(tmp / 'params.yml'),
        '--outdir', str(outdir)
    ]
    subprocess.check_call(cmd)
    return outdir


def test_wiring_p_trend(tmp_path: Path):
    outdir = _run(tmp_path)
    preds = pd.read_csv(outdir / 'preds_test.csv')
    assert 'p_trend' in preds.columns


def test_summary_metrics(tmp_path: Path):
    outdir = _run(tmp_path)
    summary = json.load(open(outdir / 'summary.json'))
    for k in ['hit_rate', 'mcc', 'cum_pnl_bps']:
        assert k in summary


def test_gate_sweep_monotonic(tmp_path: Path):
    thrs = [0.60, 0.70, 0.80, 0.95]
    counts = []
    for thr in thrs:
        outdir = _run(tmp_path / f't{int(thr*100)}', thr)
        summary = json.load(open(outdir / 'summary.json'))
        counts.append(summary['n_trades'])
    assert counts == sorted(counts, reverse=True)
