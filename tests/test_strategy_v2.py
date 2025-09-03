import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _make_dummy(tmp: Path) -> Path:
  tmp.mkdir(parents=True, exist_ok=True)
  n = 120
  ts = pd.date_range('2020-01-01', periods=n, freq='1min', tz='UTC')
  price = 100.0
  rows = []
  for i in range(n):
    open_ = price
    high = open_ + (0.1 if i < 60 else 0.5)
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
  csv_path = tmp / 'sample.csv'
  pd.DataFrame(rows).to_csv(csv_path, index=False)
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
  env = os.environ.copy()
  env.setdefault('PYTHONPATH', '.')
  subprocess.check_call(cmd, env=env)
  return outdir


def test_spec_keys():
  spec = yaml.safe_load(open('specs/strategy_v2_spec.yml'))
  comps = spec['components']
  assert 'signal' in comps and 'gating' in comps


def test_wiring_p_trend(tmp_path: Path):
  outdir = _run(tmp_path)
  preds = pd.read_csv(outdir / 'preds_test.csv')
  assert preds['p_trend'].between(0,1).all()


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


def test_ev_gate(tmp_path: Path):
  outdir = _run(tmp_path)
  dbg = json.load(open(outdir / 'gating_debug.json'))
  vals = [d.get('ev_bps') for d in dbg if 'ev_bps' in d]
  assert vals and all(isinstance(v, (int, float)) for v in vals)


def test_dynamic_atr_exits(tmp_path: Path):
  outdir = _run(tmp_path)
  dbg = json.load(open(outdir / 'gating_debug.json'))
  rec = next(d for d in dbg if d.get('tp_bps_i'))
  assert 'tp_bps_i' in rec and 'sl_bps_i' in rec


def test_artifacts_schema(tmp_path: Path):
  outdir = _run(tmp_path)
  preds = pd.read_csv(outdir / 'preds_test.csv')
  assert set(['p_trend','macd_hist']).issubset(preds.columns)
  summary = json.load(open(outdir / 'summary.json'))
  for k in ['n_trades','hit_rate','mcc','cum_pnl_bps']:
    assert k in summary


def test_accepts_datetime_column(tmp_path: Path):
  csv_path = _make_dummy(tmp_path)
  df = pd.read_csv(csv_path)
  df.rename(columns={'timestamp': 'datetime'}, inplace=True)
  df.to_csv(csv_path, index=False)
  outdir = tmp_path / 'out'
  outdir.mkdir()
  params = yaml.safe_load(open('conf/params_champion.yml'))
  yaml.safe_dump(params, open(tmp_path / 'params.yml', 'w'))
  cmd = [
    sys.executable,
    'backtest/runner_patched.py',
    '--data-root', str(csv_path.parent),
    '--csv-glob', csv_path.name,
    '--params', str(tmp_path / 'params.yml'),
    '--outdir', str(outdir)
  ]
  env = os.environ.copy()
  env.setdefault('PYTHONPATH', '.')
  subprocess.check_call(cmd, env=env)
  preds = pd.read_csv(outdir / 'preds_test.csv')
  assert len(preds) == 120
