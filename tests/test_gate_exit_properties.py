import json
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


def _run(tmp: Path) -> Path:
  csv_path = _make_dummy(tmp)
  outdir = tmp / 'out'
  outdir.mkdir()
  params = yaml.safe_load(open('conf/params_champion.yml'))
  yaml.safe_dump(params, open(tmp / 'params.yml', 'w'))
  cmd = [
    sys.executable,
    'backtest/runner_patched.py',
    '--data-root', str(csv_path.parent),
    '--csv-glob', csv_path.name,
    '--params', str(tmp / 'params.yml'),
    '--outdir', str(outdir)
  ]
  env = {'PYTHONPATH': '.'}
  subprocess.check_call(cmd, env=env)
  return outdir


@pytest.fixture
def sim_trades(tmp_path: Path):
  outdir = _run(tmp_path)
  return pd.read_csv(outdir / 'trades.csv')


def test_min_hold_never_violated(sim_trades):
  if 'bars_held' in sim_trades.columns:
    assert (sim_trades['bars_held'] >= 8).all()
