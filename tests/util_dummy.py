import json
import subprocess
import sys
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
        high = open_ + (0.5 if i > 60 else 0.1)
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


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d.setdefault(k, {})
            _deep_update(d[k], v)
        else:
            d[k] = v


def run_runner(outdir: Path, params_overrides=None, spec_overrides=None) -> Path:
    tmp = outdir.parent
    csv_path = _make_dummy(tmp)
    spec = yaml.safe_load(open('specs/strategy_v2_spec.yml'))
    params = yaml.safe_load(open('conf/params_champion.yml'))
    if spec_overrides:
        _deep_update(spec, spec_overrides)
    if params_overrides:
        _deep_update(params, params_overrides)
    spec_path = tmp / 'spec.yml'
    params_path = tmp / 'params.yml'
    yaml.safe_dump(spec, open(spec_path, 'w'))
    yaml.safe_dump(params, open(params_path, 'w'))
    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(params_path),
        '--outdir', str(outdir)
    ]
    cal_path = Path('conf/calibrator_isotonic.json')
    if cal_path.exists():
        cmd.extend(['--calibrator', str(cal_path)])
    subprocess.check_call(cmd)
    return outdir

