import json, subprocess, sys, os
from pathlib import Path

import yaml

from test_strategy_v2 import _make_dummy


def test_calibration_gate(tmp_path: Path):
    # generate dummy data and initial run
    csv_path = _make_dummy(tmp_path)
    params = yaml.safe_load(open('conf/params_champion.yml'))
    yaml.safe_dump(params, open(tmp_path / 'params.yml', 'w'))
    outdir = tmp_path / 'run0'
    outdir.mkdir()
    cmd = [
        sys.executable, 'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(tmp_path / 'params.yml'),
        '--outdir', str(outdir)
    ]
    env = os.environ.copy(); env.setdefault('PYTHONPATH', '.')
    subprocess.check_call(cmd, env=env)

    # calibrate offline
    caldir = tmp_path / 'calib'
    cmd = [
        sys.executable, 'tools/calibrate_offline.py',
        '--preds', str(outdir / 'preds_test.csv'),
        '--outdir', str(caldir)
    ]
    subprocess.check_call(cmd, env=env)
    ece = json.load(open(caldir / 'ece.json'))['ece']
    brier = json.load(open(caldir / 'brier.json'))['brier']
    assert 0.0 <= ece <= 1.0 and 0.0 <= brier <= 1.0

    # rerun with calibrator and varying thresholds
    counts = []
    for thr in (0.60, 0.80):
        params = yaml.safe_load(open('conf/params_champion.yml'))
        params.setdefault('gating', {}).setdefault('calibration', {}).setdefault('p_thr', {})['trend'] = thr
        params['gating']['calibration']['p_thr']['range'] = thr
        yaml.safe_dump(params, open(tmp_path / f'params_{int(thr*100)}.yml', 'w'))
        out = tmp_path / f'run_{int(thr*100)}'; out.mkdir()
        cmd = [
            sys.executable, 'backtest/runner_patched.py',
            '--data-root', str(csv_path.parent),
            '--csv-glob', csv_path.name,
            '--params', str(tmp_path / f'params_{int(thr*100)}.yml'),
            '--outdir', str(out),
            '--calibrator', str(caldir / 'calibrator.json')
        ]
        subprocess.check_call(cmd, env=env)
        summary = json.load(open(out / 'summary.json'))
        counts.append(summary['n_trades'])
        assert 'ece' in summary and 'brier' in summary
    assert counts[0] >= counts[1]
