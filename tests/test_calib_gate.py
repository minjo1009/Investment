import json, subprocess, sys, os
from pathlib import Path
import yaml

from test_strategy_v2 import _make_dummy


def _run(csv_path: Path, params_path: Path, outdir: Path, calibrator: Path | None = None):
    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(params_path),
        '--outdir', str(outdir)
    ]
    if calibrator is not None:
        cmd += ['--calibrator', str(calibrator)]
    env = os.environ.copy(); env.setdefault('PYTHONPATH','.')
    subprocess.check_call(cmd, env=env)


def test_calib_gate(tmp_path: Path):
    csv_path = _make_dummy(tmp_path)
    out0 = tmp_path / 'base'; out0.mkdir()
    params0 = yaml.safe_load(open('conf/params_champion.yml'))
    yaml.safe_dump(params0, open(tmp_path/'params0.yml','w'))
    _run(csv_path, tmp_path/'params0.yml', out0)

    caldir = tmp_path / 'cal'; caldir.mkdir()
    subprocess.check_call([sys.executable, 'tools/calibrate_offline.py', '--preds', str(out0/'preds_test.csv'), '--outdir', str(caldir)])
    ece = json.load(open(caldir/'ece.json'))['value']
    brier = json.load(open(caldir/'brier.json'))['value']
    assert 0 <= ece <= 0.25
    assert 0 <= brier <= 0.25

    counts = []
    for thr in [0.55, 0.65, 0.75]:
        params = yaml.safe_load(open('conf/params_champion.yml'))
        params.setdefault('gating', {}).setdefault('calibration', {}).setdefault('p_thr', {})['trend'] = thr
        params['gating']['calibration']['p_thr']['range'] = thr
        ppath = tmp_path / f'p{int(thr*100)}.yml'
        yaml.safe_dump(params, open(ppath,'w'))
        outdir = tmp_path / f'run{int(thr*100)}'; outdir.mkdir()
        _run(csv_path, ppath, outdir, caldir/'calibrator.json')
        summary = json.load(open(outdir/'summary.json'))
        counts.append(summary['n_trades'])
    assert counts[0] >= counts[1] >= counts[2]
