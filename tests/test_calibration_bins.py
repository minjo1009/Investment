import json, subprocess, sys
from pathlib import Path

import pandas as pd

from test_strategy_v2 import _run


def test_calibration_bins(tmp_path: Path):
    outdir = _run(tmp_path)
    preds = outdir / 'preds_test.csv'
    cal_out = tmp_path / 'cal.json'
    cmd = [
        sys.executable,
        'tools/fit_calibrator_bins.py',
        '--preds', str(preds),
        '--out', str(cal_out)
    ]
    subprocess.check_call(cmd)
    data = json.load(open(cal_out))
    xs = [d['x'] for d in data]
    ys = [d['y'] for d in data]
    assert len(xs) == len(ys) and all(0.0 <= x <= 1.0 for x in xs)
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert xs == sorted(xs)
