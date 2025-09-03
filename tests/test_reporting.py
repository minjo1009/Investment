import json, os, subprocess, sys
from pathlib import Path
import pandas as pd

def _run(tmp: Path):
    out = tmp / "out"; out.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "backtest/runner_patched.py",
        "--data-root", "data",
        "--csv-glob", "ETHUSDT_1min_2020_2025.csv",
        "--params", "conf/params_champion.yml",
        "--outdir", str(out),
        "--limit-bars", "50000",
        "--debug-level", "entries",
        "--no-preds"
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '.')
    subprocess.check_call(cmd, env=env)
    return out

def test_reports_and_artifacts(tmp_path: Path):
    out = _run(tmp_path)
    assert (out/"summary.json").exists()
    assert (out/"trades.csv").exists()
    assert (out/"gating_debug.csv").exists()
    assert (out/"calibration_report.json").exists()
    assert (out/"gate_waterfall.json").exists()
    # schema spot check
    df = pd.read_csv(out/"trades.csv")
    for col in ["entry_idx","exit_idx","side","entry_px","exit_px","pnl_bps"]:
        assert col in df.columns

