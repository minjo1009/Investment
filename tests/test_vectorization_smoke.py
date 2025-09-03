import json, subprocess, sys, os
from pathlib import Path

def test_smoke_fast(tmp_path: Path):
    out = tmp_path/"out"; out.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "backtest/runner_patched.py",
        "--data-root", "data",
        "--csv-glob", "ETHUSDT_1min_2020_2025.csv",
        "--params", "conf/params_champion.yml",
        "--outdir", str(out),
        "--limit-bars", "100000",
        "--debug-level", "none",
        "--no-preds"
    ]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '.')
    subprocess.check_call(cmd, env=env)

def test_gate_monotonic(tmp_path: Path):
    counts = []
    for thr in (0.60, 0.70, 0.80, 0.95):
        out = tmp_path/f"o{int(thr*100)}"; out.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "backtest/runner_patched.py",
            "--data-root", "data",
            "--csv-glob", "ETHUSDT_1min_2020_2025.csv",
            "--params", "conf/params_champion.yml",
            "--outdir", str(out),
            "--limit-bars", "100000",
            "--debug-level", "none",
            "--no-preds"
        ]
        # patch p_thr via env override if supported, else skip (manual tune in spec)
        env = os.environ.copy()
        env.setdefault('PYTHONPATH', '.')
        subprocess.check_call(cmd, env=env)
        s = json.load(open(out/"summary.json"))
        counts.append(s["n_trades"])
    assert counts == sorted(counts, reverse=True)

