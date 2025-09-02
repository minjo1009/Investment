import json
from pathlib import Path

import pandas as pd
import yaml

from .util_dummy import run_runner


def test_spec_keys():
    spec = yaml.safe_load(open('specs/strategy_v2_spec.yml'))
    assert spec['components']['signal']['macd']['fast']
    assert spec['components']['signal']['macd']['slow']
    assert spec['components']['signal']['signal'] == 9
    pthr = spec['components']['gating']['calibration']['p_thr']
    assert 'trend' in pthr and 'range' in pthr
    conv = spec['components']['gating']['conviction']
    assert conv['persistence']['m']
    assert conv['persistence']['k']
    assert conv['ev_gate']['mode']
    exits = spec['components']['exits']
    for k in ['min_hold', 'max_hold', 'breakeven_bps']:
        assert k in exits


def test_gate_sweep_monotonic(tmp_path: Path):
    counts = []
    for thr in [0.60, 0.70, 0.80, 0.95]:
        spec_over = {'components': {'gating': {'calibration': {'p_thr': {'trend': thr, 'range': thr}}}}}
        outdir = run_runner(tmp_path / f't{int(thr*100)}', spec_overrides=spec_over)
        summary = json.load(open(outdir / 'summary.json'))
        counts.append(summary['n_trades'])
    assert counts == sorted(counts, reverse=True)


def test_ev_probability_gate(tmp_path: Path):
    outdir = run_runner(tmp_path / 'evprob')
    dbg = json.load(open(outdir / 'gating_debug.json'))
    vals = [d['p_ev_req'] for d in dbg if 'p_ev_req' in d]
    assert len(vals) > 0 and all(0 < v < 1 for v in vals)


def test_dynamic_atr_exits(tmp_path: Path):
    outdir = run_runner(tmp_path / 'atr')
    trades = pd.read_csv(outdir / 'trades.csv')
    assert {'tp_bps', 'sl_bps'}.issubset(trades.columns)
    dbg = json.load(open(outdir / 'gating_debug.json'))
    assert all('tp_bps_i' in d and 'sl_bps_i' in d for d in dbg)


def test_artifacts_schema(tmp_path: Path):
    outdir = run_runner(tmp_path / 'art')
    preds = pd.read_csv(outdir / 'preds_test.csv')
    for col in ['p_trend', 'ofi', 'macd_hist']:
        assert col in preds.columns
    summary = json.load(open(outdir / 'summary.json'))
    for k in ['n_trades', 'hit_rate', 'mcc', 'cum_pnl_bps']:
        assert k in summary

