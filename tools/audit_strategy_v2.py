import json
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
import yaml


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    repo = Path('.').resolve()
    spec = load(repo / 'specs' / 'strategy_v2_spec.yml')
    params = load(repo / 'conf' / 'params_champion.yml')
    checks = []

    # spec keys
    try:
        sig = spec['components']['signal']
        checks.append(('signal.macd.fast', 'fast' in sig['macd']))
        checks.append(('signal.macd.slow', 'slow' in sig['macd']))
        checks.append(('signal.signal', sig.get('signal') == 9))
        cal = spec['components']['gating']['calibration']['p_thr']
        checks.append(('p_thr.trend', 'trend' in cal))
        checks.append(('p_thr.range', 'range' in cal))
        conv = spec['components']['gating']['conviction']
        per = conv['persistence']
        checks.append(('persistence.m', 'm' in per))
        checks.append(('persistence.k', 'k' in per))
        checks.append(('ev_gate.mode', 'mode' in conv['ev_gate']))
        exits = spec['components']['exits']
        checks.append(('exits.mode', exits.get('mode') is not None))
        checks.append(('costs.fee_bps_per_side', 'fee_bps_per_side' in spec['components']['costs']))
    except Exception:
        checks.append(('spec_structure', False))

    # params keys
    exit_params = params.get('exit', {})
    for k in ['min_hold', 'max_hold', 'tp_bps', 'sl_bps', 'breakeven_bps']:
        checks.append((f'params.exit.{k}', k in exit_params))

    # runner code checks
    runner_text = (repo / 'backtest' / 'runner_patched.py').read_text()
    runner_checks = {
        'timestamp_patch': '_patched_read_csv' in runner_text,
        'p_ev_req': 'p_ev_req' in runner_text,
        'tp_sl_dynamic': 'tp_bps_i' in runner_text and 'sl_bps_i' in runner_text,
        'p_smoothing': 'ema_span' in runner_text,
    }
    for k, v in runner_checks.items():
        checks.append((f'runner.{k}', v))

    lines = ["|Check|Result|", "|---|---|"]
    for name, ok in checks:
        lines.append(f"|{name}|{'PASS' if ok else 'FAIL'}|")

    outdir = repo / 'out'
    outdir.mkdir(exist_ok=True)
    (outdir / 'VERIFICATION.md').write_text("\n".join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

