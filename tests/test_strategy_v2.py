# -*- coding: utf-8 -*-
"""
테스트 유틸: 더미 CSV 생성 → runner_patched.py 실행 → 산출물 검증
- 컨플릭트 마커 제거
- import 정리
- thr 인자가 주어지면 specs/strategy_v2_spec.yml의 p_thr(trend/range)을 직접 패치하여
  러너가 실제로 임계값 변경을 반영하도록 수정
"""
import json, subprocess, sys
from pathlib import Path
import yaml, pandas as pd


def _make_dummy(tmp: Path) -> Path:
    n = 120
    ts = pd.date_range('2020-01-01', periods=n, freq='1min', tz='UTC')
    price = 100.0
    rows = []
    for i in range(n):
        open_ = price
        # 앞 60분은 약한 상승, 이후는 더 강한 상승으로 차등
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
            'p_hat': 0.9  # 높은 확률 가정 (게이트 단조성 테스트용)
        })
        price = close
    df = pd.DataFrame(rows)
    csv_path = tmp / 'sample.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def _patch_spec_p_thr(thr: float) -> None:
    """
    runner가 참조하는 specs/strategy_v2_spec.yml 안의
    components.gating.calibration.p_thr.{trend,range}를 직접 패치한다.
    (테스트에서 params로 임계값을 넘겨도 러너는 spec을 보므로 여기서 spec을 수정)
    """
    spec_path = Path('specs') / 'strategy_v2_spec.yml'
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f) or {}
    comp = spec.setdefault('components', {})
    gating = comp.setdefault('gating', {})
    calib = gating.setdefault('calibration', {})
    p_thr = calib.setdefault('p_thr', {})
    p_thr['trend'] = float(thr)
    p_thr['range'] = float(thr)
    with open(spec_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False)


def _run(tmp: Path, thr: float | None = None) -> Path:
    csv_path = _make_dummy(tmp)
    outdir = tmp / 'out'
    outdir.mkdir(parents=True, exist_ok=True)

    # params는 exits/costs 등 러너에서 참조 — 원본 읽어 tmp에 복사 저장
    with open('conf/params_champion.yml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f) or {}
    with open(tmp / 'params.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)

    # 임계치 전달되면 spec(p_thr)을 직접 패치하여 러너가 바로 반영하도록 함
    if thr is not None:
        _patch_spec_p_thr(thr)

    cmd = [
        sys.executable,
        'backtest/runner_patched.py',
        '--data-root', str(csv_path.parent),
        '--csv-glob', csv_path.name,
        '--params', str(tmp / 'params.yml'),
        '--outdir', str(outdir)
    ]
    subprocess.check_call(cmd)
    return outdir


def test_wiring_p_trend(tmp_path: Path):
    outdir = _run(tmp_path)
    preds = pd.read_csv(outdir / 'preds_test.csv')
    assert 'p_trend' in preds.columns, "preds_test.csv에 p_trend 컬럼이 없습니다."


def test_summary_metrics(tmp_path: Path):
    outdir = _run(tmp_path)
    with open(outdir / 'summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    for k in ['hit_rate', 'mcc', 'cum_pnl_bps']:
        assert k in summary, f"summary.json에 키가 없습니다: {k}"


def test_gate_sweep_monotonic(tmp_path: Path):
    # 임계값을 올리면 n_trades는 단조 감소해야 한다.
    thrs = [0.60, 0.70, 0.80, 0.95]
    counts = []
    for thr in thrs:
        outdir = _run(tmp_path / f't{int(thr*100)}', thr)
        with open(outdir / 'summary.json', 'r', encoding='utf-8') as f:
            summary = json.load(f)
        counts.append(summary.get('n_trades', 0))
    assert counts == sorted(counts, reverse=True), f"단조성 위배: {counts}"
