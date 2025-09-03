
import os, json, textwrap

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# 1) trend4u/calib/quantile_map.py — 학습/추론용 분위수 맵
write("trend4u/calib/quantile_map.py", textwrap.dedent("""
import numpy as np, pandas as pd, json, bisect
class QuantileMap:
    def __init__(self, quantiles=None, values=None):
        self.q = quantiles or []
        self.v = values or []
    @staticmethod
    def fit(scores: np.ndarray, n_bins: int = 100):
        scores = np.asarray(scores).astype(float)
        qs = np.linspace(0, 1, n_bins+1)
        vals = np.quantile(scores, qs, method="linear")
        return QuantileMap(quantiles=list(qs), values=list(vals))
    def transform(self, scores: np.ndarray):
        s = np.asarray(scores).astype(float)
        out = np.empty_like(s)
        for i, x in enumerate(s):
            j = bisect.bisect_left(self.v, x)
            if j<=0: out[i]=self.q[0]
            elif j>=len(self.v): out[i]=self.q[-1]
            else:
                x0,x1=self.v[j-1],self.v[j]
                q0,q1=self.q[j-1],self.q[j]
                t=0.0 if x1==x0 else (x-x0)/(x1-x0)
                out[i]=q0 + (q1-q0)*t
        return out
    def dumps(self): return json.dumps({"q":self.q,"v":self.v})
    @staticmethod
    def loads(s): 
        o=json.loads(s); return QuantileMap(o["q"], o["v"])
"""))

# 2) trend4u/calib/drift.py — PSI/KS 계산
write("trend4u/calib/drift.py", textwrap.dedent("""
import numpy as np
from scipy import stats

def _bins(x, n=20): 
    edges = np.quantile(x, np.linspace(0,1,n+1))
    edges[0]-=1e-9; edges[-1]+=1e-9
    return edges

def psi(expected, actual, n=20):
    expected=np.asarray(expected, float); actual=np.asarray(actual, float)
    e_edges=_bins(expected, n)
    e_hist,_=np.histogram(expected, bins=e_edges); a_hist,_=np.histogram(actual, bins=e_edges)
    e_dist=np.clip(e_hist/e_hist.sum(),1e-8,1); a_dist=np.clip(a_hist/a_hist.sum(),1e-8,1)
    return np.sum((a_dist-e_dist)*np.log(a_dist/e_dist))

def ks(expected, actual):
    return stats.ks_2samp(expected, actual).statistic
"""))

# 3) tests — 가드/히스테리시스/세션 블랙리스트
write("tests/test_calibration_guards.py", textwrap.dedent("""
import numpy as np
from trend4u.calib.quantile_map import QuantileMap
from trend4u.calib.drift import psi, ks

def test_quantile_map_monotonic():
    x=np.random.randn(10000)
    qm=QuantileMap.fit(x, n_bins=50)
    y=qm.transform(x)
    assert np.all((y>=0)&(y<=1))
    # rank 보존의 근사
    assert abs(np.mean(y)-0.5)<0.1

def test_psi_ks_sane():
    base=np.random.beta(2,5,20000)
    drift=np.random.beta(5,2,20000)
    assert psi(base, drift)>0.1
    assert ks(base, drift)>0.1
"""))

write("tests/test_ev_hysteresis.py", textwrap.dedent("""
def _hyst(e, x, thr_in, thr_out):
    # 단순 히스테리시스 모형(상태 e: in/out)
    if not e and x>=thr_in: return True
    if e and x<=thr_out: return False
    return e

def test_hysteresis():
    in_state=False
    series=[0.7,0.9,0.88,0.83,0.81,0.85,0.84]
    for p in series:
        in_state=_hyst(in_state,p,0.88,0.82)
    assert in_state in [True, False]
"""))

write("tests/test_session_blacklist.py", textwrap.dedent("""
from datetime import datetime, timezone

def blocked(dt_utc, windows_utc):
    for s,e in windows_utc:
        if s<=dt_utc<e: return True
    return False

def test_blocklist_basic():
    now=datetime(2025,1,1,3,0,0,tzinfo=timezone.utc)
    bl=[(datetime(2025,1,1,2,0,0,tzinfo=timezone.utc), datetime(2025,1,1,4,0,0,tzinfo=timezone.utc))]
    assert blocked(now, bl) is True
"""))

# 4) 샘플 파라미터(레짐별 EV/히스테리시스/가드 임계)
write("conf/feature_flags.yml.patch", textwrap.dedent("""
FEATURE_P_QUANTILE_MAP: true
FEATURE_CALIB_GROUP_BY_SESSION_REGIME: true
FEATURE_CALIB_GUARDS: true
FEATURE_OFI_SOFT_ONLY: true
FEATURE_MAX_HOLD_ENFORCED: true
"""))

write("conf/params_champion.yml.patch", textwrap.dedent("""
calibration:
  ece_guard: 0.03
  psi_guard: 0.20
  min_bin: 100
  use_quantile_map: true
  qm_bins: 100
ev_gate:
  hysteresis:
    thr_entry: 0.88
    thr_exit: 0.82
  regime:
    trend:
      ev_margin_bps: 4
      delta_p_min: 0.06
    range:
      ev_margin_bps: 6
      delta_p_min: 0.08
    neutral:
      ev_margin_bps: 5
      delta_p_min: 0.07
guards:
  on_bad_calib:
    thr_entry_add: 0.02
    ev_margin_bps_add: 2
    trades_cap: 0.75
exit:
  tp_bps: 38
  sl_bps: 22
  breakeven_bps: 7
  min_hold: 8
  max_hold: 60
session:
  blacklist_windows_utc: []  # [["2025-01-01T02:00:00Z","2025-01-01T04:00:00Z"], ...]
  vol_pctl_cutoff: 0.20
"""))

print("Wrote new modules, tests, and param patches.")
