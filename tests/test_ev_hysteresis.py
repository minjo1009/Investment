
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
