from dataclasses import dataclass

@dataclass
class ConvictionParams:
    m:int=5
    k:int=3
    thr_entry:float=0.88
    thr_exit:float=0.82
    alpha_cost:float=1.0  # conservative EV gate 완화(0.8~1.0 범)

class ConvictionState:
    def __init__(self):
        self.last_side=0   # -1,0,1

def passes_conviction(p_hat_calib: float,
                      p_thr: float,
                      momentum_aligned_count: int,
                      params: ConvictionParams,
                      ev: float,
                      frictions: float,
                      side: int,
                      state: ConvictionState) -> bool:
    if p_hat_calib < p_thr: return False
    if momentum_aligned_count < params.k: return False
    if ev < params.alpha_cost * frictions: return False
    if state.last_side != 0 and state.last_side != side:
        if p_hat_calib < params.thr_entry: return False
    return True

def expected_value_bps(p_win: float, tp_bps: float, sl_bps: float, frictions_bps: float) -> float:
    ev = p_win * tp_bps - (1.0 - p_win) * sl_bps
    required = ConvictionParams.alpha_cost * frictions_bps
    return ev - required

def ev_gate_by_regime(regime: str, p_win: float, frictions_bps: float) -> float:
    # 레짐별 TP/SL 차등 EV 요구치
    if regime == "trend":
        tp, sl = 38.0, 22.0
    else:
        tp, sl = 30.0, 20.0
    return expected_value_bps(p_win, tp, sl, frictions_bps)
