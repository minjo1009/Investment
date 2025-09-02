from dataclasses import dataclass

@dataclass
class ConvictionParams:
    m:int=5
    k:int=3
    thr_entry:float=0.88
    thr_exit:float=0.82
    alpha_cost:float=2.0

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
