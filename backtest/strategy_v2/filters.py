from dataclasses import dataclass

@dataclass
class RollingBalanceParams:
    win:int=60  # minutes
    tr_pctl_max:int=25

class RollingBalance:
    def __init__(self, params: RollingBalanceParams):
        self.params=params
        self.buffer=[]  # store true range values

    def update(self, high: float, low: float, prev_close: float):
        tr=max(high-low, abs(high-prev_close), abs(low-prev_close))
        self.buffer.append(tr)
        if len(self.buffer)>self.params.win: self.buffer.pop(0)

    def in_balance_box(self) -> bool:
        if not self.buffer: return False
        s=sorted(self.buffer)
        idx=max(0, min(len(s)-1, (self.params.tr_pctl_max*len(s))//100))
        thr=s[idx]
        return self.buffer[-1] <= thr

def approx_ofi(o: float, h: float, l: float, c: float, v: float) -> float:
    rng=(h-l) if h>l else 1e-12
    loc=(c-l)/rng
    return (loc-0.5)*2.0 * v
