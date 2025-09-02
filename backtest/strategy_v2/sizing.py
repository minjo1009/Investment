def conviction_scaled_size(pop: float, cap: float = 1.0, base: float = 0.3) -> float:
  # 이소토닉 보정확률 기반 선형 스케일
  return min(cap, max(0.0, base + (pop - 0.5) * 1.4))

def dd_throttle(equity_curve, floor: float = 0.9) -> float:
  dd = (equity_curve / equity_curve.cummax()).iloc[-1]
  return 0.5 if dd < floor else 1.0

__all__ = ["conviction_scaled_size", "dd_throttle"]
