from dataclasses import dataclass

@dataclass
class Frictions:
    fee_bps_per_side: float=5.0
    slippage_bps_per_side: float=2.0
    funding_bps_estimate: float=0.5

    def per_roundtrip(self) -> float:
        return (self.fee_bps_per_side+self.slippage_bps_per_side)*2.0 + self.funding_bps_estimate
