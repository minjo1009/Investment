import pandas as pd

t = pd.read_csv("out/trades.csv")
g = pd.read_json("out/gating_debug.json")

br = t.groupby(["session","regime"])["pnl_bps"].agg(["count","mean","sum"]).reset_index()
br.to_csv("out/report_session_regime.csv", index=False)

hm = g[["passed_ev","OFI_conf"]].copy()
hm["passed_ev"] = (hm["passed_ev"].astype(str).isin(["1","True","true"])).astype(int)
hm["weak_ofi"] = (hm["OFI_conf"].astype(float) < 0.30).astype(int)
hm.mean().to_frame("mean").to_csv("out/gate_rates.csv")
print("Reports written to out/*.csv")
