import pandas as pd
t = pd.read_csv("out/trades.csv")
g = pd.read_json("out/gating_debug.json", lines=False)

br = t.groupby(["session","regime"])["pnl_bps"].agg(["count","mean","sum"]).reset_index()
br.to_csv("out/report_session_regime.csv", index=False)

cols = ["passed_ev","OFI_dir_ok"]
hm = g[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).mean().to_frame("pass_rate")
hm["fail_rate"] = 1.0 - hm["pass_rate"]
hm.to_csv("out/gate_rates.csv")
