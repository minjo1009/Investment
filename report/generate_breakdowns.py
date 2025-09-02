import pandas as pd

t = pd.read_csv("out/trades.csv")
g = pd.read_json("out/gating_debug.json", lines=True)
br = t.groupby(["session","regime"])["pnl_bps"].agg(["count","mean","sum"]).reset_index()
br.to_csv("out/report_session_regime.csv", index=False)
cols = ["passed_calib","passed_ev","passed_persist","ofi_ok","in_box"]
hm = g[cols].astype(int).mean().to_frame("fail_rate").assign(fail_rate=lambda x:1-x["fail_rate"])
hm.to_csv("out/gate_fail_heatmap.csv")
