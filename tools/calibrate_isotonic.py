import pandas as pd, json, numpy as np, argparse
from sklearn.isotonic import IsotonicRegression

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--group-keys", default=None)  # "session,regime"
ap.add_argument("--min-bin", type=int, default=100)
ap.add_argument("--out", required=True)
args = ap.parse_args()

preds = pd.read_csv(args.preds)
labels = pd.read_csv(args.labels)
# 기대 컬럼: preds: ['open_time','p_raw','session','regime'] or ['p_trend'] 폴백
p = preds["p_raw"] if "p_raw" in preds else preds[preds.columns[1]]
labcol = "label" if "label" in preds else ("label" if "label" in labels else None)
if labcol is None:
  # trades.csv의 pnl_bps로 0/1 레이블 근사 (양수=1)
  y = (labels["pnl_bps"] > 0).astype(int).reindex(preds.index, fill_value=0)
else:
  y = preds[labcol]

def fit_iso(pp, yy):
  iso = IsotonicRegression(out_of_bounds="clip").fit(pp.to_numpy(), yy.to_numpy())
  xs = np.linspace(0,1,101)
  ys = iso.predict(xs)
  return {"xs": xs.tolist(), "ys": ys.tolist()}

def apply_grp(df):
  return fit_iso(df["p"], df["y"])

maps = {"_default": fit_iso(p.clip(0,1), y)}
skipped = []
if args.group_keys:
  keys = [k.strip() for k in args.group_keys.split(",")]
  if all(k in preds.columns for k in keys):
    gb = preds.assign(p=p.clip(0,1), y=y).groupby(keys)
    for k, g in gb:
      # k can be tuple or single value
      k = k if isinstance(k, tuple) else (k,)
      key_vals = {kk: vv for kk, vv in zip(keys, k)}
      grp_key = "|".join(f"{kk}={key_vals[kk]}" for kk in keys)
      if len(g) >= args.min_bin:
        maps[grp_key] = apply_grp(g)
      else:
        skipped.append(grp_key)

out_obj = {"maps": maps, "skipped": skipped}
json.dump(out_obj, open(args.out, "w"))
if skipped:
  print(f"Skipped groups due to min_bin: {skipped}")
print(f"Saved calibrator to {args.out} with {len(maps)} maps.")
