import json, argparse, numpy as np, pandas as pd, os

ap = argparse.ArgumentParser()
ap.add_argument('--preds', required=True)
ap.add_argument('--out', default='conf/calibrator_bins.json')
args = ap.parse_args()

df = pd.read_csv(args.preds)
if 'p_trend' in df.columns and 'macd_hist' in df.columns:
    p = df['p_trend'].astype(float).clip(0,1).values
else:
    raise SystemExit('need preds with p_trend and macd_hist')

bins = np.linspace(0,1,21)
centers = (bins[:-1]+bins[1:])/2.0
means = centers
out = [{'x':float(x),'y':float(y)} for x,y in zip(centers,means)]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out,'w'), indent=2)
