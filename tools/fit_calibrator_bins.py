import json, argparse, numpy as np, pandas as pd, os
ap=argparse.ArgumentParser(); ap.add_argument('--preds', required=True); ap.add_argument('--out', default='conf/calibrator_bins.json'); args=ap.parse_args()
df=pd.read_csv(args.preds)
# p와 다음바 또는 trade hit 라벨이 있으면 사용, 없으면 엔트리만의 realized 승/패를 요구(사전 생성)
if 'p_trend' in df.columns and 'macd_hist' in df.columns:
    p=df['p_trend'].astype(float).clip(0,1).values
else:
    raise SystemExit("need preds with p_trend")
# 레이블이 없으면 종료(향후 확장). 여기서는 p 분포만 bin 평균에 y≈p 가정(보수적)
bins=np.linspace(0,1,21); centers=((bins[:-1]+bins[1:])/2.0)
means=centers  # 초기 보정: 항등에 가깝게 시작
out=[{"x":float(x),"y":float(y)} for x,y in zip(centers,means)]
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out,'w'), indent=2)
