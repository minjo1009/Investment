import os, json, argparse, glob, math
import pandas as pd, numpy as np

def _find_col(cols, candidates):
    low=[c.lower() for c in cols]
    for cand in candidates:
        if cand in low: return cols[low.index(cand)]
    return None

def _read_one_csv(data_root, pattern):
    matches=glob.glob(os.path.join(data_root,pattern),recursive=True)
    if not matches: raise FileNotFoundError(f"No CSV matched: {pattern}")
    df=pd.read_csv(matches[0])
    dcol=_find_col(df.columns,["open_time","timestamp","time","datetime","date"])
    ccol=_find_col(df.columns,["close","close_price","c"])
    if not dcol or not ccol: raise ValueError("data csv needs datetime+close")
    df[dcol]=pd.to_datetime(df[dcol])
    return df.rename(columns={dcol:"open_time", ccol:"close"})[["open_time","close"]]

def _detect_event_col(cols): return _find_col(cols,["event"])

def _pair_from_trades(tr):
    evcol=_detect_event_col(tr.columns)
    if evcol is None: return None, None
    t=tr.copy()
    t["open_time"]=pd.to_datetime(t["open_time"],errors="coerce")
    e=t[evcol].astype(str).str.upper()
    t["is_entry"]=e.str.contains("ENTRY"); t["is_exit"]=e.str.contains("EXIT")
    t=t.sort_values("open_time",kind="stable").reset_index(drop=True)
    trade_id=[]; q=[]; next_id=0
    for ent,ex in zip(t["is_entry"],t["is_exit"]):
        if ent: next_id+=1; q.append(next_id); trade_id.append(next_id)
        elif ex and q: trade_id.append(q.pop(0))
        else: trade_id.append(np.nan)
    t["trade_id"]=trade_id
    ent=t[t["is_entry"]][["trade_id","open_time"]].rename(columns={"open_time":"entry_time"})
    ex =t[t["is_exit" ]][["trade_id","open_time"]].rename(columns={"open_time":"exit_time"})
    pairs=ent.merge(ex,on="trade_id",how="inner")
    return pairs,evcol

def _infer_side(tr, evcol):
    for c in tr.columns:
        cl=str(c).lower()
        if cl in ("side","direction","dir"):
            s=tr[c].astype(str).str.lower()
            arr=np.where(s.str.contains("short|sell|-1"),-1,1)
            e=tr[evcol].astype(str).str.upper().str.contains("ENTRY",na=False)
            return arr[e] if len(arr)==len(tr) else np.ones(e.sum(),dtype=int)
    return None

def enrich(outdir, data_root, csv_glob, thr=0.83, hold=9, fee_bps=5.0, slip_bps=2.0, start_capital=1000.0, position_fraction=1.0):
    sp=os.path.join(outdir,"summary.json"); tp=os.path.join(outdir,"trades.csv")
    os.makedirs(outdir,exist_ok=True)
    summ={}
    if os.path.exists(sp):
        try: summ=json.load(open(sp,"r",encoding="utf-8"))
        except: summ={}
    if os.path.exists(tp):
        tr=pd.read_csv(tp)
        if "open_time" in [c.lower() for c in tr.columns]:
            data=_read_one_csv(data_root,csv_glob)
            pairs,evcol=_pair_from_trades(tr)
            if pairs is not None and len(pairs):
                ent=pairs[["trade_id","entry_time"]].merge(
                    data.rename(columns={"open_time":"entry_time","close":"entry_price"}),on="entry_time",how="left")
                ex =pairs[["trade_id","exit_time"]].merge(
                    data.rename(columns={"open_time":"exit_time","close":"exit_price"}),on="exit_time",how="left")
                px=ent.merge(ex,on="trade_id",how="inner")
                side=_infer_side(tr,evcol); side=list(side) if side is not None else [1]*len(px)
                if len(side)<len(px): side+= [1]*(len(px)-len(side))
                px["side"]=side[:len(px)]
                px["gross_ret"]=(px["exit_price"]/px["entry_price"]-1.0)*px["side"]
                cost=2.0*(fee_bps+slip_bps)/10000.0
                px["net_ret"]=px["gross_ret"]-cost
                eq=start_capital; eq_curve=[]
                for r in px["net_ret"].fillna(0.0):
                    eq=eq*(1.0+position_fraction*r); eq_curve.append(eq)
                px["equity"]=eq_curve

                g=pd.to_numeric(px["gross_ret"],errors="coerce").fillna(0.0)
                n=pd.to_numeric(px["net_ret"],errors="coerce").fillna(0.0)
                gp,gn=float(g[g>0].sum()),float(g[g<0].sum())
                np_,nn=float(n[n>0].sum()),float(n[n<0].sum())
                summ.update({
                    "gross_exits": int(len(g)),
                    "gross_win_rate": float((g>0).sum())/max(1,len(g)),
                    "gross_profit_factor": (gp/abs(gn)) if gn!=0 else None,
                    "gross_cum_return": float(g.sum()),
                    "net_exits": int(len(n)),
                    "net_win_rate": float((n>0).sum())/max(1,len(n)),
                    "net_profit_factor": (np_/abs(nn)) if nn!=0 else None,
                    "net_cum_return": float(n.sum()),
                    "net_equity_end": float(px["equity"].iloc[-1]) if len(px) else start_capital,
                })
                t0=pd.to_datetime(pairs["entry_time"]).min(); t1=pd.to_datetime(pairs["exit_time"]).max()
                years=max((t1-t0).days/365.2422,1e-6)
                try: summ["net_cagr"]=float((summ["net_equity_end"]/start_capital)**(1.0/years)-1.0)
                except: summ["net_cagr"]=None
            else:
                evcol=_detect_event_col(tr.columns) or "event"
                summ["gross_exits"]=summ["net_exits"]=int(tr[evcol].astype(str).str.upper().str.contains("EXIT",na=False).sum())
                summ.setdefault("reason","no entry rows; could not pair trades")
        else:
            summ.setdefault("reason","trades.csv has no open_time")

    # MCC (옵션)
    ptp=os.path.join(outdir,"preds_test.csv")
    if os.path.exists(ptp):
        try:
            pt=pd.read_csv(ptp)
            prob=_find_col(pt.columns,["p","p_gate","gatep","prob","score","p_trend","p_range"])
            dcol=_find_col(pt.columns,["open_time","timestamp","time","datetime","date"])
            if prob and dcol:
                pt[dcol]=pd.to_datetime(pt[dcol])
                data=_read_one_csv(data_root,csv_glob).sort_values("open_time").reset_index(drop=True)
                data["fwd"]=data["close"].shift(-int(hold))/data["close"]-1.0
                df=pt[[dcol,prob]].rename(columns={dcol:"open_time"}).merge(data[["open_time","fwd"]],on="open_time",how="left")
                y_true=(df["fwd"]>0).astype(int); y_pred=(df[prob]>=float(thr)).astype(int)
                TP=int(((y_pred==1)&(y_true==1)).sum()); TN=int(((y_pred==0)&(y_true==0)).sum())
                FP=int(((y_pred==1)&(y_true==0)).sum()); FN=int(((y_pred==0)&(y_true==1)).sum())
                denom=float(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
                summ["mcc"]=float((TP*TN-FP*FN)/denom) if denom else 0.0
                summ["cmatrix"]={"TP":TP,"TN":TN,"FP":FP,"FN":FN}
        except: pass

    json.dump(summ, open(sp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[metrics_enforcer]", json.dumps(summ, ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--csv-glob",required=True)
    ap.add_argument("--thr",type=float,default=0.83)
    ap.add_argument("--hold",type=int,default=9)
    ap.add_argument("--fee-bps",type=float,default=5.0)
    ap.add_argument("--slip-bps",type=float,default=2.0)
    ap.add_argument("--start-capital",type=float,default=1000.0)
    ap.add_argument("--position-fraction",type=float,default=1.0)
    a=ap.parse_args(); enrich(a.outdir,a.data_root,a.csv_glob,a.thr,a.hold,a.fee_bps,a.slip_bps,a.start_capital,a.position_fraction)
