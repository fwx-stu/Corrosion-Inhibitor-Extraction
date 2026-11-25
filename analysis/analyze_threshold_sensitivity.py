#!/usr/bin/env python3
"""S18(a): apply the seven specified thresholds to unchanged saved test50 predictions."""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from _repro_common import args,conditions
THRESHOLDS=(.40,.45,.50,.55,.60,.65,.70)
def score(job):
    root,m,q,p=job
    sys.path.insert(0,str(root));from corrosion_ie.eval.legacy_wrapper import _load_legacy
    ev=_load_legacy(); gold=ev.load_gold(root/'input/test50/gold/test50.jsonl'); pred,_=ev.load_predictions(p/'predictions.jsonl');assert set(pred)==set(gold)
    counts=np.zeros(len(THRESHOLDS),dtype=int);pn=sum(map(len,pred.values()));gn=sum(map(len,gold.values()))
    for pid,g in gold.items():
        pr=pred[pid]
        if not pr or not g:continue
        s=np.array([[ev.record_similarity(a,b) for b in g] for a in pr])
        for i,t in enumerate(THRESHOLDS):
            ri,ci=linear_sum_assignment(-np.where(s>=t,s,0));counts[i]+=int((s[ri,ci]>=t).sum())
    return dict(model=m,strategy=q,**{f'{t:.2f}':2*int(n)/(pn+gn) for t,n in zip(THRESHOLDS,counts)})
def main():
    a=args(__doc__);root=a.root.resolve();jobs=[(root,m,q,p) for m,q,p,e in conditions(root)]
    with ProcessPoolExecutor(max_workers=3) as pool:r=pd.DataFrame(pool.map(score,jobs))
    r.to_csv(a.out/'S18a_threshold_sensitivity.csv',index=False);print(r.to_string(index=False))
if __name__=='__main__':main()
