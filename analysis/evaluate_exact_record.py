"""S18b: threshold-independent maximum-cardinality exact-record matching."""
import hashlib,importlib.util,sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from _repro_common import args,conditions,paper_data,ratio
CORE=('inhibitor_name','substrate','corrosive_medium','medium_concentration','inhibitor_concentration','test_method','inhibition_efficiency')
EXPECTED='1bcbd9c106490a662bd380a8579e05ab676f56cf45b9f7dadcc773d5ac5ce4a9'
def matches(ev,p,g,f):
    a,b=p.get(f),g.get(f); na,nb=ev.is_empty_value(a,f),ev.is_empty_value(b,f); return (na and nb) if (na or nb) else ev.values_match(a,b,f,relaxed=True)
def cardinality(mask):
    if not mask.size: return 0
    i,j=linear_sum_assignment(-mask.astype(np.int8)); return int(mask[i,j].sum())
def main():
    a=args(__doc__); path=a.root/'vendor'/'legacy_evaluate.py'; assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED,'Evaluator differs from frozen version'; spec=importlib.util.spec_from_file_location('frozen_evaluator',path); ev=importlib.util.module_from_spec(spec); sys.modules[spec.name]=ev; spec.loader.exec_module(ev)
    gold=ev.load_gold(a.root/'input'/'test50'/'gold'/'test50.jsonl'); assert len(gold)==50 and sum(map(len,gold.values()))==2396; extra=[f for f in ev.EVAL_FIELDS if f not in CORE]; assert len(extra)==6; rows=[]; details=[]
    for m,q,run,e in conditions(a.root):
        p,_=ev.load_predictions(run/'predictions.jsonl'); assert set(p)==set(gold); primary=paper_data(e).set_index('paper_id'); strict=core=pn=gn=0
        for pid,g in sorted(gold.items()):
            pred=p[pid]; assert len(pred)==primary.loc[pid,'pred_record_count'] and len(g)==primary.loc[pid,'gold_record_count']; cm=np.zeros((len(pred),len(g)),dtype=bool); sm=cm.copy()
            for i,pr in enumerate(pred):
                for j,gr in enumerate(g):
                    if all(matches(ev,pr,gr,f) for f in CORE): cm[i,j]=True; sm[i,j]=all(matches(ev,pr,gr,f) for f in extra)
            st,co=cardinality(sm),cardinality(cm); strict+=st; core+=co; pn+=len(pred); gn+=len(g); details.append(dict(model=m,strategy=q,paper_id=pid,pred_count=len(pred),gold_count=len(g),strict_tp=st,core_tp=co))
        rows.append(dict(model=m,strategy=q,pred_count=pn,gold_count=gn,record_f1_at_038=ratio(2*primary.record_tp.sum(),pn+gn),strict_13_tp=strict,strict_13_f1=ratio(2*strict,pn+gn),core_7_tp=core,core_7_f1=ratio(2*core,pn+gn))); print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(a.out/'S18b_exact_record.csv',index=False); pd.DataFrame(details).to_csv(a.out/'S18b_per_paper.csv',index=False)
if __name__=='__main__': main()
