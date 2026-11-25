"""Shared inputs for S18–S20 reproduction."""
import argparse
from pathlib import Path
import pandas as pd
EXP='corrosion_q0q1q2_test50_balanced_opt_final_v6'
def args(desc):
    p=argparse.ArgumentParser(description=desc); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); return a
def conditions(root):
    for m in (3,7,14):
        for q,s in enumerate(('q0_safe_full','q1_budgeted_rag','q2_table_aware_rag')):
            suffix=Path(EXP)/f'qwen2_5_{m}b'/s/'balanced'/'r1'
            yield f'{m}B',f'Q{q}',root/'output'/'exp_v2'/suffix,root/'work'/'reevaluation_t038'/suffix/'evaluation'
def paper_data(e):
    r=pd.read_csv(e/'per_paper_metrics.csv'); f=pd.read_csv(e/'per_paper_field_metrics.csv'); assert len(r)==50 and r.paper_id.is_unique and len(f)==650 and not f.duplicated(['paper_id','field']).any(); assert set(r.paper_id)==set(f.paper_id) and (f.groupby('paper_id').size()==13).all()
    return r.merge(f.groupby('paper_id',as_index=False)[['tp','pred_count','gold_count']].sum(),on='paper_id',validate='one_to_one')
def ratio(n,d): return n/d if d else float('nan')
def aggregate(d):
    return dict(field_f1=ratio(2*d.tp.sum(),d.pred_count.sum()+d.gold_count.sum()),record_f1=ratio(2*d.record_tp.sum(),d.pred_record_count.sum()+d.gold_record_count.sum()),null_accuracy=ratio(d.correct_null_count.sum(),d.gold_null_count_matched.sum()))
