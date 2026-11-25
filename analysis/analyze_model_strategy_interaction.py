"""S19: paper-clustered CR1 joint Wald chi-square, Holm correction."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from _repro_common import args,conditions,paper_data,EXP
def main():
    a=args(__doc__); frames=[]
    for m,q,_,e in conditions(a.root):
        d=paper_data(e).assign(model=m,strategy=q); d['field_f1']=2*d.tp/(d.pred_count+d.gold_count); d['record_f1']=d.F1; d['conditional_null_accuracy']=d.correct_null_count/d.gold_null_count_matched.replace(0,np.nan); frames.append(d)
    f=pd.read_csv(a.root/'work'/'reevaluation_t038'/EXP/'document_features.csv'); assert len(f)==50 and f.paper_id.is_unique
    for k,v in {'log_length_z':np.log(f.document_chars),'table_fraction_z':f.table_char_fraction}.items(): f[k]=(v-v.mean())/v.std(ddof=0)
    d=pd.concat(frames).merge(f[['paper_id','log_length_z','table_fraction_z']],on='paper_id',validate='many_to_one'); assert len(d)==450; rows=[]
    for y in ['field_f1','record_f1','conditional_null_accuracy']:
        sub=d.dropna(subset=[y]); fit=smf.ols(f'{y} ~ C(model)*C(strategy) + log_length_z + table_fraction_z',data=sub).fit(cov_type='cluster',cov_kwds={'groups':sub.paper_id,'use_correction':True}); terms=[i for i,n in enumerate(fit.params.index) if ':' in n]; assert len(terms)==4
        w=fit.wald_test(np.eye(len(fit.params))[terms],use_f=False,scalar=True); rows.append(dict(outcome=y,observations=len(sub),papers=sub.paper_id.nunique(),wald_chi2=float(w.statistic),df=4,raw_p=float(w.pvalue)))
    r=pd.DataFrame(rows); r['holm_p']=multipletests(r.raw_p,method='holm')[1]; r.to_csv(a.out/'S19_interaction.csv',index=False); print(r.to_string(index=False))
if __name__=='__main__': main()
