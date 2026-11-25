"""S20: exclude parse errors or output truncation; micro-aggregate counts."""
import json
import pandas as pd
from _repro_common import args,conditions,paper_data,aggregate
def main():
    a=args(__doc__); rows=[]; flags=[]
    for m,q,run,e in conditions(a.root):
        d=paper_data(e); rec=[]
        for line in (run/'predictions.jsonl').read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            o=json.loads(line); g=(o.get('metadata') or {}).get('generation') or {}; assert g,(run,o['paper_id']); err=bool(o.get('parse_error')); trunc=bool(g.get('reached_max_new_tokens')) or g.get('finish_reason')=='length'; rec.append(dict(paper_id=o['paper_id'],parse_error=err,truncated=trunc,valid=not(err or trunc)))
        f=pd.DataFrame(rec); assert len(f)==50 and f.paper_id.is_unique and set(f.paper_id)==set(d.paper_id)
        meta=json.loads((run/'run_meta.json').read_text()); assert f.parse_error.sum()==meta['parse_errors'] and f.truncated.sum()==meta['output_truncations']
        d=d.merge(f,on='paper_id',validate='one_to_one'); v=d[d.valid]; rows.append(dict(model=m,strategy=q,n_all=len(d),n_valid=len(v),n_parse_error=int(f.parse_error.sum()),n_truncated=int(f.truncated.sum()),n_truncated_without_parse_error=int((f.truncated & ~f.parse_error).sum()),**{f'{k}_all':x for k,x in aggregate(d).items()},**{f'{k}_valid':x for k,x in aggregate(v).items()})); flags.append(f.assign(model=m,strategy=q))
    r=pd.DataFrame(rows); r.to_csv(a.out/'S20_format_validity.csv',index=False); pd.concat(flags).to_csv(a.out/'S20_paper_flags.csv',index=False); print(r.to_string(index=False))
if __name__=='__main__': main()
