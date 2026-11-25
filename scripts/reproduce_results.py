#!/usr/bin/env python3
"""CPU-only result reproduction from the public, reduced predictions. No source text or API required."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from corrosion_ie.eval.legacy_wrapper import _load_legacy

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def score(item,out):
    ev=_load_legacy();ev.RECORD_MATCH_THRESHOLD=.38
    split=item['split']; pred=ROOT/item['prediction']; gold=ev.load_gold(ROOT/f'input/{split}/gold/{split}.jsonl'); expected={'dev37':(37,670),'test50':(50,2396)}[split]
    assert (len(gold),sum(map(len,gold.values())))==expected
    predmap,dups=ev.load_predictions(pred); assert set(predmap)==set(gold) and all(n==1 for n in dups.values())
    lengths={r['paper_id']:int(r['document_chars']) for r in rows(ROOT/'input/paper_manifest.csv') if r['split']==split}
    ref=ROOT/item['reference_summary']; baseline=rows(ref)[0]
    result=ev.evaluate_strategy(baseline['strategy'],pred,gold,doc_length_map=lengths)
    dest=out/item['stage']/item['condition'];ev.write_evaluation_outputs(result,dest)
    checks=[]
    for n in ('summary_metrics.csv','per_paper_metrics.csv','per_field_metrics.csv','per_paper_field_metrics.csv'):
        a,b=rows(ref.parent/n),rows(dest/n)
        assert len(a)==len(b),(item['condition'],n,len(a),len(b))
        for x,y in zip(a,b):
            for k,v in x.items():
                if k not in y:raise AssertionError((n,k))
                w=y[k]
                try:
                    p,q=float(v),float(w);eq=math.isclose(p,q,rel_tol=1e-10,abs_tol=1e-12) or (math.isnan(p) and math.isnan(q))
                except (ValueError,TypeError):eq=v==w
                if not eq:raise AssertionError((item['condition'],n,k,v,w))
        checks.append(n)
    return {'stage':item['stage'],'condition':item['condition'],'paper_count':len(gold),'tables_verified':len(checks),'record_f1':result['summary']['record_f1'],'field_micro_f1':result['summary']['field_micro_f1'],'null_accuracy':result['summary']['null_accuracy']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reproduced');p.add_argument('--stage',choices=['Qwen','E','GLM']);p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=1);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    m=json.loads((ROOT/'release_manifest.json').read_text());assert sha(ROOT/'vendor/legacy_evaluate.py')==m['evaluator_sha256']; items=m['conditions'];assert len(items)==103 and sum(x['paper_count'] for x in items)==4097
    if 'paper_manifest_sha256' in m: assert sha(ROOT/'input/paper_manifest.csv')==m['paper_manifest_sha256']
    for x in items:
        assert sha(ROOT/x['prediction'])==x['release_sha256']; assert (ROOT/x['reference_summary']).is_file()
        ids=[json.loads(l)['paper_id'] for l in (ROOT/x['prediction']).read_text().splitlines() if l.strip()]
        g=[json.loads(l)['paper_id'] for l in (ROOT/f"input/{x['split']}/gold/{x['split']}.jsonl").read_text().splitlines() if l.strip()]
        assert len(ids)==len(set(ids))==x['paper_count'] and set(ids)==set(g)
    print('103 conditions, 4097 paper-level predictions: integrity and paper coverage verified.',flush=True)
    if a.check_only:return
    selected=[x for x in items if not a.stage or x['stage']==a.stage]; selected=selected[:a.limit] if a.limit else selected
    a.output.mkdir(exist_ok=True,parents=True); out=a.output.resolve(); reports=[]
    if a.workers==1:
        for x in selected:
            r=score(x,out);reports.append(r);print(r['stage'],r['condition'],'PASS',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for r in pool.map(score,selected,[out]*len(selected)):
                reports.append(r);print(r['stage'],r['condition'],'PASS',flush=True)
    (out/'verification.json').write_text(json.dumps(reports,indent=2)+'\n');print(f'Passed {len(reports)}/{len(selected)} conditions.',flush=True)
if __name__=='__main__':main()
