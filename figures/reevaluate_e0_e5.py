"""Re-evaluate 54 saved GLM runs on dev37, keeping the frozen evaluator unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

STRATEGY_MAP={'E0':'E0','E1':'E1','E3':'E2','E4':'E3','E5':'E4','E6':'E5'}
PRIMARY=['field_micro_f1','record_f1','null_accuracy']
LEGACY_THRESHOLD=.60  # Verified in the original strong-LLM config.py; overrides evaluator default .55.
AUDIT=PRIMARY+['record_tp','record_pred_total','record_gold_total','field_tp_total',
              'field_pred_total','field_gold_total','null_correct','null_total']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def interval(values):
    a=np.asarray(values,dtype=float)
    assert len(a)==9 and np.isfinite(a).all()
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,8)*sd/3)
    return dict(mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half,n_runs=9)

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--source-run-root',type=Path,required=True)
    p.add_argument('--frozen',type=Path,default=here/'data/selected_threshold_frozen.json')
    p.add_argument('--reference-provenance',type=Path,default=here/'data/provenance.json')
    p.add_argument('--output-dir',type=Path,default=here/'e0_e5_evaluation_t038')
    a=p.parse_args();root=a.project_root.resolve();src=a.source_run_root.resolve();out=a.output_dir.resolve()
    frozen=json.loads(a.frozen.read_text());reference=json.loads(a.reference_provenance.read_text())
    assert frozen['dataset']=='dev37' and frozen['selected_threshold']==.38
    assert frozen['primary_matching_order']=='threshold_before_hungarian'
    evpath=root/'vendor/legacy_evaluate.py';goldpath=root/'input/dev37/gold/dev37.jsonl'
    assert sha(evpath)==reference['evaluator_sha256'], 'Evaluator differs from Qwen/GLM test50'
    assert sha(goldpath)==frozen['gold_sha256'], 'Gold data changed'
    sys.path.insert(0,str(root))
    from corrosion_ie.eval.legacy_wrapper import _load_legacy
    ev=_load_legacy();gold=ev.load_gold(goldpath)
    assert len(gold)==37 and sum(map(len,gold.values()))==670
    lengths={str(z.paper_id):int(z.document_chars) for z in pd.read_csv(root/'input/dev37_document_features.csv').itertuples()};assert len(lengths)==37
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='running',dataset='dev37',record_threshold=.38,
        matching_order='threshold_before_hungarian',gold_sha256=sha(goldpath),evaluator_sha256=sha(evpath),
        frozen_sha256=sha(a.frozen),script_sha256=sha(__file__),strategy_mapping=STRATEGY_MAP,
        paper_count=37,gold_record_count=670,expected_runs=54,legacy_record_threshold=LEGACY_THRESHOLD,
        mean_and_ci='Arithmetic mean over 3 prompt variants x 3 repetitions; two-sided Student t interval, df=8, sample SD (ddof=1)',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'reevaluation_manifest.json',manifest)
    rows=[];fields=[];comparisons=[];sources=[]
    for v in (1,2,3):
        for repeat in (1,2,3):
            run=f'v{v}_r{repeat}';runroot=src/run
            oldfile=runroot/'evaluation/summary_metrics.csv'
            old=pd.read_csv(oldfile).set_index('strategy');assert set(old.index)==set(STRATEGY_MAP)
            for legacy,formal in STRATEGY_MAP.items():
                found=[f for f in (runroot/'results').glob('*/*_results.jsonl')
                    if f.parent.name.upper()==legacy and f.stem.upper()==legacy+'_RESULTS']
                assert len(found)==1,(run,legacy,found)
                pred=found[0];predmap,dups=ev.load_predictions(pred)
                assert set(predmap)==set(gold),(run,legacy,'paper IDs differ')
                assert all(n==1 for n in dups.values()),(run,legacy,'duplicate paper IDs')
                raw=[json.loads(line) for line in pred.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                assert {row.get('model') for row in raw}=={'glm-4-plus'}
                assert {str(row.get('strategy')).upper() for row in raw}=={legacy}
                ev.RECORD_MATCH_THRESHOLD=LEGACY_THRESHOLD
                old_result=ev.evaluate_strategy(legacy,pred,gold,doc_length_map=lengths)
                old_summary=old_result['summary']
                parity={m:float(old_summary[m])-float(old.loc[legacy,m]) for m in AUDIT}
                if max(map(abs,parity.values()))>1e-10:
                    raise ValueError(f'Cannot reproduce old {LEGACY_THRESHOLD:.2f} result {run}/{legacy}: {parity}')
                ev.RECORD_MATCH_THRESHOLD=.38
                result=ev.evaluate_strategy(formal,pred,gold,doc_length_map=lengths)
                dest=out/'runs'/run/formal;ev.write_evaluation_outputs(result,dest)
                identity=dict(run_id=run,prompt_variant=v,repeat=repeat,legacy_strategy=legacy,
                    manuscript_strategy=formal,model='glm-4-plus',dataset='dev37',record_threshold=.38)
                meta=dict(**identity,source_prediction=str(pred),prediction_sha256=sha(pred),
                    old_summary_sha256=sha(oldfile),legacy_parity=parity,legacy_record_threshold=LEGACY_THRESHOLD,
                    gold_sha256=sha(goldpath),evaluator_sha256=sha(evpath),matching_order='threshold_before_hungarian',
                    output_sha256={f.name:sha(f) for f in sorted(dest.glob('*.csv'))})
                write_json(dest/'evaluation_meta.json',meta)
                rows.append({**identity,**result['summary']})
                fields.extend({**identity,**fr} for fr in result['per_field'])
                for metric in PRIMARY:
                    comparisons.append({**identity,'metric':metric,'legacy_record_threshold':LEGACY_THRESHOLD,'value_legacy':old_summary[metric],
                        'value_t038':result['summary'][metric],
                        'difference_t038_minus_legacy':result['summary'][metric]-old_summary[metric]})
                sources.append(dict(path=str(pred.relative_to(src)),sha256=sha(pred),run_id=run,
                    legacy_strategy=legacy,manuscript_strategy=formal))
                print(f"[{len(rows):02d}/54] {run} {legacy}->{formal} parity=OK field={result['summary']['field_micro_f1']:.4f} record={result['summary']['record_f1']:.4f} null={result['summary']['null_accuracy']:.4f}",flush=True)
    runs=pd.DataFrame(rows);field=pd.DataFrame(fields);assert len(runs)==54 and len(field)==54*13
    runs.to_csv(out/'per_run_metrics.csv',index=False)
    field.to_csv(out/'per_run_field_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'comparison_legacy_t038.csv',index=False)
    summary=[]
    for strategy,g in runs.groupby('manuscript_strategy',sort=True):
        for metric in PRIMARY+['field_micro_precision','field_micro_recall','record_precision','record_recall']:
            summary.append({'strategy':strategy,'metric':metric,**interval(g[metric])})
    summary=pd.DataFrame(summary);summary.to_csv(out/'strategy_metrics.csv',index=False)
    fsummary=[]
    for (strategy,f),g in field.groupby(['manuscript_strategy','field'],sort=True):
        row={'strategy':strategy,'field':f}
        for metric in ['precision','recall','f1']:
            row.update({metric+'_'+k:v for k,v in interval(g[metric]).items()})
        fsummary.append(row)
    pd.DataFrame(fsummary).to_csv(out/'field_metrics.csv',index=False)
    diffs=[]
    for f,g in field.groupby('field',sort=True):
        paired=g.pivot(index='run_id',columns='manuscript_strategy',values='f1')
        assert len(paired)==9 and not paired[['E4','E0']].isna().any().any()
        diffs.append({'field':f,'comparison':'E4-E0',**interval(paired.E4-paired.E0)})
    pd.DataFrame(diffs).to_csv(out/'e4_minus_e0_fields.csv',index=False)
    pd.DataFrame([{'legacy_strategy':k,'manuscript_strategy':v} for k,v in STRATEGY_MAP.items()]).to_csv(out/'strategy_mapping.csv',index=False)
    manifest.update(status='complete',runs_evaluated=54,legacy_parity_passed=54,sources=sources,
        output_sha256={f.name:sha(f) for f in sorted(out.glob('*.csv'))},finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'reevaluation_manifest.json',manifest)
    print(summary[summary.metric.isin(PRIMARY)].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
