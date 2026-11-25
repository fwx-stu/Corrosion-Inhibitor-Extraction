"""Prepare traceable plot tables from threshold-0.38 outputs, without inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

DEV = 'corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6'
TEST = 'corrosion_q0q1q2_test50_balanced_opt_final_v6'
CONTROL = 'corrosion_reviewer_controls_test50_v1'
METRICS = ('field_micro_f1', 'record_f1', 'null_accuracy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(directory):
    paper = pd.read_csv(directory / 'per_paper_metrics.csv').set_index('paper_id').sort_index()
    field = pd.read_csv(directory / 'per_paper_field_metrics.csv').groupby('paper_id')[['tp','pred_count','gold_count']].sum()
    assert paper.index.is_unique and len(paper) == 50
    field = field.reindex(paper.index)
    a = np.column_stack([
        field.tp, field.pred_count, field.gold_count,
        paper.record_tp, paper.pred_record_count, paper.gold_record_count,
        paper.correct_null_count, paper.gold_null_count_matched,
    ]).astype(float)
    assert np.isfinite(a).all()
    return paper.index, a


def metrics(a):
    # Pooled counts are recomputed for every bootstrap sample, not averaged paper F1.
    den = np.stack([a[...,1]+a[...,2], a[...,4]+a[...,5], a[...,7]],axis=-1)
    num = np.stack([2*a[...,0], 2*a[...,3], a[...,6]],axis=-1)
    return np.divide(num, den, out=np.full_like(num,np.nan), where=den>0)


def include_e0_e5(output_dir, evaluation_dir):
    """Append verified Stage-I tables without changing the Qwen/GLM test50 tables."""
    output_dir=Path(output_dir);evaluation_dir=Path(evaluation_dir)
    manifest_path=evaluation_dir/'reevaluation_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    provenance_path=output_dir/'provenance.json'
    provenance=json.loads(provenance_path.read_text())
    assert manifest['status']=='complete' and manifest['runs_evaluated']==54
    assert manifest['legacy_parity_passed']==54 and manifest['legacy_record_threshold']==.60
    assert manifest['record_threshold']==provenance['record_threshold']==.38
    assert manifest['evaluator_sha256']==provenance['evaluator_sha256']
    assert manifest['matching_order']==provenance['matching_order']
    for name,digest in manifest['output_sha256'].items():
        assert sha(evaluation_dir/name)==digest, f'E0-E5 output changed: {name}'
    mapping={
        'strategy_metrics.csv':'e0_e5_strategy_metrics.csv',
        'field_metrics.csv':'e0_e5_field_metrics.csv',
        'e4_minus_e0_fields.csv':'e4_minus_e0_fields.csv',
        'per_run_metrics.csv':'e0_e5_per_run_metrics.csv',
        'comparison_legacy_t038.csv':'e0_e5_comparison_legacy_t038.csv',
    }
    for source,target in mapping.items():
        shutil.copyfile(evaluation_dir/source,output_dir/target)
        provenance['sources'][str((evaluation_dir/source).resolve())]=sha(evaluation_dir/source)
    provenance['sources'][str(manifest_path.resolve())]=sha(manifest_path)
    provenance.update(verified_e0_e5_runs=54,e0_e5_legacy_threshold=.60,
        e0_e5_strategy_mapping=manifest['strategy_mapping'],e0_e5_ci=manifest['mean_and_ci'],
        ready_figures=['5','6','7','8','S1','S2','S3','S4'],pending_figures=[],pending_reason=None)
    provenance['plot_data_sha256']={f.name:sha(f) for f in sorted(output_dir.glob('*.csv'))}
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reevaluation-root',type=Path)
    p.add_argument('--glm-evaluation',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data')
    p.add_argument('--e0-e5-evaluation',type=Path,default=Path(__file__).resolve().parent/'e0_e5_evaluation_t038')
    p.add_argument('--e0-e5-only',action='store_true',help='Merge the verified Stage-I tables into the existing plot data only')
    a=p.parse_args()
    if a.e0_e5_only:
        include_e0_e5(a.output,a.e0_e5_evaluation)
        print('Merged 54 verified E0-E5 runs into the existing plot data.')
        return
    if a.reevaluation_root is None or a.glm_evaluation is None:
        p.error('--reevaluation-root and --glm-evaluation are required unless --e0-e5-only is used')
    r=a.reevaluation_root.resolve(); out=a.output
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((r/'reevaluation_manifest.json').read_text())
    assert manifest['status']=='complete' and not manifest['smoke_test']
    assert manifest['runs_evaluated']==48 and manifest['record_threshold']==0.38
    assert manifest['matching_order']=='threshold_before_hungarian'
    sources={}
    def record(path):
        path=Path(path); sources[str(path.resolve())]=sha(path)
    record(r/'reevaluation_manifest.json')
    record(r/'calibration/selected_threshold_frozen.json')
    assert sha(r/'calibration/selected_threshold_frozen.json')==manifest['frozen_sha256']
    paths=list(r.glob('*/qwen*/**/evaluation/evaluation_meta.json'))
    assert len(paths)==48
    for path in paths:
        meta=json.loads(path.read_text());record(path)
        assert meta['record_threshold']==0.38 and meta['matching_order']==manifest['matching_order']
        assert meta['evaluator_sha256']==manifest['evaluator_sha256']
        pred=path.parent.parent/'predictions.jsonl'
        assert sha(pred)==meta['prediction_sha256'];record(pred)
        for name,digest in meta['output_sha256'].items():
            f=path.parent/name;assert sha(f)==digest;record(f)
    gm=json.loads((a.glm_evaluation/'evaluation_meta.json').read_text())
    assert gm['record_threshold']==0.38 and gm['matching_order']==manifest['matching_order']
    assert gm['evaluator_sha256']==manifest['evaluator_sha256'] and gm['paper_count']==50
    record(a.glm_evaluation/'evaluation_meta.json')
    for name,digest in gm['output_sha256'].items():
        f=a.glm_evaluation/name;assert sha(f)==digest;record(f)
    summary=pd.read_csv(r/'all_formal_summary_t038.csv');record(r/'all_formal_summary_t038.csv')
    assert len(summary)==48 and (summary.record_threshold==0.38).all()
    dev=summary[summary.experiment==DEV].copy();test=summary[summary.experiment==TEST].copy()
    assert len(dev)==27 and len(test)==9
    dev.to_csv(out/'dev37_prompt_matrix.csv',index=False)
    glm=pd.read_csv(a.glm_evaluation/'summary_metrics.csv')
    for k,v in dict(model_key='glm_4_plus',evidence_strategy='q0_safe_full',dataset='test50',prompt_variant='balanced',seed=1,record_threshold=0.38).items():
        glm[k]=v
    pd.concat([test,glm],ignore_index=True).to_csv(out/'test50_model_metrics.csv',index=False)
    fields=[]
    for strategy in ('q0_safe_full','q1_budgeted_rag','q2_table_aware_rag'):
        path=r/TEST/'qwen2_5_14b'/strategy/'balanced/r1/evaluation/per_field_metrics.csv'
        f=pd.read_csv(path);f['model_key']='qwen2_5_14b';f['evidence_strategy']=strategy;fields.append(f)
    f=pd.read_csv(a.glm_evaluation/'per_field_metrics.csv');f['model_key']='glm_4_plus';f['evidence_strategy']='q0_safe_full';fields.append(f)
    pd.concat(fields,ignore_index=True).to_csv(out/'test50_field_metrics.csv',index=False)
    for source,name in [
        (r/TEST/'stratified_strategy_differences.csv','structure_differences.csv'),
        (r/TEST/'document_features.csv','document_features.csv'),
        (r/TEST/'condition_metrics_with_bootstrap_ci.csv','test50_bootstrap_ci.csv'),
        (r/CONTROL/'paired_comparisons_manuscript_holm.csv','test50_paired_comparisons.csv'),
    ]:
        record(source);shutil.copyfile(source,out/name)
    # Paired paper bootstrap for Qwen2.5-14B Q1 minus GLM-4-Plus Q0.
    # Share each paper resample between the two models and across the three metrics.
    qpath=r/TEST/'qwen2_5_14b/q1_budgeted_rag/balanced/r1/evaluation'
    qi,q=counts(qpath);gi,g=counts(a.glm_evaluation)
    assert qi.equals(gi)
    qpoint=metrics(q.sum(0));gpoint=metrics(g.sum(0))
    qt=test[(test.model_key=='qwen2_5_14b')&(test.evidence_strategy=='q1_budgeted_rag')].iloc[0]
    for i,m in enumerate(METRICS):
        assert abs(qpoint[i]-qt[m])<1e-12 and abs(gpoint[i]-glm.iloc[0][m])<1e-12
    rng=np.random.default_rng(20260717)
    idx=rng.integers(0,50,size=(20000,50))
    delta=metrics(q[idx].sum(1))-metrics(g[idx].sum(1))
    assert np.isfinite(delta).all()
    ci=np.quantile(delta,[.025,.975],axis=0)
    paired=pd.DataFrame([dict(metric=m,value_qwen14b_q1=qpoint[i],value_glm_q0=gpoint[i],
        difference_high_minus_low=qpoint[i]-gpoint[i],ci_low=ci[0,i],ci_high=ci[1,i],
        bootstrap_iterations=20000,bootstrap_seed=20260717,paper_count=50,record_threshold=0.38)
        for i,m in enumerate(METRICS)])
    paired.to_csv(out/'glm_paired_comparison.csv',index=False)
    provenance=dict(record_threshold=0.38,matching_order=manifest['matching_order'],
        verified_qwen_runs=48,glm_test_papers=50,evaluator_sha256=manifest['evaluator_sha256'],
        qwen_bootstrap=dict(iterations=10000,seed=20260710),glm_bootstrap=dict(iterations=20000,seed=20260717),
        ready_figures=['6','7','8','S2','S3','S4'],pending_figures=['5','S1'],
        pending_reason='E0-E5 original predictions are not available; no E0-E5 values were substituted.',
        sources=sources,plot_data_sha256={p.name:sha(p) for p in sorted(out.glob('*.csv'))})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if (a.e0_e5_evaluation/'reevaluation_manifest.json').is_file():
        include_e0_e5(out,a.e0_e5_evaluation)
    print(paired[['metric','difference_high_minus_low','ci_low','ci_high']].to_string(index=False))
    print(f'Prepared plot tables: {out.resolve()}')


if __name__=='__main__':
    main()
