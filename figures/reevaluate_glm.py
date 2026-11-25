"""Reuse saved GLM predictions and the frozen evaluator; no inference calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--reevaluation-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous-evaluation', type=Path)
    a = p.parse_args()
    root = a.project_root.resolve()
    manifest = json.loads((a.reevaluation_root / 'reevaluation_manifest.json').read_text())
    frozen_path = a.reevaluation_root / 'calibration/selected_threshold_frozen.json'
    frozen = json.loads(frozen_path.read_text())
    assert manifest['status'] == 'complete' and not manifest['smoke_test']
    assert sha(frozen_path) == manifest['frozen_sha256']
    assert frozen['dataset'] == 'dev37'
    assert frozen['selected_threshold'] == manifest['record_threshold'] == 0.38
    assert manifest['matching_order'] == 'threshold_before_hungarian'
    ev_path = root / 'vendor/legacy_evaluate.py'
    assert sha(ev_path) == manifest['evaluator_sha256'], 'Evaluator differs from Qwen re-evaluation'
    sys.path.insert(0, str(root))
    from corrosion_ie.eval.legacy_wrapper import _load_legacy
    ev = _load_legacy()
    # This changes only the imported module in this process; the project is untouched.
    gold_path = root / 'input/test50/gold/test50.jsonl'
    gold = ev.load_gold(gold_path)
    pred, duplicates = ev.load_predictions(a.predictions)
    assert len(gold) == 50 and set(pred) == set(gold), 'GLM must cover exactly test50'
    assert all(n == 1 for n in duplicates.values()), 'Duplicate paper IDs require inspection'
    lengths = {str(z.paper_id):int(z.document_chars) for z in pd.read_csv(root/'input/test50_document_features.csv').itertuples()}
    parity = None
    if a.previous_evaluation:
        old = pd.read_csv(a.previous_evaluation / 'summary_metrics.csv').iloc[0]
        ev.RECORD_MATCH_THRESHOLD = 0.55
        result_old = ev.evaluate_strategy('glm_4_plus__q0_safe_full__balanced__r1', a.predictions, gold, doc_length_map=lengths)
        parity = {}
        for m in ('field_micro_f1', 'record_f1', 'null_accuracy'):
            difference = float(result_old['summary'][m]) - float(old[m])
            assert abs(difference) < 1e-12, f'Old GLM metric not reproduced: {m}'
            parity[m] = difference
    ev.RECORD_MATCH_THRESHOLD = frozen['selected_threshold']
    result = ev.evaluate_strategy('glm_4_plus__q0_safe_full__balanced__r1', a.predictions, gold, doc_length_map=lengths)
    ev.write_evaluation_outputs(result, a.output)
    meta = dict(
        model_key='glm_4_plus', evidence_strategy='q0_safe_full', dataset='test50',
        prompt_variant='balanced', seed=1, record_threshold=0.38,
        matching_order=manifest['matching_order'], evaluator_sha256=sha(ev_path),
        gold_sha256=sha(gold_path), prediction_sha256=sha(a.predictions),
        frozen_sha256=sha(frozen_path), source_prediction=str(a.predictions.resolve()),
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(), paper_count=50,
        previous_055_parity=parity,
        output_sha256={f.name:sha(f) for f in sorted(a.output.glob('*.csv'))},
    )
    (a.output / 'evaluation_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({m:result['summary'][m] for m in ('field_micro_f1','record_f1','null_accuracy')},indent=2))


if __name__ == '__main__':
    main()
