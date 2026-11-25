# Evaluation and statistical analysis

All commands are run from the repository root using the dependencies in `requirements-reproduction.txt`. Saved predictions are used throughout.

## Threshold calibration

The matching threshold was selected on dev37 using 309 expert matching annotations. The primary matching procedure filters edges below the threshold before Hungarian assignment. To repeat the calibration:

```bash
python scripts/calibrate_record_threshold.py fit --project-root . --annotations work/record_threshold/record_threshold_calibration_server_ready.xlsx --output-dir reproduced/calibration --bootstrap 10000
```

The selected threshold is 0.38. The release preserves the original evaluator and records both source and release prediction checksums. The calibration workbook references the distributed predictions; the annotation decisions are unchanged.

## Full statistical workflow

```bash
python scripts/reevaluate_public_038.py --project-root . --output-root reproduced/qwen
python figures/reevaluate_e0_e5.py --project-root . --source-run-root figures/E_predictions --output-dir reproduced/e0_e5
python figures/reevaluate_glm.py --project-root . --reevaluation-root reproduced/qwen --predictions figures/GLM_predictions/predictions.jsonl --output reproduced/glm
python figures/prepare_plot_data.py --reevaluation-root reproduced/qwen --glm-evaluation reproduced/glm --e0-e5-evaluation reproduced/e0_e5 --output reproduced/plot_data
python figures/plot_figures_t038.py --data-dir reproduced/plot_data --output-dir reproduced/figures_from_analysis
```

The Qwen workflow uses 10,000 bootstrap and permutation iterations (seed 20260710). Holm correction is applied separately to 81 strategy comparisons and 63 model comparisons. The Qwen–GLM paired bootstrap uses 20,000 iterations (seed 20260717). The evaluation entry point for this distribution is `reevaluate_public_038.py`; `reevaluate_saved_038.py` records the original experimental workflow.

## Supplementary analyses

```bash
python analysis/analyze_threshold_sensitivity.py --root . --out reproduced/supplementary
python analysis/evaluate_exact_record.py --root . --out reproduced/supplementary
python analysis/analyze_model_strategy_interaction.py --root . --out reproduced/supplementary
python analysis/analyze_format_validity.py --root . --out reproduced/supplementary
```

| Analysis | Method |
| --- | --- |
| S18(a) | Matching-threshold sensitivity |
| S18(b) | Strict 13-field and seven-core-field exact-record F1 with explicit null agreement and maximum-cardinality matching |
| S19 | Paper-clustered CR1 OLS, chi-square Wald tests and Holm correction |
| S20 | Format-validity diagnostics and micro-aggregated metrics on valid samples |

The supplementary scripts implement the reported analysis definitions. Reference outputs are in `analysis/reference/`. The archived numerical figure data in `figures/data/` can also be plotted directly.
