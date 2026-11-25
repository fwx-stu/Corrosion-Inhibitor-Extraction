# Structured Corrosion Inhibitor Literature Extraction

Research software and data for **Structured Corrosion Inhibitor Literature Extraction: The Roles of Evidence Source, Model Capacity, and Document Structure**.

This study evaluates the effects of evidence source, model capacity and document structure on the extraction of 13 corrosion-inhibitor fields from scientific literature.

## Dataset and experiments

| Experiment | Papers per condition | Conditions |
| --- | ---: | ---: |
| GLM-4-Plus evidence strategies E0–E5 (dev37) | 37 | 54 |
| Qwen2.5 development experiments (dev37) | 37 | 27 |
| Qwen2.5 main experiments (test50) | 50 | 9 |
| Matched-budget and component controls (test50) | 50 | 12 |
| GLM-4-Plus reference (test50) | 50 | 1 |
| **Total** | | **103** |

The expert reference dataset contains 87 papers and 3,066 normalized, deduplicated records (670 in dev37 and 2,396 in test50). The release includes 4,097 paper-level structured predictions, the reference annotations, evaluation code, statistical analyses and numerical figure data.

## Reproduce the evaluation

Use Python 3.10+ and run the following commands from the repository root. On Windows, enable Python UTF-8 mode (`$env:PYTHONUTF8 = "1"` in PowerShell).

```bash
python -m pip install -r requirements-reproduction.txt
python scripts/reproduce_results.py --check-only
python scripts/reproduce_results.py --stage GLM --output reproduced/quick
python scripts/reproduce_results.py --workers 4 --output reproduced/full
```

Evaluation uses the frozen record-matching threshold of 0.38, with threshold filtering before Hungarian assignment. The script computes and compares the summary, per-paper, per-field and per-paper-field metrics against the supplied reference tables. The `--stage` option selects `Qwen`, `E` or `GLM`. These commands use saved predictions and require neither model inference nor source articles.

## Reproduce the analyses and figures

```bash
python analysis/evaluate_exact_record.py --root . --out reproduced/supplementary
python analysis/analyze_model_strategy_interaction.py --root . --out reproduced/supplementary
python analysis/analyze_format_validity.py --root . --out reproduced/supplementary
python analysis/analyze_threshold_sensitivity.py --root . --out reproduced/supplementary
python figures/plot_figures_t038.py --data-dir figures/data --output-dir reproduced/figures
```

See [statistical analysis](docs/supplemental_analysis.md) for threshold calibration, the full statistical workflow and figure-data generation. For extraction methods, input preparation and model configuration, see [data preparation and inference](docs/DATA_AND_INFERENCE.md).

## Repository structure

| Path | Contents |
| --- | --- |
| `corrosion_ie/` | Qwen extraction, retrieval and postprocessing |
| `strong_llm/` | GLM evidence strategies and three prompt sets |
| `configs/`, `prompts/`, `scripts/` | Experiment configurations, prompts and execution tools |
| `input/` | Gold annotations, paper/DOI manifest and document features |
| `output/exp_v2/` | Qwen structured predictions and run metadata |
| `vendor/`, `work/record_threshold/` | Evaluator, calibration labels and frozen threshold |
| `work/reevaluation_t038/` | Qwen evaluation and statistical tables |
| `analysis/` | Supplementary analyses and reference tables |
| `figures/` | GLM predictions, evaluation, plot data and plotting code |
| `release_manifest.json`, `SHA256SUMS.txt` | Data provenance and file integrity |

## Data availability and citation

The data release includes the structured reference annotations and formal predictions. Bibliographic identifiers and input checksums are supplied in `input/paper_manifest.csv`; source articles are subject to their respective copyright terms and are not redistributed. New inference requires independently acquired input documents and the model/API and retrieval resources described in the inference guide.

Code: [MIT License](LICENSE). Author-produced data: [CC BY 4.0](DATA_LICENSE.md). Please cite the associated study and software using [CITATION.cff](CITATION.cff).
