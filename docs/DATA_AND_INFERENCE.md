# Data preparation and inference

## Input data

`input/paper_manifest.csv` maps the 87 paper IDs to the dev37/test50 splits, bibliographic DOIs, source checksums and document lengths. The `doi_status` column distinguishes title-matched identifiers from source-header candidates. The normalized document features used in statistical analysis are provided separately.

Obtain the source articles through authorized channels. Convert PDFs using MinerU 2.5 or convert publisher XML to Markdown, retaining the original section order and table structure. Store each prepared document as `input/<split>/md/<paper_id>.md`. The manifest's source checksums identify the original experimental Markdown. The public package provides the downstream retrieval and table-processing code; the historical PDF/XML conversion environment is not included.

## Qwen2.5 experiments

Install `requirements.txt` in a model-inference environment. Configure Qwen2.5-3B/7B/14B-Instruct, BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 using `configs/model_registry.yaml` and the model paths. The recorded runtime and dependency versions are in `docs/environment/`.

| Configuration | Experiment |
| --- | --- |
| `experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml` | 27 development conditions |
| `experiment_q0q1q2_test50_balanced_opt_v6.yaml` | 9 main test conditions |
| `experiment_q0q1q2_test50_balanced_opt_v6_gpu_3b7b.yaml`, `..._cpu_14b.yaml` | Hardware-specific main test batches |
| `experiment_reviewer_controls_test50_v1_gpu_3b7b.yaml`, `..._cpu_14b.yaml` | 12 matched-budget and component controls |

The formal generation settings use 23,000 input tokens, 8,192 maximum output tokens and greedy decoding (temperature 0). The configurations specify retrieval parameters, evidence budgets and prompt policies.

| Strategy | Description |
| --- | --- |
| Q0 | Sequential full-text input under the token budget |
| Q1 | Field-group local retrieval and evidence budgeting |
| Q2 | Table-aware evidence budgeting and condition anchors |
| B0 | Uniform non-retrieval sampling matched to Q1 input budget |
| B1 | Section-aware non-retrieval sampling matched to Q2 input budget |
| A1 | Table-budget component |
| A2 | Condition-anchor component |

For new inference, copy the desired configuration within `configs/` (for example, `configs/local_test50.yaml`), choose a separate `experiment_id` and `output_root`, and configure the model paths. Inspect the new matrix before execution:

```bash
python scripts/run_matrix.py --config configs/local_test50.yaml --dry-run
python scripts/run_matrix.py --config configs/local_test50.yaml
```

The dry run also writes a matrix manifest, so use the separate output root for both commands. The saved token budgets and frozen protocols document the original experiments. Use `scripts/freeze_protocol.py` when preparing a new frozen run. Preserve the published prediction directories for evaluation.

## GLM-4-Plus evidence strategies

Use a separate environment for `strong_llm/requirements.txt`. Configure API credentials through environment variables or a local `.env` (see `.env.example`), and configure the embedding and reranking backend in `strong_llm/config.py`. Place the prepared dev37 Markdown files in `strong_llm/input/md/` and the reference data in `strong_llm/input/gold/train.jsonl`.

The strong-model implementation uses the following strategy identifiers:

| Manuscript | Code | Method |
| --- | --- | --- |
| E0 | E0 | Full text |
| E1 | E1 | Wiki-distilled external context |
| E2 | E3 | Local RAG |
| E3 | E4 | Post-extraction verification |
| E4 | E5 | Global domain RAG |
| E5 | E6 | Hybrid RAG |

Example generation commands, from `strong_llm/`:

```bash
python run.py --mode compare --strategies E0 --provider zhipu --model glm-4-plus --limit 1
python multi_run.py --strategies E0,E1,E3,E4,E5,E6 --variants 1,2,3 --runs 3
```

The second command assumes all external knowledge and retrieval resources have been configured. The external domain corpus and Milvus index are separate resources required for external-context strategies. Retrieval schema, parameters, weighting and same-DOI exclusion are implemented in `strong_llm/config.py` and `strong_llm/retrieval.py`. The external corpus and index are not part of the distributed dataset. Exact generation from raw articles depends on these resources and on source conversion and API versions.

## GLM test50 reference

The test50 reference predictions are supplied in `figures/GLM_predictions/predictions.jsonl`. The available model registry specifies GLM-4-Plus, a 23,000-token input budget, 8,192 maximum output tokens, temperature 0.05, top-p 1 and sampling enabled. The saved predictions are the input for reproducing the reference metrics. The registry documents the available settings rather than a complete historical launch manifest.

For publication metrics, use the common 0.38 evaluation workflow in the root README instead of the strong-model pipeline's standalone evaluation defaults.
