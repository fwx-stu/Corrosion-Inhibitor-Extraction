#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate the one-to-one record-matching threshold on dev37.

This is a post-processing tool. It never loads an LLM and never changes the
retrieval, prompt, similarity formula, field weights, or saved predictions.

Workflow
--------
1. Scan available prediction runs and create a condition allow-list candidate::

     python scripts/calibrate_record_threshold.py scan \
       --run-root output/exp_v2/EXPERIMENT \
       --output work/record_threshold/conditions.yaml

   Review the YAML and change ``include: false`` to ``include: true`` only for
   the final conditions reported in the paper.

2. Select complete paper x condition x seed blocks and create the workbook::

     python scripts/calibrate_record_threshold.py prepare \
       --project-root . \
       --gold input/dev37/gold/dev37.jsonl \
       --docs-dir input/dev37/md \
       --conditions work/record_threshold/conditions.yaml \
       --target-rows 300 \
       --boundary-rows 30 \
       --output work/record_threshold/record_threshold_review.xlsx

   An AI may prefill ``ai_gold_idx`` and ``ai_reason``. The expert must fill
   ``manual_gold_idx`` (G000, G001, ... or NONE) and set
   ``expert_confirmed`` to YES for every formal calibration row.

3. Scan thresholds, bootstrap by paper, and freeze the selected threshold::

     python scripts/calibrate_record_threshold.py fit \
       --project-root . \
       --gold input/dev37/gold/dev37.jsonl \
       --annotations work/record_threshold/record_threshold_review.xlsx \
       --output-dir work/record_threshold/results

Selection rule: maximize pooled (micro) Matching F1 on dev37; if several
thresholds tie, choose the largest. Paper-level macro F1 is reported as a
sensitivity analysis. The current evaluator's order (threshold edges first,
then Hungarian assignment) is the primary analysis. Hungarian first, threshold
second is written only as a diagnostic.

Runtime dependencies already used by the project: numpy, scipy, matplotlib,
PyYAML. Workbook generation additionally needs openpyxl>=3.1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.55
DEFAULT_BOOTSTRAP_SEED = 20260710
FIELD_NAMES = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
]


@dataclass(frozen=True)
class RunInfo:
    experiment: str
    model: str
    strategy: str
    prompt: str
    seed: str
    prediction_file: Path
    run_root: Path

    @property
    def condition_id(self) -> str:
        return "::".join((self.experiment, self.model, self.strategy, self.prompt))

    @property
    def run_id(self) -> str:
        return f"{self.condition_id}::seed={self.seed}"


@dataclass
class Block:
    run: RunInfo
    paper_id: str
    records: list[dict[str, Any]]
    prediction_sha256: str
    content_sha256: str
    top1_scores: list[float]

    @property
    def block_id(self) -> str:
        raw = f"{self.run.run_id}|{self.paper_id}|{self.content_sha256}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @property
    def row_count(self) -> int:
        return len(self.records)

    @property
    def uncertainty(self) -> float:
        if not self.top1_scores:
            return 0.0
        mean_distance = sum(abs(x - DEFAULT_THRESHOLD) for x in self.top1_scores) / len(self.top1_scores)
        return 1.0 / (0.02 + mean_distance)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_legacy(project_root: Path):
    path = project_root / "vendor" / "legacy_evaluate.py"
    if not path.exists():
        raise SystemExit(f"[error] evaluator not found: {path}")
    module_name = "legacy_evaluate_threshold_calibration"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[error] cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def ensure_dev37(gold_path: Path, gold_map: Mapping[str, Any]) -> None:
    if "test50" in str(gold_path).lower():
        raise SystemExit("[error] test50 is frozen and cannot be used for threshold calibration")
    if len(gold_map) != 37:
        raise SystemExit(
            f"[error] expected the complete dev37 Gold file (37 papers), found {len(gold_map)}: {gold_path}"
        )


def find_experiment_root(prediction_file: Path, boundary: Path) -> Path:
    for parent in prediction_file.parents:
        if parent == boundary.parent:
            break
        if (parent / "matrix_manifest.json").exists():
            return parent
        if parent == boundary:
            break
    return boundary


def parse_run(prediction_file: Path, supplied_root: Path) -> RunInfo | None:
    experiment_root = find_experiment_root(prediction_file, supplied_root)
    try:
        parts = prediction_file.relative_to(experiment_root).parts
    except ValueError:
        return None
    if len(parts) < 5:
        return None
    model, strategy, prompt, seed_dir = parts[-5:-1]
    seed_match = re.fullmatch(r"r(.+)", seed_dir)
    if not seed_match:
        return None
    return RunInfo(
        experiment=experiment_root.name,
        model=model,
        strategy=strategy,
        prompt=prompt,
        seed=seed_match.group(1),
        prediction_file=prediction_file.resolve(),
        run_root=experiment_root.resolve(),
    )


def discover_runs(run_roots: Iterable[Path]) -> list[RunInfo]:
    runs: list[RunInfo] = []
    seen: set[Path] = set()
    for root in run_roots:
        root = root.resolve()
        if not root.exists():
            raise SystemExit(f"[error] run root does not exist: {root}")
        for path in sorted(root.rglob("predictions.jsonl")):
            path = path.resolve()
            if path in seen:
                continue
            run = parse_run(path, root)
            if run is not None:
                runs.append(run)
                seen.add(path)
    if not runs:
        raise SystemExit("[error] no predictions.jsonl files were discovered")
    return sorted(runs, key=lambda x: (x.condition_id, x.seed, str(x.prediction_file)))


def scan_conditions(args: argparse.Namespace) -> None:
    roots = [Path(x).expanduser().resolve() for x in args.run_root]
    runs = discover_runs(roots)
    grouped: dict[str, list[RunInfo]] = defaultdict(list)
    for run in runs:
        grouped[run.condition_id].append(run)
    conditions = []
    for condition_id, members in sorted(grouped.items()):
        first = members[0]
        conditions.append(
            {
                "condition_id": condition_id,
                "experiment": first.experiment,
                "model": first.model,
                "strategy": first.strategy,
                "prompt": first.prompt,
                "seeds": sorted({m.seed for m in members}),
                "prediction_files": len(members),
                "include": False,
            }
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": "dev37",
        "created_at_utc": utc_now(),
        "run_roots": [str(x) for x in roots],
        "instructions": "Set include: true only for final paper conditions; keep test50 excluded.",
        "conditions": conditions,
    }
    output.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[scan] runs={len(runs)} conditions={len(conditions)}")
    print(f"[scan] review and edit: {output}")


def load_condition_allowlist(path: Path) -> tuple[set[str], list[Path]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(payload.get("dataset", "")).lower() != "dev37":
        raise SystemExit("[error] condition YAML must declare dataset: dev37")
    included = {
        str(item["condition_id"])
        for item in payload.get("conditions", [])
        if bool(item.get("include", False))
    }
    if not included:
        raise SystemExit("[error] no conditions are enabled; review YAML and set include: true")
    roots = [Path(x).expanduser().resolve() for x in payload.get("run_roots", [])]
    if not roots:
        raise SystemExit("[error] condition YAML contains no run_roots")
    return included, roots


def record_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in FIELD_NAMES}


def record_summary(record: Mapping[str, Any]) -> str:
    preferred = [
        "inhibitor_name",
        "substrate",
        "corrosive_medium",
        "medium_concentration",
        "inhibitor_concentration",
        "temperature",
        "immersion_time",
        "test_method",
        "inhibition_efficiency",
    ]
    pieces = [f"{key}={record.get(key)}" for key in preferred if record.get(key) not in (None, "")]
    return " | ".join(pieces)


def similarity_matrix(ev: Any, pred: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(pred), len(gold)), dtype=float)
    for i, pred_record in enumerate(pred):
        for j, gold_record in enumerate(gold):
            matrix[i, j] = float(ev.record_similarity(dict(pred_record), dict(gold_record)))
    return matrix


def dedupe_blocks(blocks: Sequence[Block]) -> tuple[list[Block], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[Block]] = defaultdict(list)
    for block in blocks:
        groups[(block.run.condition_id, block.paper_id, block.content_sha256)].append(block)
    kept: list[Block] = []
    audit: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        members = sorted(members, key=lambda b: (b.run.seed, str(b.run.prediction_file)))
        chosen_index = int(key[2][:8], 16) % len(members)
        chosen = members[chosen_index]
        kept.append(chosen)
        for member in members:
            audit.append(
                {
                    "condition_id": member.run.condition_id,
                    "paper_id": member.paper_id,
                    "seed": member.run.seed,
                    "content_sha256": member.content_sha256,
                    "kept": member is chosen,
                    "reason": "unique" if len(members) == 1 else "exact_content_hash_dedup",
                }
            )
    return sorted(kept, key=lambda b: (b.run.condition_id, b.run.seed, b.paper_id)), audit


def load_blocks(
    ev: Any,
    runs: Sequence[RunInfo],
    included_conditions: set[str],
    gold_map: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[Block], list[dict[str, Any]]]:
    blocks: list[Block] = []
    for run in runs:
        if run.condition_id not in included_conditions:
            continue
        pred_map, _ = ev.load_predictions(run.prediction_file)
        prediction_sha = sha256_file(run.prediction_file)
        for paper_id in sorted(gold_map):
            records = [record_view(x) for x in pred_map.get(paper_id, [])]
            if not records:
                continue
            gold_records = gold_map[paper_id]
            scores = similarity_matrix(ev, records, gold_records)
            top1 = scores.max(axis=1).tolist() if scores.size else [0.0] * len(records)
            blocks.append(
                Block(
                    run=run,
                    paper_id=paper_id,
                    records=records,
                    prediction_sha256=prediction_sha,
                    content_sha256=stable_hash(records),
                    top1_scores=[float(x) for x in top1],
                )
            )
    missing = sorted(included_conditions - {b.run.condition_id for b in blocks})
    if missing:
        raise SystemExit("[error] enabled conditions contain no non-empty dev37 blocks:\n  " + "\n  ".join(missing))
    return dedupe_blocks(blocks)


def paper_features(gold_map: Mapping[str, list[dict[str, Any]]], docs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for paper_id, gold_records in sorted(gold_map.items()):
        doc_path = docs_dir / f"{paper_id}.md"
        text = doc_path.read_text(encoding="utf-8", errors="ignore") if doc_path.exists() else ""
        lines = text.splitlines()
        table_chars = sum(len(line) for line in lines if line.lstrip().startswith("|"))
        quantum = sum(
            1
            for record in gold_records
            if any(record.get(field) not in (None, "") for field in ("homo", "lumo", "adsorption_energy"))
        )
        rows.append(
            {
                "paper_id": paper_id,
                "char_count": len(text),
                "table_ratio": table_chars / max(1, len(text)),
                "gold_count": len(gold_records),
                "quantum_record_count": quantum,
            }
        )
    return rows


def percentile(values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    result = np.zeros(len(values), dtype=float)
    if len(values) <= 1:
        return result.tolist()
    for rank, index in enumerate(order):
        result[index] = rank / (len(values) - 1)
    return result.tolist()


def select_representative_papers(features: list[dict[str, Any]], count: int) -> list[str]:
    if count >= len(features):
        return [str(x["paper_id"]) for x in features]
    columns = [
        percentile([math.log1p(float(x["char_count"])) for x in features]),
        percentile([float(x["table_ratio"]) for x in features]),
        percentile([math.log1p(float(x["gold_count"])) for x in features]),
        percentile([float(x["quantum_record_count"]) for x in features]),
    ]
    matrix = np.column_stack(columns)
    center = np.full(matrix.shape[1], 0.5)
    first = int(np.argmin(np.linalg.norm(matrix - center, axis=1)))
    chosen = [first]
    while len(chosen) < count:
        remaining = [i for i in range(len(features)) if i not in chosen]
        distances = []
        for index in remaining:
            minimum = min(float(np.linalg.norm(matrix[index] - matrix[j])) for j in chosen)
            distances.append((minimum, str(features[index]["paper_id"]), index))
        chosen.append(max(distances)[2])
    return [str(features[i]["paper_id"]) for i in chosen]


def choose_blocks(
    blocks: Sequence[Block],
    conditions: set[str],
    paper_pool: set[str],
    target_rows: int,
    max_rows: int,
    min_papers: int,
) -> list[Block]:
    eligible = [b for b in blocks if b.paper_id in paper_pool]
    by_condition: dict[str, list[Block]] = defaultdict(list)
    for block in eligible:
        by_condition[block.run.condition_id].append(block)
    unavailable = sorted(condition for condition in conditions if not by_condition.get(condition))
    if unavailable:
        raise SystemExit(
            "[error] representative paper pool has no usable block for conditions:\n  " + "\n  ".join(unavailable)
        )

    selected: list[Block] = []
    selected_ids: set[str] = set()
    paper_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()

    def add(block: Block) -> None:
        selected.append(block)
        selected_ids.add(block.block_id)
        paper_counts[block.paper_id] += 1
        seed_counts[block.run.seed] += 1
        condition_counts[block.run.condition_id] += 1

    # Mandatory coverage: one complete block per final condition.
    for condition in sorted(conditions):
        candidates = by_condition[condition]
        candidates = sorted(
            candidates,
            key=lambda b: (
                paper_counts[b.paper_id] > 0,
                seed_counts[b.run.seed],
                b.row_count,
                -b.uncertainty,
                b.paper_id,
            ),
        )
        add(candidates[0])

    mandatory_rows = sum(b.row_count for b in selected)
    if mandatory_rows > max_rows:
        detail = ", ".join(f"{c}={min(b.row_count for b in by_condition[c])}" for c in sorted(conditions))
        raise SystemExit(
            f"[error] minimum one-block-per-condition coverage needs {mandatory_rows} rows, "
            f"exceeding --max-rows={max_rows}. Smallest blocks: {detail}"
        )

    remaining = [b for b in eligible if b.block_id not in selected_ids]
    while remaining and (sum(b.row_count for b in selected) < target_rows or len(paper_counts) < min_papers):
        current_rows = sum(b.row_count for b in selected)
        candidates = [b for b in remaining if current_rows + b.row_count <= max_rows]
        if not candidates:
            break

        def priority(block: Block) -> tuple[float, str]:
            new_paper = 1.0 if block.paper_id not in paper_counts else 0.0
            paper_need = 5.0 * new_paper if len(paper_counts) < min_papers else 0.7 * new_paper
            condition_balance = 3.0 / (1.0 + condition_counts[block.run.condition_id])
            seed_balance = 2.0 / (1.0 + seed_counts[block.run.seed])
            uncertainty = min(3.0, block.uncertainty / 10.0)
            overshoot = abs(target_rows - (current_rows + block.row_count)) / max(1, target_rows)
            return (paper_need + condition_balance + seed_balance + uncertainty - overshoot, block.block_id)

        chosen = max(candidates, key=priority)
        add(chosen)
        remaining = [b for b in remaining if b.block_id != chosen.block_id]

    total_rows = sum(b.row_count for b in selected)
    if len(paper_counts) < min_papers:
        raise SystemExit(
            f"[error] only {len(paper_counts)} papers fit within {max_rows} rows; "
            f"required at least {min_papers}. Selected rows={total_rows}"
        )
    return sorted(selected, key=lambda b: (b.run.condition_id, b.run.seed, b.paper_id))


def ranked_candidates(ev: Any, pred: Mapping[str, Any], gold: Sequence[Mapping[str, Any]], top_k: int) -> list[tuple[int, float]]:
    values = [(index, float(ev.record_similarity(dict(pred), dict(record)))) for index, record in enumerate(gold)]
    return sorted(values, key=lambda x: (-x[1], x[0]))[:top_k]


def formal_rows(
    ev: Any,
    selected: Sequence[Block],
    gold_map: Mapping[str, list[dict[str, Any]]],
    top_k: int,
    project_root: Path,
    current_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in selected:
        gold = gold_map[block.paper_id]
        try:
            prediction_relpath = str(block.run.prediction_file.relative_to(project_root))
        except ValueError:
            prediction_relpath = str(block.run.prediction_file)
        for pred_index, pred in enumerate(block.records):
            candidates = ranked_candidates(ev, pred, gold, top_k)
            annotation_id = stable_hash(
                {"block_id": block.block_id, "pred_index": pred_index, "record": pred}
            )[:24]
            row: dict[str, Any] = {
                "annotation_id": annotation_id,
                "block_id": block.block_id,
                "condition_id": block.run.condition_id,
                "experiment": block.run.experiment,
                "model": block.run.model,
                "strategy": block.run.strategy,
                "prompt": block.run.prompt,
                "seed": block.run.seed,
                "paper_id": block.paper_id,
                "pred_idx": pred_index,
                "block_pred_count": block.row_count,
                "gold_count": len(gold),
                "current_threshold": current_threshold,
                "prediction_relpath": prediction_relpath,
                "prediction_sha256": block.prediction_sha256,
                "block_content_sha256": block.content_sha256,
                "pred_summary": record_summary(pred),
                "pred_record_json": json.dumps(pred, ensure_ascii=False, sort_keys=True),
            }
            for field in FIELD_NAMES:
                row[f"pred_{field}"] = pred.get(field)
            for rank in range(top_k):
                if rank < len(candidates):
                    gold_index, score = candidates[rank]
                    row[f"top{rank + 1}_gold_idx"] = f"G{gold_index:03d}"
                    row[f"top{rank + 1}_score"] = score
                    row[f"top{rank + 1}_summary"] = record_summary(gold[gold_index])
                else:
                    row[f"top{rank + 1}_gold_idx"] = ""
                    row[f"top{rank + 1}_score"] = ""
                    row[f"top{rank + 1}_summary"] = ""
            row.update(
                {
                    "ai_gold_idx": "",
                    "ai_reason": "",
                    "manual_gold_idx": "",
                    "expert_confirmed": "",
                    "expert_note": "",
                }
            )
            rows.append(row)
    return rows


def boundary_rows(
    ev: Any,
    blocks: Sequence[Block],
    gold_map: Mapping[str, list[dict[str, Any]]],
    count: int,
    center: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in blocks:
        gold = gold_map[block.paper_id]
        for pred_index, pred in enumerate(block.records):
            for gold_index, gold_record in enumerate(gold):
                score = float(ev.record_similarity(pred, gold_record))
                row = {
                    "excluded_from_threshold_selection": True,
                    "condition_id": block.run.condition_id,
                    "seed": block.run.seed,
                    "paper_id": block.paper_id,
                    "pred_idx": pred_index,
                    "gold_idx": f"G{gold_index:03d}",
                    "similarity": score,
                    "distance_from_current_threshold": abs(score - center),
                    "pred_summary": record_summary(pred),
                    "gold_summary": record_summary(gold_record),
                    "manual_same_record": "",
                    "error_note": "",
                }
                candidates.append((abs(score - center), row))
    candidates.sort(key=lambda x: (x[0], x[1]["condition_id"], x[1]["paper_id"]))
    result: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    for _, row in candidates:
        if len(result) >= count:
            break
        condition_id = str(row["condition_id"])
        if condition_counts[condition_id] >= 3:
            continue
        result.append(row)
        condition_counts[condition_id] += 1
    return result


def import_openpyxl():
    try:
        import openpyxl
        from openpyxl.formatting.rule import ColorScaleRule
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError as exc:
        raise SystemExit("[error] workbook generation requires: pip install 'openpyxl>=3.1'") from exc
    return openpyxl, ColorScaleRule, Alignment, Font, PatternFill, DataValidation


def append_table(ws: Any, rows: Sequence[Mapping[str, Any]], styles: tuple[Any, ...]) -> None:
    _, ColorScaleRule, Alignment, Font, PatternFill, _ = styles
    if not rows:
        ws.append(["no_rows"])
        return
    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(header, "") for header in headers])
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 36
    for column_cells in ws.columns:
        letter = column_cells[0].column_letter
        header = str(column_cells[0].value or "")
        if "summary" in header or "json" in header or "reason" in header or "note" in header:
            width = 42
        elif "sha256" in header or "condition_id" in header or "relpath" in header:
            width = 28
        else:
            width = min(24, max(11, len(header) + 2))
        ws.column_dimensions[letter].width = width
        for cell in column_cells[1:]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    score_columns = [i + 1 for i, header in enumerate(headers) if header.endswith("_score") or header == "similarity"]
    for col in score_columns:
        if ws.max_row >= 2:
            ws.conditional_formatting.add(
                f"{ws.cell(2, col).coordinate}:{ws.cell(ws.max_row, col).coordinate}",
                ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50,
                               mid_color="FFEB84", end_type="max", end_color="63BE7B"),
            )
            for row in range(2, ws.max_row + 1):
                ws.cell(row, col).number_format = "0.000"


def create_workbook(
    output: Path,
    rows: list[dict[str, Any]],
    gold_map: Mapping[str, list[dict[str, Any]]],
    selected_blocks: Sequence[Block],
    feature_rows: Sequence[Mapping[str, Any]],
    boundary: Sequence[Mapping[str, Any]],
    dedup_audit: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    openpyxl, ColorScaleRule, Alignment, Font, PatternFill, DataValidation = import_openpyxl()
    styles = (openpyxl, ColorScaleRule, Alignment, Font, PatternFill, DataValidation)
    wb = openpyxl.Workbook()
    instructions = wb.active
    instructions.title = "Instructions"
    instructions_rows = [
        ("Purpose", "Calibrate the record-matching threshold exclusively on dev37."),
        ("Formal unit", "Every Calibration row is one prediction record from a complete paper × condition × seed block."),
        ("Same record", "Use the same original experiment/table row. Missing or wrong fields are allowed."),
        ("Mixed record", "If one prediction merges multiple Gold records, enter NONE."),
        ("AI step", "AI may fill ai_gold_idx and ai_reason, but these are not ground truth."),
        ("Expert step", "Fill manual_gold_idx with G000/G001/... or NONE, then set expert_confirmed=YES."),
        ("One-to-one rule", "Within one block, the same Gold ID cannot be assigned to two predictions."),
        ("Visibility", "Similarity, candidate rank, and current threshold are intentionally visible."),
        ("Boundary sheet", "Boundary_analysis is for error analysis only and is excluded from threshold selection."),
        ("Selection", "Highest pooled Micro Matching F1; ties use the larger threshold."),
    ]
    for key, value in instructions_rows:
        instructions.append([key, value])
    instructions.column_dimensions["A"].width = 22
    instructions.column_dimensions["B"].width = 100
    for cell in instructions["A"]:
        cell.font = Font(bold=True, color="1F4E78")
    for row in instructions.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    calibration = wb.create_sheet("Calibration")
    append_table(calibration, rows, styles)
    headers = {cell.value: cell.column for cell in calibration[1]}
    confirm_col = headers.get("expert_confirmed")
    if confirm_col and calibration.max_row >= 2:
        dv = DataValidation(type="list", formula1='"YES,NO"', allow_blank=True)
        calibration.add_data_validation(dv)
        dv.add(f"{calibration.cell(2, confirm_col).coordinate}:{calibration.cell(calibration.max_row, confirm_col).coordinate}")
    for name in ("ai_gold_idx", "ai_reason", "manual_gold_idx", "expert_confirmed", "expert_note"):
        col = headers.get(name)
        if col:
            fill = PatternFill("solid", fgColor="FFF2CC" if name.startswith("ai_") else "E2F0D9")
            for row_index in range(2, calibration.max_row + 1):
                calibration.cell(row_index, col).fill = fill

    gold_rows: list[dict[str, Any]] = []
    selected_papers = sorted({b.paper_id for b in selected_blocks})
    for paper_id in selected_papers:
        for index, record in enumerate(gold_map[paper_id]):
            row = {"paper_id": paper_id, "gold_idx": f"G{index:03d}", "gold_summary": record_summary(record)}
            row.update({field: record.get(field) for field in FIELD_NAMES})
            gold_rows.append(row)
    gold_ws = wb.create_sheet("Gold_catalog")
    append_table(gold_ws, gold_rows, styles)

    block_rows = [
        {
            "block_id": b.block_id,
            "condition_id": b.run.condition_id,
            "model": b.run.model,
            "strategy": b.run.strategy,
            "prompt": b.run.prompt,
            "seed": b.run.seed,
            "paper_id": b.paper_id,
            "pred_count": b.row_count,
            "mean_top1_similarity": float(np.mean(b.top1_scores)),
            "min_top1_similarity": float(np.min(b.top1_scores)),
            "max_top1_similarity": float(np.max(b.top1_scores)),
            "block_content_sha256": b.content_sha256,
        }
        for b in selected_blocks
    ]
    blocks_ws = wb.create_sheet("Selected_blocks")
    append_table(blocks_ws, block_rows, styles)

    paper_ws = wb.create_sheet("Paper_selection")
    selected_set = set(selected_papers)
    selected_feature_rows = [dict(row, selected=(str(row["paper_id"]) in selected_set)) for row in feature_rows]
    append_table(paper_ws, selected_feature_rows, styles)

    boundary_ws = wb.create_sheet("Boundary_analysis")
    append_table(boundary_ws, boundary, styles)
    if boundary_ws.max_row >= 2:
        headers_boundary = {cell.value: cell.column for cell in boundary_ws[1]}
        col = headers_boundary.get("manual_same_record")
        if col:
            dv = DataValidation(type="list", formula1='"YES,NO,UNCERTAIN"', allow_blank=True)
            boundary_ws.add_data_validation(dv)
            dv.add(f"{boundary_ws.cell(2, col).coordinate}:{boundary_ws.cell(boundary_ws.max_row, col).coordinate}")

    audit_ws = wb.create_sheet("Dedup_audit")
    append_table(audit_ws, dedup_audit, styles)

    meta_ws = wb.create_sheet("Audit_metadata")
    meta_rows = [{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                 for key, value in metadata.items()]
    append_table(meta_ws, meta_rows, styles)
    meta_ws.sheet_state = "hidden"

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def prepare(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    gold_path = resolve_path(project_root, args.gold)
    docs_dir = resolve_path(project_root, args.docs_dir)
    condition_path = resolve_path(project_root, args.conditions)
    output = resolve_path(project_root, args.output)
    ev = load_legacy(project_root)
    gold_map = ev.load_gold(gold_path)
    ensure_dev37(gold_path, gold_map)
    included, roots = load_condition_allowlist(condition_path)
    runs = discover_runs(roots)
    unknown = sorted(included - {r.condition_id for r in runs})
    if unknown:
        raise SystemExit("[error] enabled conditions were not found in run roots:\n  " + "\n  ".join(unknown))
    blocks, dedup_audit = load_blocks(ev, runs, included, gold_map)
    features = paper_features(gold_map, docs_dir)
    representative = select_representative_papers(features, args.paper_count)
    selected = choose_blocks(
        blocks=blocks,
        conditions=included,
        paper_pool=set(representative),
        target_rows=args.target_rows,
        max_rows=args.max_rows,
        min_papers=args.min_papers,
    )
    rows = formal_rows(ev, selected, gold_map, args.top_k, project_root, args.current_threshold)
    boundary = boundary_rows(ev, selected, gold_map, args.boundary_rows, args.current_threshold)
    total_review = len(rows) + len(boundary)
    if total_review > args.max_total_rows:
        raise SystemExit(
            f"[error] formal rows ({len(rows)}) + boundary rows ({len(boundary)}) = {total_review}, "
            f"exceeding --max-total-rows={args.max_total_rows}"
        )
    condition_counts = Counter(b.run.condition_id for b in selected)
    missing = sorted(included - set(condition_counts))
    if missing:
        raise SystemExit("[error] internal coverage failure: " + ", ".join(missing))
    metadata = {
        "schema_version": 1,
        "dataset": "dev37",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "conditions_path": str(condition_path),
        "conditions_sha256": sha256_file(condition_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "current_threshold": args.current_threshold,
        "formal_row_count": len(rows),
        "boundary_row_count": len(boundary),
        "selected_block_count": len(selected),
        "selected_paper_count": len({b.paper_id for b in selected}),
        "selected_condition_count": len(condition_counts),
        "selection_rule": "max_micro_matching_f1_then_largest_threshold",
        "primary_matching_order": "threshold_before_hungarian",
        "alternate_matching_order": "hungarian_before_threshold_diagnostic_only",
    }
    create_workbook(output, rows, gold_map, selected, features, boundary, dedup_audit, metadata)
    print(
        f"[prepare] conditions={len(condition_counts)} blocks={len(selected)} "
        f"papers={len({b.paper_id for b in selected})} formal_rows={len(rows)} "
        f"boundary_rows={len(boundary)}"
    )
    print(f"[prepare] workbook: {output}")


def read_sheet_rows(ws: Any) -> list[dict[str, Any]]:
    headers = [str(cell.value or "").strip() for cell in ws[1]]
    result = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        if not any(value not in (None, "") for value in values):
            continue
        result.append({headers[i]: values[i] for i in range(len(headers))})
    return result


def parse_manual_gold(value: Any, gold_count: int) -> int | None:
    text = str(value or "").strip().upper()
    if text in {"NONE", "NO_MATCH", "NULL"}:
        return None
    match = re.fullmatch(r"G?(\d+)", text)
    if not match:
        raise ValueError(f"expected G000/G001/... or NONE, got {value!r}")
    index = int(match.group(1))
    if not 0 <= index < gold_count:
        raise ValueError(f"Gold index G{index:03d} is outside 0..{gold_count - 1}")
    return index


def load_annotations(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    openpyxl, *_ = import_openpyxl()
    wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    if "Calibration" not in wb.sheetnames or "Audit_metadata" not in wb.sheetnames:
        raise SystemExit("[error] workbook lacks Calibration or Audit_metadata sheet")
    rows = read_sheet_rows(wb["Calibration"])
    meta_rows = read_sheet_rows(wb["Audit_metadata"])
    metadata = {str(row["key"]): row.get("value") for row in meta_rows}
    if not rows:
        raise SystemExit("[error] Calibration sheet is empty")
    required = {
        "annotation_id", "block_id", "condition_id", "seed", "paper_id", "pred_idx",
        "gold_count", "prediction_relpath", "prediction_sha256", "block_content_sha256",
        "manual_gold_idx", "expert_confirmed",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit("[error] Calibration sheet missing columns: " + ", ".join(missing))
    seen_annotations: set[str] = set()
    used_gold: dict[str, set[int]] = defaultdict(set)
    errors: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        annotation_id = str(row["annotation_id"])
        if annotation_id in seen_annotations:
            errors.append(f"row {row_number}: duplicate annotation_id {annotation_id}")
        seen_annotations.add(annotation_id)
        if str(row.get("expert_confirmed") or "").strip().upper() != "YES":
            errors.append(f"row {row_number}: expert_confirmed must be YES")
            continue
        try:
            manual = parse_manual_gold(row.get("manual_gold_idx"), int(row["gold_count"]))
            row["manual_gold_index"] = manual
        except Exception as exc:
            errors.append(f"row {row_number}: {exc}")
            continue
        block_id = str(row["block_id"])
        if manual is not None:
            if manual in used_gold[block_id]:
                errors.append(f"row {row_number}: duplicate G{manual:03d} assignment within block {block_id}")
            used_gold[block_id].add(manual)
    if errors:
        preview = "\n  ".join(errors[:30])
        suffix = f"\n  ... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise SystemExit(f"[error] annotation validation failed:\n  {preview}{suffix}")
    return rows, metadata


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    start_d, stop_d, step_d = Decimal(str(start)), Decimal(str(stop)), Decimal(str(step))
    if step_d <= 0 or stop_d < start_d:
        raise SystemExit("[error] threshold range must satisfy step>0 and max>=min")
    result = []
    value = start_d
    while value <= stop_d + Decimal("1e-12"):
        result.append(float(value))
        value += step_d
    return result


def matched_pairs(scores: np.ndarray, threshold: float, order: str) -> set[tuple[int, int]]:
    if scores.size == 0:
        return set()
    if order == "threshold_before_hungarian":
        objective = np.where(scores >= threshold, scores, 0.0)
    elif order == "hungarian_before_threshold":
        objective = scores
    else:
        raise ValueError(order)
    row_ind, col_ind = linear_sum_assignment(-objective)
    return {
        (int(i), int(j))
        for i, j in zip(row_ind, col_ind)
        if float(scores[int(i), int(j)]) >= threshold
    }


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_block_data(
    ev: Any,
    project_root: Path,
    gold_map: Mapping[str, list[dict[str, Any]]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotation_rows:
        grouped[str(row["block_id"])].append(row)
    cache: dict[Path, Mapping[str, list[dict[str, Any]]]] = {}
    result: dict[str, dict[str, Any]] = {}
    for block_id, rows in sorted(grouped.items()):
        first = rows[0]
        pred_path = resolve_path(project_root, str(first["prediction_relpath"]))
        expected_sha = str(first["prediction_sha256"])
        actual_sha = sha256_file(pred_path)
        if actual_sha != expected_sha:
            raise SystemExit(f"[error] prediction hash changed: {pred_path}")
        if pred_path not in cache:
            cache[pred_path], _ = ev.load_predictions(pred_path)
        paper_id = str(first["paper_id"])
        pred_records = [record_view(x) for x in cache[pred_path].get(paper_id, [])]
        content_sha = stable_hash(pred_records)
        if content_sha != str(first["block_content_sha256"]):
            raise SystemExit(f"[error] block content changed: {block_id}")
        expected_indices = set(range(len(pred_records)))
        annotated_indices = {int(row["pred_idx"]) for row in rows}
        if annotated_indices != expected_indices:
            raise SystemExit(
                f"[error] block {block_id} is not fully annotated: expected {len(expected_indices)} rows, "
                f"found {len(annotated_indices)}"
            )
        manual_pairs = {
            (int(row["pred_idx"]), int(row["manual_gold_index"]))
            for row in rows
            if row.get("manual_gold_index") is not None
        }
        gold_records = gold_map[paper_id]
        result[block_id] = {
            "block_id": block_id,
            "paper_id": paper_id,
            "condition_id": str(first["condition_id"]),
            "seed": str(first["seed"]),
            "scores": similarity_matrix(ev, pred_records, gold_records),
            "manual_pairs": manual_pairs,
            "prediction_file": str(pred_path),
            "prediction_sha256": actual_sha,
        }
    return result


def evaluate_thresholds(
    block_data: Mapping[str, Mapping[str, Any]],
    thresholds: Sequence[float],
    order: str,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, tuple[int, int, int]]]]:
    metrics: list[dict[str, Any]] = []
    per_paper: dict[float, dict[str, list[int]]] = {}
    for threshold in thresholds:
        total_tp = total_fp = total_fn = 0
        paper_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        accepted = 0
        for block in block_data.values():
            predicted = matched_pairs(block["scores"], threshold, order)
            manual = set(block["manual_pairs"])
            tp = len(predicted & manual)
            fp = len(predicted - manual)
            fn = len(manual - predicted)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            accepted += len(predicted)
            counts = paper_counts[str(block["paper_id"])]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
        precision, recall, f1 = prf(total_tp, total_fp, total_fn)
        paper_f1 = [prf(*counts)[2] for counts in paper_counts.values()]
        metrics.append(
            {
                "threshold": threshold,
                "matching_precision": precision,
                "matching_recall": recall,
                "matching_f1_micro": f1,
                "matching_f1_macro_paper": float(np.mean(paper_f1)) if paper_f1 else 0.0,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "accepted_pairs": accepted,
                "papers": len(paper_counts),
                "blocks": len(block_data),
                "matching_order": order,
            }
        )
        per_paper[threshold] = {paper: tuple(counts) for paper, counts in paper_counts.items()}
    return metrics, per_paper


def select_threshold(metrics: Sequence[Mapping[str, Any]]) -> float:
    max_f1 = max(float(row["matching_f1_micro"]) for row in metrics)
    tied = [float(row["threshold"]) for row in metrics if math.isclose(float(row["matching_f1_micro"]), max_f1,
                                                                         rel_tol=0.0, abs_tol=1e-12)]
    return max(tied)


def bootstrap_by_paper(
    per_paper: Mapping[float, Mapping[str, tuple[int, int, int]]],
    thresholds: Sequence[float],
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    papers = sorted(next(iter(per_paper.values())).keys())
    counts = np.asarray(
        [[per_paper[t][paper] for t in thresholds] for paper in papers],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    f1_samples = np.zeros((iterations, len(thresholds)), dtype=float)
    selected_counts: Counter[float] = Counter()
    for iteration in range(iterations):
        sampled = rng.integers(0, len(papers), size=len(papers))
        totals = counts[sampled].sum(axis=0)
        tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=(precision + recall) != 0)
        f1_samples[iteration] = f1
        maximum = float(f1.max())
        best_indices = np.flatnonzero(np.isclose(f1, maximum, rtol=0.0, atol=1e-12))
        selected_counts[float(thresholds[int(best_indices[-1])])] += 1
    ci_rows = []
    frequency_rows = []
    for index, threshold in enumerate(thresholds):
        values = f1_samples[:, index]
        ci_rows.append(
            {
                "threshold": threshold,
                "bootstrap_mean_micro_f1": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "iterations": iterations,
            }
        )
        frequency_rows.append(
            {
                "threshold": threshold,
                "selected_count": selected_counts[float(threshold)],
                "selected_frequency": selected_counts[float(threshold)] / iterations,
                "iterations": iterations,
            }
        )
    return ci_rows, frequency_rows


def plot_metrics(path: Path, primary: Sequence[Mapping[str, Any]], alternate: Sequence[Mapping[str, Any]], selected: float) -> None:
    import matplotlib.pyplot as plt

    thresholds = [float(row["threshold"]) for row in primary]
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(thresholds, [row["matching_precision"] for row in primary], label="Precision", linewidth=2)
    plt.plot(thresholds, [row["matching_recall"] for row in primary], label="Recall", linewidth=2)
    plt.plot(thresholds, [row["matching_f1_micro"] for row in primary], label="Micro F1 (primary)", linewidth=2.5)
    plt.plot(thresholds, [row["matching_f1_micro"] for row in alternate], label="Micro F1 (alternate diagnostic)",
             linewidth=1.5, linestyle="--")
    plt.axvline(selected, color="black", linestyle=":", linewidth=1.5, label=f"Selected = {selected:.2f}")
    plt.xlabel("Record-matching threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def fit(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    gold_path = resolve_path(project_root, args.gold)
    annotation_path = resolve_path(project_root, args.annotations)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ev = load_legacy(project_root)
    gold_map = ev.load_gold(gold_path)
    ensure_dev37(gold_path, gold_map)
    rows, workbook_meta = load_annotations(annotation_path)
    expected_gold_sha = str(workbook_meta.get("gold_sha256") or "").strip()
    if expected_gold_sha and expected_gold_sha != sha256_file(gold_path):
        raise SystemExit("[error] Gold file hash differs from the workbook audit metadata")
    block_data = load_block_data(ev, project_root, gold_map, rows)
    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    primary, per_paper = evaluate_thresholds(block_data, thresholds, "threshold_before_hungarian")
    alternate, _ = evaluate_thresholds(block_data, thresholds, "hungarian_before_threshold")
    selected = select_threshold(primary)
    alternate_selected = select_threshold(alternate)
    ci_rows, frequency_rows = bootstrap_by_paper(
        per_paper, thresholds, args.bootstrap, args.bootstrap_seed
    )
    sensitivity_values = set(threshold_grid(0.40, 0.70, 0.05))
    sensitivity = [row for row in primary if any(math.isclose(float(row["threshold"]), value, abs_tol=1e-12)
                                                  for value in sensitivity_values)]
    diagnostics = []
    alternate_by_threshold = {float(row["threshold"]): row for row in alternate}
    for row in primary:
        threshold = float(row["threshold"])
        other = alternate_by_threshold[threshold]
        diagnostics.append(
            {
                "threshold": threshold,
                "primary_micro_f1": row["matching_f1_micro"],
                "alternate_micro_f1": other["matching_f1_micro"],
                "delta_alternate_minus_primary": float(other["matching_f1_micro"]) - float(row["matching_f1_micro"]),
                "primary_accepted_pairs": row["accepted_pairs"],
                "alternate_accepted_pairs": other["accepted_pairs"],
            }
        )
    write_csv(output_dir / "threshold_metrics_primary.csv", primary)
    write_csv(output_dir / "threshold_metrics_alternate_diagnostic.csv", alternate)
    write_csv(output_dir / "threshold_sensitivity_0.40_0.70.csv", sensitivity)
    write_csv(output_dir / "threshold_bootstrap_ci.csv", ci_rows)
    write_csv(output_dir / "threshold_selection_frequency.csv", frequency_rows)
    write_csv(output_dir / "matching_order_diagnostics.csv", diagnostics)
    plot_metrics(output_dir / "threshold_calibration.png", primary, alternate, selected)
    prediction_hashes = sorted({
        (str(block["prediction_file"]), str(block["prediction_sha256"])) for block in block_data.values()
    })
    selected_row = next(row for row in primary if math.isclose(float(row["threshold"]), selected, abs_tol=1e-12))
    freeze = {
        "schema_version": 1,
        "dataset": "dev37",
        "frozen_at_utc": utc_now(),
        "selected_threshold": selected,
        "selection_rule": "maximum pooled micro Matching F1; ties resolved to the largest threshold",
        "selected_metrics": selected_row,
        "threshold_scan": {
            "minimum": args.threshold_min,
            "maximum": args.threshold_max,
            "step": args.threshold_step,
        },
        "primary_matching_order": "threshold_before_hungarian",
        "alternate_matching_order_diagnostic_only": "hungarian_before_threshold",
        "alternate_selected_threshold_not_used": alternate_selected,
        "bootstrap": {
            "unit": "paper",
            "iterations": args.bootstrap,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "paper_count": len({str(block["paper_id"]) for block in block_data.values()}),
        "block_count": len(block_data),
        "annotation_row_count": len(rows),
        "gold_file": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "annotations_file": str(annotation_path),
        "annotations_sha256": sha256_file(annotation_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "prediction_files": [
            {"path": path, "sha256": digest} for path, digest in prediction_hashes
        ],
        "test50_used": False,
    }
    write_json(output_dir / "selected_threshold_frozen.json", freeze)
    print(
        f"[fit] selected_threshold={selected:.2f} micro_f1={float(selected_row['matching_f1_micro']):.4f} "
        f"macro_f1={float(selected_row['matching_f1_macro_paper']):.4f}"
    )
    print(f"[fit] frozen result: {output_dir / 'selected_threshold_frozen.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan runs and write a YAML condition candidate list")
    scan_parser.add_argument("--run-root", action="append", required=True,
                             help="experiment run root; repeat for additional final dev37 experiments")
    scan_parser.add_argument("--output", required=True)
    scan_parser.set_defaults(func=scan_conditions)

    prepare_parser = subparsers.add_parser("prepare", help="create the expert-review workbook")
    prepare_parser.add_argument("--project-root", default=str(ROOT))
    prepare_parser.add_argument("--gold", default="input/dev37/gold/dev37.jsonl")
    prepare_parser.add_argument("--docs-dir", default="input/dev37/md")
    prepare_parser.add_argument("--conditions", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--paper-count", type=int, default=18, choices=range(15, 21), metavar="15..20")
    prepare_parser.add_argument("--min-papers", type=int, default=15)
    prepare_parser.add_argument("--target-rows", type=int, default=300,
                                help="target formal calibration rows from complete blocks")
    prepare_parser.add_argument("--max-rows", type=int, default=320,
                                help="hard maximum formal calibration rows")
    prepare_parser.add_argument("--boundary-rows", type=int, default=30)
    prepare_parser.add_argument("--max-total-rows", type=int, default=350)
    prepare_parser.add_argument("--top-k", type=int, default=5)
    prepare_parser.add_argument("--current-threshold", type=float, default=DEFAULT_THRESHOLD)
    prepare_parser.set_defaults(func=prepare)

    fit_parser = subparsers.add_parser("fit", help="validate annotations, scan thresholds, and freeze the result")
    fit_parser.add_argument("--project-root", default=str(ROOT))
    fit_parser.add_argument("--gold", default="input/dev37/gold/dev37.jsonl")
    fit_parser.add_argument("--annotations", required=True)
    fit_parser.add_argument("--output-dir", required=True)
    fit_parser.add_argument("--threshold-min", type=float, default=0.30)
    fit_parser.add_argument("--threshold-max", type=float, default=0.80)
    fit_parser.add_argument("--threshold-step", type=float, default=0.01)
    fit_parser.add_argument("--bootstrap", type=int, default=10000)
    fit_parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    fit_parser.set_defaults(func=fit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
