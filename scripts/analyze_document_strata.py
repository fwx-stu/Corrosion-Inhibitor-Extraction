#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata

STRATEGY_PAIRS = [
    ("q0_safe_full", "q1_budgeted_rag", "Q1-Q0"),
    ("q1_budgeted_rag", "q2_table_aware_rag", "Q2-Q1"),
    ("q0_safe_full", "q2_table_aware_rag", "Q2-Q0"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def prf(tp: float, pred: float, gold: float) -> Tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def markdown_table_chars(text: str) -> int:
    total = 0
    for block in re.finditer(r"(?:^|\n)(?:\s*\|[^\n]*\|\s*\n){2,}", text, re.M):
        total += len(block.group(0))
    return total


def document_features(md_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(md_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        html_tables = list(re.finditer(r"<table\b[^>]*>.*?</table>", text, re.I | re.S))
        table_chars = sum(len(match.group(0)) for match in html_tables) + markdown_table_chars(text)
        rows.append({
            "paper_id": path.stem,
            "document_chars": len(text),
            "html_table_count": len(html_tables),
            "table_chars": table_chars,
            "table_char_fraction": table_chars / max(1, len(text)),
        })
    return rows


def quartile_labels(values: Sequence[float]) -> List[str]:
    if not values:
        return []
    ordered = sorted(values)

    def quantile(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(len(ordered) - 1, lo + 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    q1, q2, q3 = quantile(0.25), quantile(0.50), quantile(0.75)
    labels = []
    for value in values:
        if value <= q1:
            labels.append("Q1_low")
        elif value <= q2:
            labels.append("Q2_mid_low")
        elif value <= q3:
            labels.append("Q3_mid_high")
        else:
            labels.append("Q4_high")
    return labels


def condition_paths(run_root: Path) -> List[Dict[str, Any]]:
    manifest = json.loads((run_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    return list(manifest.get("jobs", []))


def load_condition(job: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    run_dir = Path(job["run_dir"])
    paper_rows = read_csv(run_dir / "evaluation" / "per_paper_metrics.csv")
    field_rows = read_csv(run_dir / "evaluation" / "per_paper_field_metrics.csv")
    result: Dict[str, Dict[str, float]] = {}
    for row in paper_rows:
        pid = row["paper_id"]
        result[pid] = {
            "record_tp": num(row.get("record_tp")),
            "record_pred": num(row.get("pred_record_count")),
            "record_gold": num(row.get("gold_record_count")),
            "field_tp": 0.0,
            "field_pred": 0.0,
            "field_gold": 0.0,
            "null_correct": num(row.get("correct_null_count")),
            "null_total": num(row.get("gold_null_count_matched")),
        }
    for row in field_rows:
        pid = row["paper_id"]
        if pid not in result:
            continue
        result[pid]["field_tp"] += num(row.get("tp"))
        result[pid]["field_pred"] += num(row.get("pred_count"))
        result[pid]["field_gold"] += num(row.get("gold_count"))
    return result


def aggregate(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(records)
    r_tp = sum(row["record_tp"] for row in rows)
    r_pred = sum(row["record_pred"] for row in rows)
    r_gold = sum(row["record_gold"] for row in rows)
    f_tp = sum(row["field_tp"] for row in rows)
    f_pred = sum(row["field_pred"] for row in rows)
    f_gold = sum(row["field_gold"] for row in rows)
    r_p, r_r, r_f1 = prf(r_tp, r_pred, r_gold)
    f_p, f_r, f_f1 = prf(f_tp, f_pred, f_gold)
    null_total = sum(row["null_total"] for row in rows)
    null_correct = sum(row["null_correct"] for row in rows)
    return {
        "paper_count": len(rows),
        "record_precision": r_p,
        "record_recall": r_r,
        "record_f1": r_f1,
        "field_micro_precision": f_p,
        "field_micro_recall": f_r,
        "field_micro_f1": f_f1,
        "null_accuracy": null_correct / null_total if null_total else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze model/strategy performance by document length and table density quartiles.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_root = Path(args.run_root) if args.run_root else cfg.run_root
    features = document_features(cfg.md_dir)
    if not features:
        # Derived numeric features distributed without copyrighted source text.
        source = ROOT / "input" / f"{cfg.dataset_name}_document_features.csv"
        features = read_csv(source)
    length_labels = quartile_labels([float(row["document_chars"]) for row in features])
    table_labels = quartile_labels([float(row["table_char_fraction"]) for row in features])
    for row, length_label, table_label in zip(features, length_labels, table_labels):
        row["document_length_quartile"] = length_label
        row["table_density_quartile"] = table_label
    feature_map = {row["paper_id"]: row for row in features}
    write_csv(run_root / "document_features.csv", features)

    expected_evaluation = expected_evaluation_metadata(cfg)
    condition_data: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, float]]] = {}
    strata_rows: List[Dict[str, Any]] = []
    skipped_conditions: List[Dict[str, Any]] = []
    for job in condition_paths(run_root):
        key = (str(job["model_key"]), str(job["strategy"]), str(job["prompt_variant"]), f"r{int(job['seed'])}")
        run_dir = Path(job["run_dir"])
        current, reason, _ = evaluation_integrity(run_dir, expected_evaluation)
        if not current:
            skipped_conditions.append({"condition": "|".join(key), "run_dir": str(run_dir), "reason": reason})
            print(f"[strata-warn] skipped {run_dir}: {reason}")
            continue
        try:
            data = load_condition(job)
        except Exception as exc:
            skipped_conditions.append({"condition": "|".join(key), "run_dir": str(run_dir), "reason": repr(exc)})
            print(f"[strata-warn] skipped {run_dir}: {exc!r}")
            continue
        condition_data[key] = data
        for stratum_type, feature_key in [
            ("document_length_quartile", "document_length_quartile"),
            ("table_density_quartile", "table_density_quartile"),
        ]:
            for label in ["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]:
                ids = [pid for pid, feat in feature_map.items() if feat[feature_key] == label and pid in data]
                metrics = aggregate(data[pid] for pid in ids)
                strata_rows.append({
                    "model_key": key[0],
                    "strategy": key[1],
                    "prompt_variant": key[2],
                    "repeat": key[3],
                    "stratum_type": stratum_type,
                    "stratum": label,
                    **metrics,
                })
    write_csv(run_root / "stratified_condition_metrics.csv", strata_rows)

    lookup = {
        (row["model_key"], row["strategy"], row["prompt_variant"], row["repeat"], row["stratum_type"], row["stratum"]): row
        for row in strata_rows
    }
    diff_rows: List[Dict[str, Any]] = []
    models = sorted({key[0] for key in condition_data})
    prompts_repeats = sorted({(key[2], key[3]) for key in condition_data})
    for model in models:
        for prompt, repeat in prompts_repeats:
            for low, high, label in STRATEGY_PAIRS:
                for stratum_type in ["document_length_quartile", "table_density_quartile"]:
                    for stratum in ["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]:
                        lo = lookup.get((model, low, prompt, repeat, stratum_type, stratum))
                        hi = lookup.get((model, high, prompt, repeat, stratum_type, stratum))
                        if not lo or not hi:
                            continue
                        diff_rows.append({
                            "model_key": model,
                            "comparison": label,
                            "strategy_low": low,
                            "strategy_high": high,
                            "prompt_variant": prompt,
                            "repeat": repeat,
                            "stratum_type": stratum_type,
                            "stratum": stratum,
                            "paper_count": min(int(lo["paper_count"]), int(hi["paper_count"])),
                            "record_f1_difference": float(hi["record_f1"]) - float(lo["record_f1"]),
                            "field_micro_f1_difference": float(hi["field_micro_f1"]) - float(lo["field_micro_f1"]),
                            "null_accuracy_difference": float(hi["null_accuracy"]) - float(lo["null_accuracy"]),
                        })
    write_csv(run_root / "stratified_strategy_differences.csv", diff_rows)
    print(json.dumps({
        "run_root": str(run_root),
        "documents": len(features),
        "condition_strata_rows": len(strata_rows),
        "strategy_difference_rows": len(diff_rows),
        "skipped_conditions": skipped_conditions,
        "outputs": [
            str(run_root / "document_features.csv"),
            str(run_root / "stratified_condition_metrics.csv"),
            str(run_root / "stratified_strategy_differences.csv"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
