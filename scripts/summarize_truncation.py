#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.io_utils import read_jsonl, write_csv


def generation_rows(metadata: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(metadata.get("generation"), dict):
        yield "single", metadata["generation"]
    if isinstance(metadata.get("stage_generation"), dict):
        for stage, row in metadata["stage_generation"].items():
            if isinstance(row, dict):
                yield str(stage), row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize input packing and output-length diagnostics.")
    ap.add_argument("--run-root", required=True, type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.run_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    detail_rows = []
    summary_rows = []
    for job in manifest.get("jobs", []):
        run_dir = Path(job["run_dir"])
        calls = []
        raw_path = run_dir / "raw_outputs.jsonl"
        if raw_path.exists():
            for obj in read_jsonl(raw_path):
                metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                packing = metadata.get("packing") if isinstance(metadata.get("packing"), dict) else {}
                for stage, row in generation_rows(metadata):
                    detail = {
                        "model_key": job["model_key"],
                        "strategy": job["strategy"],
                        "prompt_variant": job["prompt_variant"],
                        "paper_id": obj.get("paper_id"),
                        "stage": stage,
                        "input_tokens": row.get("input_tokens_after_truncation"),
                        "max_input_tokens": row.get("max_input_tokens"),
                        "input_truncated": row.get("truncated"),
                        "input_over_budget": row.get("input_over_budget"),
                        "generated_tokens": row.get("generated_tokens"),
                        "max_new_tokens": row.get("max_new_tokens"),
                        "finish_reason": row.get("finish_reason"),
                        "output_truncated": bool(row.get("reached_max_new_tokens")) or row.get("finish_reason") == "length",
                        "json_closed_by_outer_delimiter": row.get("json_closed_by_outer_delimiter"),
                        "json_repeat_guard_triggered": row.get("json_repeat_guard_triggered"),
                        "json_repeat_guard_cycle_length": row.get("json_repeat_guard_cycle_length"),
                        "json_repeat_guard_rows_before_trim": row.get("json_repeat_guard_rows_before_trim"),
                        "json_repeat_guard_rows_after_trim": row.get("json_repeat_guard_rows_after_trim"),
                        "packed_evidence_count": packing.get("packed_evidence_count"),
                        "packed_table_count": packing.get("packed_table_count"),
                        "dropped_for_token_budget": packing.get("dropped_for_token_budget"),
                        "final_chat_tokens_from_packer": packing.get("final_chat_tokens"),
                        "parse_error": bool(obj.get("parse_error")),
                    }
                    detail_rows.append(detail)
                    calls.append(detail)
        known = [x for x in calls if isinstance(x.get("input_truncated"), bool)]
        input_values = [int(x["input_tokens"]) for x in calls if isinstance(x.get("input_tokens"), (int, float))]
        output_values = [int(x["generated_tokens"]) for x in calls if isinstance(x.get("generated_tokens"), (int, float))]
        summary_rows.append({
            "model_key": job["model_key"],
            "strategy": job["strategy"],
            "prompt_variant": job["prompt_variant"],
            "papers": len({x.get("paper_id") for x in calls if x.get("paper_id")}),
            "generation_calls": len(calls),
            "input_truncated_calls": sum(bool(x.get("input_truncated")) for x in known),
            "input_over_budget_calls": sum(bool(x.get("input_over_budget")) for x in calls),
            "output_truncated_calls": sum(bool(x.get("output_truncated")) for x in calls),
            "parse_error_calls": sum(bool(x.get("parse_error")) for x in calls),
            "repeat_guard_calls": sum(bool(x.get("json_repeat_guard_triggered")) for x in calls),
            "max_input_tokens": max(input_values) if input_values else "",
            "mean_input_tokens": sum(input_values) / max(1, len(input_values)) if input_values else "",
            "max_generated_tokens": max(output_values) if output_values else "",
            "mean_generated_tokens": sum(output_values) / max(1, len(output_values)) if output_values else "",
        })
    write_csv(args.run_root / "generation_diagnostics.csv", detail_rows)
    write_csv(args.run_root / "generation_summary.csv", summary_rows)
    print(f"[generation-diagnostics] {args.run_root / 'generation_summary.csv'}")


if __name__ == "__main__":
    main()
