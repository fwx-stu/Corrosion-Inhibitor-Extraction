#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc


def collect(
    run_root: Path,
    *,
    models: list[str],
    strategies: list[str],
    prompt: str,
    seed: int,
    expected_papers: int,
) -> Dict[str, Any]:
    budgets: Dict[str, Dict[str, Dict[str, int]]] = {}
    sources: Dict[str, Dict[str, Dict[str, Any]]] = {}
    canonical_ids: set[str] | None = None
    for model in models:
        budgets[model] = {}
        sources[model] = {}
        for strategy in strategies:
            predictions = run_root / model / strategy / prompt / f"r{seed}" / "predictions.jsonl"
            if not predictions.is_file():
                raise FileNotFoundError(f"missing completed predictions: {predictions}")
            values: Dict[str, int] = {}
            for row in read_jsonl(predictions):
                paper_id = str(row.get("paper_id") or "").strip()
                packing = ((row.get("metadata") or {}).get("packing") or {})
                tokens = packing.get("final_chat_tokens")
                if not paper_id or tokens is None:
                    raise ValueError(
                        f"missing paper_id/final_chat_tokens in {predictions}: {paper_id or '<blank>'}"
                    )
                if paper_id in values:
                    raise ValueError(f"duplicate paper_id {paper_id} in {predictions}")
                values[paper_id] = int(tokens)
            if len(values) != expected_papers:
                raise ValueError(
                    f"expected {expected_papers} papers for {model}/{strategy}; found {len(values)}"
                )
            ids = set(values)
            if canonical_ids is None:
                canonical_ids = ids
            elif ids != canonical_ids:
                raise ValueError(
                    f"paper-ID mismatch for {model}/{strategy}: "
                    f"missing={sorted(canonical_ids - ids)}, extra={sorted(ids - canonical_ids)}"
                )
            budgets[model][strategy] = dict(sorted(values.items()))
            sources[model][strategy] = {
                "path": str(predictions),
                "sha256": sha256_file(predictions),
                "paper_count": len(values),
                "mean_final_chat_tokens": round(sum(values.values()) / max(1, len(values)), 4),
            }
    return {
        "schema": "token-matched-non-rag-budget-manifest-v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(run_root),
        "prompt_variant": prompt,
        "seed": int(seed),
        "expected_paper_count": expected_papers,
        "paper_ids": sorted(canonical_ids or []),
        "budgets": budgets,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze per-paper Q1/Q2 final-chat token counts for non-RAG controls."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="qwen2_5_3b,qwen2_5_7b,qwen2_5_14b")
    parser.add_argument("--strategies", default="q1_budgeted_rag,q2_table_aware_rag")
    parser.add_argument("--prompt", default="balanced")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected-papers", type=int, required=True)
    args = parser.parse_args()

    payload = collect(
        Path(args.run_root).expanduser().resolve(),
        models=[value.strip() for value in args.models.split(",") if value.strip()],
        strategies=[value.strip() for value in args.strategies.split(",") if value.strip()],
        prompt=args.prompt,
        seed=args.seed,
        expected_papers=args.expected_papers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "written": str(output),
        "paper_count": payload["expected_paper_count"],
        "models": sorted(payload["budgets"]),
        "strategies": sorted(next(iter(payload["budgets"].values()))),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
