#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.retrieval.chunker_factory import resolve_chunking_backend, split_document


def signature(chunks) -> str:
    payload = json.dumps(
        [
            {
                "chunk_id": c.get("chunk_id"),
                "content_type": c.get("content_type"),
                "section_title": c.get("section_title"),
                "text": c.get("text"),
            }
            for c in chunks
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print the effective retrieval protocol and hash document chunks."
    )
    ap.add_argument("--config", default="configs/experiment_q0q1q2_dev37_balanced_opt.yaml")
    ap.add_argument("--paper-id", default="xml_6028")
    args = ap.parse_args()

    cfg = load_config(args.config)
    retrieval = cfg.retrieval
    md = cfg.md_dir / f"{args.paper_id}.md"
    if not md.exists():
        raise SystemExit(f"missing markdown: {md}")
    chunks = split_document(
        md.read_text(encoding="utf-8", errors="ignore"),
        paper_id=args.paper_id,
        retrieval_cfg=retrieval,
    )
    print(f"config             : {cfg.config_path}")
    print(f"retrieval_backend  : {retrieval.get('backend')}")
    print(f"chunking_backend   : {resolve_chunking_backend(retrieval)}")
    print(f"scoring_protocol   : {retrieval.get('scoring_protocol')}")
    print(f"strict_loading     : {retrieval.get('strict_loading')}")
    print(f"paper_id           : {args.paper_id}")
    print(f"chunk_count        : {len(chunks)}")
    print(f"table_chunk_count  : {sum(c.get('content_type') == 'table' for c in chunks)}")
    print(f"chunk_sha256       : {signature(chunks)}")
    if resolve_chunking_backend(retrieval) != "legacy_exact":
        raise SystemExit("[protocol-error] SLM/LLM default must use legacy_exact chunking")


if __name__ == "__main__":
    main()
