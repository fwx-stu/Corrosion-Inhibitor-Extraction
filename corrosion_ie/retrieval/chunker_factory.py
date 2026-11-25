from __future__ import annotations

"""Centralized document chunker selection.

The original project used ``CustomMarkdownSplitter`` for both proprietary LLM
RAG and local SLM RAG.  To keep model comparisons fair, that splitter is the
framework default.  The refactored structured splitter remains available only
through an explicit ``chunking_backend: v2_structured`` ablation.
"""

from typing import Any, Dict, List, Mapping

DEFAULT_CHUNKING_BACKEND = "legacy_exact"

_LEGACY_ALIASES = {
    "legacy_exact",
    "legacy",
    "old",
    "old_splitter",
    "old_q1_exact",
    "custom_markdown_splitter",
}
_STRUCTURED_ALIASES = {
    "v2",
    "v2_structured",
    "structured",
    "refactored",
    "new",
}


def normalize_chunking_backend(value: Any) -> str:
    name = str(value or DEFAULT_CHUNKING_BACKEND).strip().lower()
    if name in _LEGACY_ALIASES:
        return "legacy_exact"
    if name in _STRUCTURED_ALIASES:
        return "v2_structured"
    raise ValueError(
        f"Unsupported chunking_backend={value!r}. "
        "Use 'legacy_exact' (default) or 'v2_structured' (explicit ablation)."
    )


def resolve_chunking_backend(retrieval_cfg: Mapping[str, Any] | None = None) -> str:
    cfg = retrieval_cfg or {}
    # ``chunker`` is accepted for older local experiments, but the canonical
    # key written to resolved configs is ``chunking_backend``.
    value = cfg.get("chunking_backend", cfg.get("chunker", DEFAULT_CHUNKING_BACKEND))
    return normalize_chunking_backend(value)


def split_document(
    text: str,
    paper_id: str,
    retrieval_cfg: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    cfg = retrieval_cfg or {}
    chunk_size = int(cfg.get("chunk_size", 1000))
    chunk_overlap = int(cfg.get("chunk_overlap", 50))
    backend = resolve_chunking_backend(cfg)

    if backend == "legacy_exact":
        from .legacy_exact_chunking import split_markdown_legacy_exact

        chunks = split_markdown_legacy_exact(
            text,
            paper_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        from .chunking import split_markdown

        chunks = split_markdown(
            text,
            paper_id=paper_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    for chunk in chunks:
        chunk.setdefault("chunking_backend", backend)
        metadata = chunk.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("chunking_backend", backend)
    return chunks
