from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from ..schemas import EvidenceUnit

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u0394\u03b4][a-zA-Z0-9_°]*|%|℃")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def score_text(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    t = tokenize(text)
    if not t:
        return 0.0
    tc = Counter(t)
    score = 0.0
    for term in q:
        if term in tc:
            score += 1.0 + math.log(1 + tc[term])
    # Domain boost for numeric tables / corrosion units.
    lower = text.lower()
    if "%" in text and any(x in query.lower() for x in ["efficiency", "inhibition", "ie"]):
        score += 1.5
    if any(x in lower for x in ["homo", "lumo", "ehomo", "elumo"]):
        score += 1.0 if "homo" in query.lower() or "lumo" in query.lower() else 0.0
    if any(x in lower for x in ["gads", "δg", "Δg", "adsorption"]):
        score += 1.0 if "adsorption" in query.lower() or "gibbs" in query.lower() else 0.0
    return score / max(1.0, math.sqrt(len(t)))


def retrieve_units(
    paper_id: str,
    units: List[Dict[str, Any]],
    field_groups: Dict[str, Any],
    include_groups: List[str],
    top_k_per_group: int = 8,
    max_chars: int = 32000,
) -> List[EvidenceUnit]:
    out: List[EvidenceUnit] = []
    seen = set()
    total_chars = 0
    for group in include_groups:
        queries = field_groups.get(group, {}).get("queries", [])
        group_hits: List[EvidenceUnit] = []
        for query in queries:
            scored = []
            for u in units:
                text = u.get("text", "")
                score = score_text(query, text)
                if score <= 0:
                    continue
                scored.append((score, u))
            for rank, (score, u) in enumerate(sorted(scored, key=lambda x: x[0], reverse=True)[:top_k_per_group], 1):
                unit_id = u.get("unit_id") or u.get("chunk_id")
                ev = EvidenceUnit(
                    paper_id=paper_id,
                    unit_id=unit_id,
                    unit_type=u.get("unit_type", "chunk"),
                    text=u.get("text", ""),
                    query_group=group,
                    query=query,
                    rank=rank,
                    score=float(score),
                    section=u.get("section", ""),
                    caption=u.get("caption", ""),
                    columns=u.get("columns", {}) or {},
                    metadata={k: v for k, v in u.items() if k not in {"text", "columns"}},
                )
                group_hits.append(ev)
        group_hits.sort(key=lambda x: x.score, reverse=True)
        for ev in group_hits:
            if ev.unit_id in seen:
                continue
            if max_chars > 0 and total_chars + len(ev.text) > max_chars:
                continue
            out.append(ev)
            seen.add(ev.unit_id)
            total_chars += len(ev.text)
    return out


def _legacy_source_label(ev: EvidenceUnit) -> str:
    meta = ev.metadata or {}
    unit_type = str(ev.unit_type or "").lower()
    content_type = str(meta.get("content_type") or meta.get("chunk_type") or "").lower()
    if unit_type in {"table", "table_row", "row"} or content_type == "table" or ev.columns:
        return "table"
    return "text"



def rich_evidence_block(ev: EvidenceUnit) -> str:
    """Serialize one rich evidence unit without a trailing separator."""
    header = f"[unit_id={ev.unit_id}; type={ev.unit_type}; group={ev.query_group}; score={ev.score:.3f}]"
    if ev.caption:
        header += f"\ncaption: {ev.caption}"
    # Table-row ``text`` already contains a compact ``row:`` representation.
    # Avoid repeating the same columns a second time in the header.
    if ev.columns and ev.unit_type != "table_row":
        cols = "; ".join(f"{k}: {v}" for k, v in ev.columns.items())
        header += f"\ncolumns: {cols}"
    return header + "\n" + ev.text.strip()


def rich_evidence_block_chars(ev: EvidenceUnit, *, include_separator: bool = True) -> int:
    size = len(rich_evidence_block(ev))
    if include_separator:
        size += len("\n\n---\n\n")
    return size

def evidence_to_text(
    evidence: List[EvidenceUnit],
    max_chars: int | None = None,
    *,
    paper_id: str | None = None,
    format: str = "rich",
) -> str:
    """Serialize evidence without cutting a chunk in the middle.

    ``format="legacy"`` reproduces the old Q1 evidence envelope:
    ``[Target Paper ID]`` followed by numbered ``[Evidence ...]`` blocks.
    The default ``rich`` format retains v2 retrieval metadata for debugging.

    A non-positive ``max_chars`` means unlimited.  When a positive limit is
    used, only complete evidence blocks are included; partial table/text chunks
    are never emitted.
    """
    fmt = str(format or "rich").strip().lower()
    limit = int(max_chars) if max_chars is not None else 0
    parts: List[str] = []
    total = 0

    if fmt in {"legacy", "legacy_q1", "old_q1"}:
        pid = paper_id or (evidence[0].paper_id if evidence else "")
        header = f"[Target Paper ID]: {pid}\n"
        parts.append(header)
        total = len(header)
        for idx, ev in enumerate(evidence, start=1):
            source = _legacy_source_label(ev)
            block = (
                f"[Evidence {idx} | source: {source} | chunk_id: {ev.unit_id}]\n"
                f"{ev.text.strip()}"
            )
            separator_len = 2 if parts else 0
            if limit > 0 and total + separator_len + len(block) > limit:
                break
            parts.append(block)
            total += separator_len + len(block)
        return "\n\n".join(parts)

    for ev in evidence:
        block = rich_evidence_block(ev)
        separator_len = len("\n\n---\n\n") if parts else 0
        if limit > 0 and total + separator_len + len(block) > limit:
            # Skip an oversized unit rather than terminating the whole pack.
            # This preserves later evidence sources when an early chunk is long.
            continue
        parts.append(block)
        total += separator_len + len(block)
    return "\n\n---\n\n".join(parts)
