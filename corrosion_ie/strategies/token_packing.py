from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..llm.clients import LLMClient
from ..schemas import EvidenceUnit
from .common import PromptEnvelope
from .table_normalization import normalize_and_sanitize_tables, table_to_grid


_TARGET_GROUPS = ("conditions", "performance", "quantum", "adsorption")
_TABLE_MARKER_RE = re.compile(r"<table\b|(?:^|\n)\s*\|[^\n]+\|", re.I)
_CONDITION_RE = re.compile(
    r"(?:experimental\s+conditions?|materials?\s+and\s+methods?|weight\s+loss\s+(?:method|measurement)|"
    r"electrochemical\s+(?:measurements?|tests?)|corrosive\s+(?:medium|solution)|immersion\s+time|"
    r"carbon\s+steel|mild\s+steel|acid\s+solution|\bHCl\b|\bH2SO4\b|\bNaCl\b)",
    re.I,
)
_PERFORMANCE_RE = re.compile(
    r"(?:inhibition\s+efficien|\bIE\s*%|\bIE\b|[ηƞ]\s*%|weight\s+loss|potentiodynamic|polarization|"
    r"\bEIS\b|impedance|\bRct\b|\bIcorr\b|corrosion\s+rate)",
    re.I,
)
_QUANTUM_RE = re.compile(r"(?:\bHOMO\b|\bLUMO\b|E[_\- ]?HOMO|E[_\- ]?LUMO|quantum\s+chemical|\bDFT\b)", re.I)
_ADSORPTION_RE = re.compile(
    r"(?:Δ\s*G\s*[_\- ]?ads|delta\s+g\s*[_\- ]?ads|\bGads\b|adsorption\s+(?:free\s+)?energy|"
    r"\bE\s*[_\- ]?ads\b|\bEads\b|Monte\s+Carlo|molecular\s+dynamics)",
    re.I,
)
_TARGET_COLUMN_RE = re.compile(
    r"(?:inhibitor|compound|name|substrate|metal|medium|acid|solution|concentration|conc\.?|dosage|"
    r"temperature|temp\.?|immersion|time|method|efficien|\bIE\b|[ηƞ]|HOMO|LUMO|Gads|Eads|adsorption)",
    re.I,
)
_LOW_VALUE_TABLE_RE = re.compile(
    r"(?:\bEDS\b|elemental\s+composition|\bXPS\b|peak\s+assignments?|\bFTIR\b|IR\s+band|"
    r"synthesis\s+characteri[sz]ation|NMR\s+data|mass\s+spectrum|molecular\s+structures?|"
    r"chemical\s+properties|melting\s+point|yield\s*%?)",
    re.I,
)


@dataclass
class PackedEvidence:
    evidence: List[EvidenceUnit]
    text: str
    messages: Tuple[str, str]
    diagnostics: Dict[str, Any]


def is_window_unit(ev: EvidenceUnit) -> bool:
    return str(ev.query_group or "").lower() == "window" or "window expansion" in str(ev.query or "").lower()


def is_table_unit(ev: EvidenceUnit) -> bool:
    meta = ev.metadata or {}
    unit_type = str(ev.unit_type or "").lower()
    content_type = str(meta.get("content_type") or meta.get("chunk_type") or "").lower()
    return bool(
        unit_type in {"table", "table_row", "row"}
        or content_type == "table"
        or ev.columns
        or _TABLE_MARKER_RE.search(ev.text or "")
    )


def _table_semantic_context(ev: EvidenceUnit) -> str:
    """Table classification must depend only on source evidence semantics.

    Retrieval query text and query_group are intentionally excluded: otherwise
    an EDS/XPS table retrieved by a performance/quantum query can incorrectly
    classify itself as target evidence.
    """

    meta = ev.metadata or {}
    headers = meta.get("table_headers") or []
    header_text = " | ".join(str(x or "") for x in headers) if isinstance(headers, list) else str(headers or "")
    return "\n".join(str(x or "") for x in [ev.caption, ev.section, header_text, (ev.text or "")[:12000]])


def _non_table_context(ev: EvidenceUnit) -> str:
    return "\n".join(str(x or "") for x in [ev.section, ev.query_group, ev.query, (ev.text or "")[:8000]])


def _evidence_context(ev: EvidenceUnit) -> str:
    return _table_semantic_context(ev) if is_table_unit(ev) else _non_table_context(ev)


def is_low_value_table(ev: EvidenceUnit) -> bool:
    if not is_table_unit(ev):
        return False
    text = _table_semantic_context(ev)
    explicit_target_signal = bool(
        _PERFORMANCE_RE.search(text) or _QUANTUM_RE.search(text) or _ADSORPTION_RE.search(text)
    )
    return bool(_LOW_VALUE_TABLE_RE.search(text) and not explicit_target_signal)


def infer_target_group(ev: EvidenceUnit) -> str | None:
    if is_table_unit(ev):
        text = _table_semantic_context(ev)
        # The actual table caption/header/body is authoritative.  Retrieval
        # labels are provenance only and cannot promote a low-value table.
        if _QUANTUM_RE.search(text):
            return "quantum"
        if _ADSORPTION_RE.search(text):
            return "adsorption"
        if _PERFORMANCE_RE.search(text):
            return "performance"
        if _CONDITION_RE.search(text):
            return "conditions"
        return None

    explicit = str(ev.query_group or "").lower().strip()
    if explicit in _TARGET_GROUPS:
        return explicit
    text = _non_table_context(ev)
    if _QUANTUM_RE.search(text):
        return "quantum"
    if _ADSORPTION_RE.search(text):
        return "adsorption"
    if _PERFORMANCE_RE.search(text):
        return "performance"
    if _CONDITION_RE.search(text):
        return "conditions"
    return None


def is_condition_anchor(ev: EvidenceUnit) -> bool:
    if is_table_unit(ev):
        return False
    return infer_target_group(ev) == "conditions" or bool(_CONDITION_RE.search(_evidence_context(ev)))


def _legacy_source_label(ev: EvidenceUnit) -> str:
    return "table" if is_table_unit(ev) else "text"


def _document_index(ev: EvidenceUnit) -> int:
    meta = ev.metadata or {}
    for key in ("chunk_index_in_doc", "chunk_index", "document_index", "source_index"):
        try:
            if meta.get(key) is not None:
                return int(meta.get(key))
        except Exception:
            pass
    return 10**9


def _selection_score(ev: EvidenceUnit) -> float:
    meta = ev.metadata or {}
    for key in ("selection_score", "fused_rerank_score", "rerank_score", "retrieval_score"):
        try:
            if meta.get(key) is not None:
                return float(meta.get(key))
        except Exception:
            pass
    return float(ev.score or 0.0)


def serialize_legacy_evidence(paper_id: str, evidence: Sequence[EvidenceUnit]) -> str:
    """Serialize selected evidence with structural context, not model scores."""
    parts = [f"[Target Paper ID]: {paper_id}\n"]
    for idx, ev in enumerate(evidence, start=1):
        header = (
            f"[Evidence {idx} | source: {_legacy_source_label(ev)} | "
            f"group: {infer_target_group(ev) or ev.query_group or 'other'} | chunk_id: {ev.unit_id}]"
        )
        context: List[str] = [header]
        if str(ev.section or "").strip():
            context.append(f"[Section] {str(ev.section).strip()}")
        caption = str(ev.caption or "").strip()
        if caption and caption.lower() not in str(ev.text or "")[:500].lower():
            context.append(f"[Caption] {caption}")
        context.append((ev.text or "").strip())
        parts.append("\n".join(context))
    return "\n\n".join(parts)

def _clone_fragment(
    ev: EvidenceUnit,
    text: str,
    part: int,
    total: int,
    *,
    extra_metadata: Dict[str, Any] | None = None,
) -> EvidenceUnit:
    metadata = dict(ev.metadata or {})
    metadata.update({
        "parent_unit_id": ev.unit_id,
        "table_fragment_index": part,
        "table_fragment_count": total,
        "table_fragmented": total > 1,
        **(extra_metadata or {}),
    })
    return EvidenceUnit(
        paper_id=ev.paper_id,
        unit_id=f"{ev.unit_id}::table_part_{part:03d}" if total > 1 else ev.unit_id,
        unit_type="table",
        text=text,
        query_group=ev.query_group,
        query=ev.query,
        rank=ev.rank,
        score=ev.score,
        section=ev.section,
        caption=ev.caption,
        columns=dict(ev.columns or {}),
        metadata=metadata,
    )


def _render_markdown_fragment(caption: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def esc(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|")

    parts: List[str] = []
    if caption:
        parts.append(f"caption: {esc(caption)}")
    parts.append("| " + " | ".join(esc(x) for x in headers) + " |")
    parts.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        padded = list(row[: len(headers)]) + [""] * max(0, len(headers) - len(row))
        parts.append("| " + " | ".join(esc(x) for x in padded) + " |")
    return "\n".join(parts)


def _compact_oversized_row(
    caption: str,
    headers: Sequence[str],
    row: Sequence[str],
    *,
    client: LLMClient,
    max_fragment_tokens: int,
) -> Tuple[List[str], bool, bool]:
    """Compact only non-target descriptive cells.

    Target columns (inhibitor, concentrations, temperature, method, efficiency,
    quantum and adsorption fields) are never silently shortened.
    """
    cells = [str(x or "") for x in row]
    rendered = _render_markdown_fragment(caption, headers, [cells])
    if client.count_text_tokens(rendered) <= max_fragment_tokens:
        return cells, False, False
    protected = {i for i, header in enumerate(headers) if _TARGET_COLUMN_RE.search(str(header or ""))}
    compacted = False
    for _ in range(100):
        candidates = [i for i, value in enumerate(cells) if i not in protected and len(value) > 48]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: len(cells[i]))
        value = cells[idx]
        keep = max(18, int(len(value) * 0.70))
        left = max(8, keep // 2)
        right = max(8, keep - left)
        cells[idx] = value[:left] + " …[non-target cell compacted]… " + value[-right:]
        compacted = True
        rendered = _render_markdown_fragment(caption, headers, [cells])
        if client.count_text_tokens(rendered) <= max_fragment_tokens:
            return cells, compacted, False
    return cells, compacted, client.count_text_tokens(rendered) > max_fragment_tokens


def fragment_table_unit(
    ev: EvidenceUnit,
    *,
    client: LLMClient,
    max_fragment_tokens: int = 3500,
) -> List[EvidenceUnit]:
    """Normalize a table, expand rowspan/colspan, and split only at logical rows."""

    if not is_table_unit(ev):
        return [ev]
    grid = table_to_grid(ev.text or "", caption=ev.caption or "")
    if grid is None:
        return [ev]
    normalized_text = grid.render_markdown()
    normalized = EvidenceUnit(
        paper_id=ev.paper_id,
        unit_id=ev.unit_id,
        unit_type="table",
        text=normalized_text,
        query_group=ev.query_group,
        query=ev.query,
        rank=ev.rank,
        score=ev.score,
        section=ev.section,
        caption=grid.caption or ev.caption,
        columns=dict(ev.columns or {}),
        metadata={
            **dict(ev.metadata or {}),
            "table_normalized": True,
            "table_headers": list(grid.headers),
            "table_row_count": len(grid.rows),
            **grid.diagnostics,
        },
    )
    if client.count_text_tokens(normalized_text) <= max_fragment_tokens:
        return [normalized]

    groups: List[List[List[str]]] = []
    current: List[List[str]] = []
    compacted_rows = 0
    protected_overflow_rows = 0
    for original_row in grid.rows:
        row, compacted, protected_overflow = _compact_oversized_row(
            grid.caption,
            grid.headers,
            original_row,
            client=client,
            max_fragment_tokens=max_fragment_tokens,
        )
        compacted_rows += int(compacted)
        protected_overflow_rows += int(protected_overflow)
        candidate = current + [row]
        candidate_text = _render_markdown_fragment(grid.caption, grid.headers, candidate)
        if current and client.count_text_tokens(candidate_text) > max_fragment_tokens:
            groups.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        groups.append(current)

    fragments: List[EvidenceUnit] = []
    total = len(groups)
    for idx, rows in enumerate(groups, start=1):
        text = _render_markdown_fragment(grid.caption, grid.headers, rows)
        fragments.append(_clone_fragment(
            normalized,
            text,
            idx,
            total,
            extra_metadata={
                "oversized_single_rows_compacted": compacted_rows,
                "protected_target_rows_over_fragment_limit": protected_overflow_rows,
                "fragment_row_count": len(rows),
            },
        ))
    return fragments or [normalized]


def _stable_unique(items: Iterable[EvidenceUnit]) -> List[EvidenceUnit]:
    out: List[EvidenceUnit] = []
    seen: set[str] = set()
    for ev in items:
        if ev.unit_id in seen:
            continue
        seen.add(ev.unit_id)
        out.append(ev)
    return out


def _score_key(ev: EvidenceUnit) -> Tuple[float, int, int, str]:
    return (-_selection_score(ev), int(ev.rank or 10**9), _document_index(ev), ev.unit_id)


def limit_windows(evidence: Sequence[EvidenceUnit], max_window_chunks: int) -> Tuple[List[EvidenceUnit], int]:
    non_windows = [ev for ev in evidence if not is_window_unit(ev)]
    windows = sorted([ev for ev in evidence if is_window_unit(ev)], key=_score_key)
    cap = max(0, int(max_window_chunks))
    return non_windows + windows[:cap], max(0, len(windows) - cap)


def order_budgeted_rag(evidence: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
    direct = sorted(
        [ev for ev in evidence if not is_window_unit(ev) and bool((ev.metadata or {}).get("direct_hit", True))],
        key=_score_key,
    )
    extras = sorted(
        [ev for ev in evidence if not is_window_unit(ev) and not bool((ev.metadata or {}).get("direct_hit", True))],
        key=_score_key,
    )
    windows = sorted(
        [ev for ev in evidence if is_window_unit(ev)],
        key=lambda ev: (
            int((ev.metadata or {}).get("parent_rerank_rank", 10**9)),
            int((ev.metadata or {}).get("window_distance", 10**9)),
            -_selection_score(ev),
            _document_index(ev),
        ),
    )
    return _stable_unique(direct + extras + windows)


def present_budgeted_rag(evidence: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
    """After relevance selection, restore local document order for readability."""
    return sorted(evidence, key=lambda ev: (_document_index(ev), int(ev.rank or 10**9), ev.unit_id))


def present_table_aware_rag(
    evidence: Sequence[EvidenceUnit],
    *,
    condition_anchors_first: bool = True,
) -> List[EvidenceUnit]:
    """Present table-priority evidence without silently enabling anchors.

    Q2 uses ``condition_anchors_first=True``.  The table-only reviewer
    ablation disables that component while retaining table normalization,
    fragmentation, schema-group balancing, and table-priority presentation.
    """
    group_order = {"performance": 0, "quantum": 1, "adsorption": 2}
    anchors = sorted([ev for ev in evidence if is_condition_anchor(ev)], key=lambda ev: (_document_index(ev), _score_key(ev)))
    tables = sorted(
        [ev for ev in evidence if is_table_unit(ev) and infer_target_group(ev) in group_order],
        key=lambda ev: (group_order[infer_target_group(ev) or "performance"], _document_index(ev), _score_key(ev)),
    )
    direct_text = sorted(
        [ev for ev in evidence if not is_table_unit(ev) and not is_window_unit(ev) and ev not in anchors],
        key=lambda ev: (_document_index(ev), _score_key(ev)),
    )
    windows = sorted([ev for ev in evidence if is_window_unit(ev)], key=lambda ev: (_document_index(ev), _score_key(ev)))
    other_tables = sorted(
        [ev for ev in evidence if is_table_unit(ev) and infer_target_group(ev) not in group_order],
        key=lambda ev: (_document_index(ev), _score_key(ev)),
    )
    if condition_anchors_first:
        return _stable_unique(anchors + tables + direct_text + windows + other_tables)
    return _stable_unique(tables + direct_text + anchors + windows + other_tables)


def present_condition_anchor_rag(evidence: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
    """Present reserved condition anchors first and otherwise preserve order."""

    anchors = sorted(
        [ev for ev in evidence if is_condition_anchor(ev)],
        key=lambda ev: (_document_index(ev), _score_key(ev)),
    )
    remainder = sorted(
        [ev for ev in evidence if ev not in anchors],
        key=lambda ev: (_document_index(ev), int(ev.rank or 10**9), ev.unit_id),
    )
    return _stable_unique(anchors + remainder)


def _prepare_table_aware_candidates(
    evidence: Sequence[EvidenceUnit],
    *,
    client: LLMClient,
    max_table_fragment_tokens: int,
    allow_reference_tables: bool = False,
) -> Tuple[List[EvidenceUnit], Dict[str, Any]]:
    prepared: List[EvidenceUnit] = []
    diag: Dict[str, Any] = {
        "table_candidates": 0,
        "reference_tables_excluded": 0,
        "mixed_reference_tables_sanitized": 0,
        "review_reference_tables_kept": 0,
        "low_value_tables_excluded": 0,
        "table_normalization_failures": 0,
        "table_fragments_added": 0,
        "rowspan_cells_expanded": 0,
        "colspan_cells_expanded": 0,
        "oversized_single_rows_compacted": 0,
        "protected_target_rows_over_fragment_limit": 0,
    }
    for ev in evidence:
        if not is_table_unit(ev):
            prepared.append(ev)
            continue
        diag["table_candidates"] += 1
        normalized_items, ref_info = normalize_and_sanitize_tables(
            ev, allow_reference_tables=allow_reference_tables
        )
        logical_infos = list(ref_info.get("logical_table_infos") or [ref_info])
        diag["logical_subtables_detected"] = diag.get("logical_subtables_detected", 0) + int(
            ref_info.get("logical_table_count", len(normalized_items) or 1)
        )
        diag["logical_subtables_excluded"] = diag.get("logical_subtables_excluded", 0) + int(
            ref_info.get("logical_tables_excluded", 0)
        )
        diag["reference_tables_excluded"] += sum(
            1 for info in logical_infos if str(info.get("action", "")).startswith("exclude_")
        )
        diag["mixed_reference_tables_sanitized"] += sum(
            1 for info in logical_infos if info.get("action") == "keep_current_study_rows_only"
        )
        diag["review_reference_tables_kept"] += sum(
            1 for info in logical_infos if info.get("action") == "keep_review_reference_table"
        )
        if not normalized_items:
            continue
        for normalized in normalized_items:
            if not (normalized.metadata or {}).get("table_normalized"):
                diag["table_normalization_failures"] += 1
            if is_low_value_table(normalized):
                diag["low_value_tables_excluded"] += 1
                continue
            pieces = fragment_table_unit(
                normalized,
                client=client,
                max_fragment_tokens=max(256, int(max_table_fragment_tokens)),
            )
            diag["table_fragments_added"] += max(0, len(pieces) - 1)
            for piece in pieces:
                meta = piece.metadata or {}
                diag["rowspan_cells_expanded"] += int(meta.get("rowspan_count", 0))
                diag["colspan_cells_expanded"] += int(meta.get("colspan_count", 0))
                diag["oversized_single_rows_compacted"] += int(meta.get("oversized_single_rows_compacted", 0))
                diag["protected_target_rows_over_fragment_limit"] += int(meta.get("protected_target_rows_over_fragment_limit", 0))
            prepared.extend(pieces)
    return prepared, diag


def _round_robin_table_fragments(items: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
    """Expose one fragment from each source table before later fragments.

    This prevents a very large weight-loss table from occupying all final slots
    before PDP/EIS/quantum/adsorption tables are considered.
    """
    buckets: Dict[str, List[EvidenceUnit]] = {}
    parent_order: List[str] = []
    for ev in sorted(items, key=_score_key):
        parent = str((ev.metadata or {}).get("parent_unit_id") or ev.unit_id)
        if parent not in buckets:
            buckets[parent] = []
            parent_order.append(parent)
        buckets[parent].append(ev)
    out: List[EvidenceUnit] = []
    depth = 0
    while True:
        added = False
        for parent in parent_order:
            bucket = buckets[parent]
            if depth < len(bucket):
                out.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return out


def _table_aware_categories(
    evidence: Sequence[EvidenceUnit],
) -> Tuple[List[EvidenceUnit], Dict[str, List[EvidenceUnit]], List[EvidenceUnit], List[EvidenceUnit], List[EvidenceUnit]]:
    anchors = sorted([ev for ev in evidence if is_condition_anchor(ev)], key=_score_key)
    group_tables: Dict[str, List[EvidenceUnit]] = {
        group: _round_robin_table_fragments(
            [ev for ev in evidence if is_table_unit(ev) and infer_target_group(ev) == group]
        )
        for group in ("performance", "quantum", "adsorption")
    }
    direct_text = sorted(
        [ev for ev in evidence if not is_table_unit(ev) and not is_window_unit(ev) and ev not in anchors],
        key=_score_key,
    )
    windows = sorted([ev for ev in evidence if is_window_unit(ev)], key=_score_key)
    other_tables = sorted(
        [ev for ev in evidence if is_table_unit(ev) and all(ev not in values for values in group_tables.values())],
        key=_score_key,
    )
    return anchors, group_tables, direct_text, windows, other_tables



def _round_robin_target_groups(group_tables: Dict[str, List[EvidenceUnit]]) -> List[EvidenceUnit]:
    """Interleave performance, quantum and adsorption evidence by depth."""

    groups = ("performance", "quantum", "adsorption")
    out: List[EvidenceUnit] = []
    depth = 0
    while True:
        added = False
        for group in groups:
            items = group_tables.get(group, [])
            if depth < len(items):
                out.append(items[depth])
                added = True
        if not added:
            break
        depth += 1
    return out


def order_table_aware_rag(
    evidence: Sequence[EvidenceUnit],
    *,
    condition_anchor_quota: int = 4,
    enable_condition_anchors: bool = True,
) -> Tuple[List[EvidenceUnit], Dict[str, Any]]:
    anchors, group_tables, direct_text, windows, other_tables = _table_aware_categories(evidence)
    anchor_quota = max(0, int(condition_anchor_quota)) if enable_condition_anchors else 0
    reserved: List[EvidenceUnit] = anchors[:anchor_quota]
    for group in ("performance", "quantum", "adsorption"):
        candidates = group_tables[group] or [ev for ev in direct_text if infer_target_group(ev) == group]
        if candidates:
            reserved.append(candidates[0])
    relevant_tables = _round_robin_target_groups(group_tables)
    if enable_condition_anchors:
        supporting_text = anchors + direct_text
    else:
        # Without the anchor component, condition-like chunks return to the
        # ordinary relevance-ranked text pool instead of receiving priority.
        supporting_text = sorted(anchors + direct_text, key=_score_key)
    ordered = _stable_unique(reserved + relevant_tables + supporting_text + windows + other_tables)
    return ordered, {
        "condition_anchor_candidates": len(anchors),
        "condition_anchor_quota": anchor_quota,
        "condition_anchors_enabled": bool(enable_condition_anchors),
        "schema_relevant_table_candidates": len(relevant_tables),
        "cross_group_round_robin": True,
        "schema_relevant_source_tables": len({str((ev.metadata or {}).get("parent_unit_id") or ev.unit_id) for ev in relevant_tables}),
        "direct_text_candidates": len(direct_text),
        "window_candidates": len(windows),
        "available_groups": [
            group for group in ("conditions", "performance", "quantum", "adsorption")
            if (
                anchors if (group == "conditions" and enable_condition_anchors) else []
                if group == "conditions"
                else (group_tables.get(group) or [ev for ev in direct_text if infer_target_group(ev) == group])
            )
        ],
    }


def order_condition_anchor_rag(
    evidence: Sequence[EvidenceUnit],
    *,
    condition_anchor_quota: int = 4,
) -> Tuple[List[EvidenceUnit], Dict[str, Any]]:
    """Reserve condition evidence without enabling table-aware processing."""

    anchors = sorted([ev for ev in evidence if is_condition_anchor(ev)], key=_score_key)
    quota = max(0, int(condition_anchor_quota))
    reserved = anchors[:quota]
    ordered = _stable_unique(reserved + order_budgeted_rag(evidence))
    return ordered, {
        "condition_anchor_candidates": len(anchors),
        "condition_anchor_quota": quota,
        "condition_anchors_enabled": True,
        "available_groups": ["conditions"] if anchors and quota > 0 else [],
    }


def pack_evidence_for_prompt(
    *,
    paper_id: str,
    evidence: Sequence[EvidenceUnit],
    envelope: PromptEnvelope,
    client: LLMClient,
    max_input_tokens: int | None,
    max_items: int = 40,
    max_window_chunks: int = 8,
    safety_margin_tokens: int = 32,
    table_aware: bool = False,
    condition_aware: bool | None = None,
    max_table_fragment_tokens: int = 3500,
    condition_anchor_quota: int = 4,
    allow_reference_tables: bool = False,
) -> PackedEvidence:
    """Pack complete evidence blocks into the frozen chat budget.

    Q2 first guarantees one fitting candidate from each available schema group,
    then fills the remaining budget by the deterministic priority order.  An
    oversized candidate never blocks later, smaller evidence blocks.
    """

    # Backward compatibility is deliberate: existing Q2 calls specify only
    # ``table_aware=True`` and therefore retain their original anchor behavior;
    # existing Q1 calls retain both components disabled.
    condition_aware = bool(table_aware) if condition_aware is None else bool(condition_aware)

    limited, windows_dropped = limit_windows(evidence, max_window_chunks)
    prep_diag: Dict[str, Any] = {}
    if table_aware:
        prepared, prep_diag = _prepare_table_aware_candidates(
            limited,
            client=client,
            max_table_fragment_tokens=max_table_fragment_tokens,
            allow_reference_tables=allow_reference_tables,
        )
        ordered, ordering_diag = order_table_aware_rag(
            prepared,
            condition_anchor_quota=condition_anchor_quota,
            enable_condition_anchors=condition_aware,
        )
    elif condition_aware:
        prepared = list(limited)
        ordered, ordering_diag = order_condition_anchor_rag(
            prepared,
            condition_anchor_quota=condition_anchor_quota,
        )
    else:
        prepared = list(limited)
        ordered = order_budgeted_rag(prepared)
        ordering_diag = {}

    max_items = max(1, int(max_items))
    max_input = int(max_input_tokens or 0)
    hard_limit = max_input - max(0, int(safety_margin_tokens)) if max_input > 0 else 0
    empty_text = serialize_legacy_evidence(paper_id, [])
    empty_messages = envelope.render(empty_text)
    fixed_tokens = client.count_chat_tokens(*empty_messages)
    if hard_limit and fixed_tokens > hard_limit:
        raise ValueError(f"fixed RAG prompt requires {fixed_tokens} tokens, above safe limit {hard_limit}")

    selected: List[EvidenceUnit] = []
    selected_ids: set[str] = set()
    dropped_budget: List[str] = []
    dropped_item_cap: List[str] = []
    final_text = empty_text
    final_messages = empty_messages
    final_tokens = fixed_tokens

    def try_add(ev: EvidenceUnit) -> bool:
        nonlocal selected, final_text, final_messages, final_tokens
        if ev.unit_id in selected_ids:
            return True
        if len(selected) >= max_items:
            dropped_item_cap.append(ev.unit_id)
            return False
        candidate = selected + [ev]
        candidate_text = serialize_legacy_evidence(paper_id, candidate)
        candidate_messages = envelope.render(candidate_text)
        candidate_tokens = client.count_chat_tokens(*candidate_messages)
        if hard_limit and candidate_tokens > hard_limit:
            dropped_budget.append(ev.unit_id)
            return False
        selected = candidate
        selected_ids.add(ev.unit_id)
        final_text = candidate_text
        final_messages = candidate_messages
        final_tokens = candidate_tokens
        return True

    packed_groups: set[str] = set()
    group_coverage_failures: Dict[str, str] = {}
    if condition_aware:
        anchors, group_tables, direct_text, _, _ = _table_aware_categories(prepared)
        # Conditions may require several short anchors; every successful one is
        # useful, but only one is required for minimum group coverage.
        condition_candidates = anchors[: max(0, int(condition_anchor_quota))]
        if condition_candidates:
            for ev in condition_candidates:
                if try_add(ev):
                    packed_groups.add("conditions")
            if "conditions" not in packed_groups:
                group_coverage_failures["conditions"] = "all candidates exceeded item/token budget"

    if table_aware:
        anchors, group_tables, direct_text, _, _ = _table_aware_categories(prepared)
        for group in ("performance", "quantum", "adsorption"):
            candidates = group_tables[group] or [ev for ev in direct_text if infer_target_group(ev) == group]
            if not candidates:
                continue
            for ev in candidates:
                if try_add(ev):
                    packed_groups.add(group)
                    break
            if group not in packed_groups:
                group_coverage_failures[group] = "all candidates exceeded item/token budget"

    for ev in ordered:
        try_add(ev)

    selection_order_ids = [ev.unit_id for ev in selected]
    if table_aware:
        presented = present_table_aware_rag(
            selected,
            condition_anchors_first=condition_aware,
        )
    elif condition_aware:
        presented = present_condition_anchor_rag(selected)
    else:
        presented = present_budgeted_rag(selected)
    final_text = serialize_legacy_evidence(paper_id, presented)
    final_messages = envelope.render(final_text)
    final_tokens = client.count_chat_tokens(*final_messages)
    if hard_limit and final_tokens > hard_limit:
        raise ValueError(
            f"presentation reorder exceeded frozen chat budget: {final_tokens} > {hard_limit}"
        )
    selected = presented

    group_counts: Dict[str, int] = {}
    table_count = 0
    window_count = 0
    for ev in selected:
        group = infer_target_group(ev) or str(ev.query_group or "other")
        group_counts[group] = group_counts.get(group, 0) + 1
        table_count += int(is_table_unit(ev))
        window_count += int(is_window_unit(ev))
        if group in _TARGET_GROUPS:
            packed_groups.add(group)

    available_groups = (
        set(ordering_diag.get("available_groups", []))
        if (table_aware or condition_aware)
        else set()
    )
    if table_aware and condition_aware:
        packing_mode = "table_aware_select_then_present_v4"
    elif table_aware:
        packing_mode = "table_only_select_then_present_v1"
    elif condition_aware:
        packing_mode = "condition_anchor_select_then_present_v1"
    else:
        packing_mode = "relevance_select_then_present_v4"
    diagnostics: Dict[str, Any] = {
        "packing_mode": packing_mode,
        "table_aware_enabled": bool(table_aware),
        "condition_anchors_enabled": bool(condition_aware),
        # Backward-compatible diagnostic name used by existing QA scripts.
        "reference_table_candidates": int(prep_diag.get("reference_tables_excluded", 0)),
        "candidate_evidence_count": len(evidence),
        "candidate_after_window_cap": len(limited),
        "candidate_after_table_preparation": len(prepared),
        "packed_evidence_count": len(selected),
        "packed_table_count": table_count,
        "packed_window_count": window_count,
        "packed_group_counts": group_counts,
        "available_groups": sorted(available_groups),
        "packed_groups": sorted(packed_groups),
        "group_coverage_failures": group_coverage_failures,
        "group_coverage_complete": not bool(available_groups - packed_groups),
        "dropped_window_cap": windows_dropped,
        "dropped_for_token_budget": len(set(dropped_budget)),
        "dropped_for_item_cap": len(set(dropped_item_cap)),
        "dropped_for_token_budget_ids": list(dict.fromkeys(dropped_budget))[:100],
        "dropped_for_item_cap_ids": list(dict.fromkeys(dropped_item_cap))[:100],
        "max_final_chunks": max_items,
        "max_window_chunks": max_window_chunks,
        "fixed_prompt_tokens": fixed_tokens,
        "packed_evidence_tokens": client.count_text_tokens(final_text),
        "final_chat_tokens": final_tokens,
        "max_input_tokens": max_input_tokens,
        "safe_input_limit": hard_limit or None,
        "safety_margin_tokens": safety_margin_tokens,
        "input_truncated": False,
        "allow_reference_tables": bool(allow_reference_tables),
        "selection_order_unit_ids": selection_order_ids,
        "presentation_order_unit_ids": [ev.unit_id for ev in selected],
        "packed_evidence_unit_ids": [ev.unit_id for ev in selected],
        **prep_diag,
        **ordering_diag,
        **client.token_counter_metadata(),
    }
    return PackedEvidence(selected, final_text, final_messages, diagnostics)
