from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os
import re

from ..io_utils import read_yaml_or_json
from ..llm.clients import GenerationConfig, LLMClient
from ..postprocess.json_parser import parse_llm_json
from ..postprocess.grounding_filter import filter_records_with_report
from ..postprocess.normalizer import dedup_records, normalize_records, split_obvious_multivalue_records
from ..prompts import load_prompt, render_prompt
from ..retrieval.chunker_factory import split_document
from ..retrieval.local import retrieve_units
from ..retrieval.legacy_hybrid import build_legacy_high_recall_evidence
from ..retrieval.table_rows import extract_markdown_tables
from ..schemas import EvidenceUnit
from ..debug.trace import DebugTrace


_CONTENT_SENTINEL = "<<<CORROSION_IE_CONTENT_SLOT_7F4A2C>>>"


@dataclass(frozen=True)
class PromptEnvelope:
    """Rendered prompt with a protected content slot in exactly one chat message."""

    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    content_location: str
    template_name: str
    policy_name: str

    def render(self, content: str) -> Tuple[str, str]:
        if self.content_location == "system":
            return self.system_prefix + content + self.system_suffix, self.user_prefix + self.user_suffix
        return self.system_prefix + self.system_suffix, self.user_prefix + content + self.user_suffix


def load_fields_config(config_path: Path) -> Dict[str, Any]:
    return read_yaml_or_json(config_path.parent / "fields.yaml")


def _policy_parts(prompts_dir: Path, policy_name: str) -> Tuple[str, str]:
    policy_path = prompts_dir / "policies" / f"{policy_name}.txt"
    raw_policy = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    separator = "\n<<<FINAL_CHECK>>>\n"
    if separator in raw_policy:
        return tuple(raw_policy.split(separator, 1))  # type: ignore[return-value]
    return raw_policy, ""


def prepare_prompt_envelope(
    *,
    paper_id: str,
    prompts_dir: Path,
    prompt_variant: str,
    prompt_template: str | None = None,
    policy_variant_override: str | None = None,
) -> PromptEnvelope:
    """Render all fixed instructions while leaving the article/evidence slot protected."""

    template_name = prompt_template or prompt_variant
    template = load_prompt(template_name, prompts_dir)
    policy_name = policy_variant_override or prompt_variant
    policy_text, policy_final_check = _policy_parts(prompts_dir, policy_name)
    msg = render_prompt(
        template,
        paper_id=paper_id,
        evidence=_CONTENT_SENTINEL,
        content=_CONTENT_SENTINEL,
        target_article=_CONTENT_SENTINEL,
        policy_instructions=policy_text,
        policy_final_check=policy_final_check,
    )
    system = msg.get("system", "")
    user = msg.get("user", "")
    occurrences = system.count(_CONTENT_SENTINEL) + user.count(_CONTENT_SENTINEL)
    if occurrences != 1:
        raise ValueError(
            f"prompt template {template_name!r} must contain exactly one content placeholder; "
            f"rendered occurrences={occurrences}"
        )
    if _CONTENT_SENTINEL in system:
        before, after = system.split(_CONTENT_SENTINEL, 1)
        return PromptEnvelope(before, after, user, "", "system", template_name, policy_name)
    before, after = user.split(_CONTENT_SENTINEL, 1)
    return PromptEnvelope(system, "", before, after, "user", template_name, policy_name)



def _retreat_to_safe_document_boundary(prefix: str) -> Tuple[str, str]:
    """Avoid submitting a partial HTML/Markdown table or broken final line."""
    if not prefix:
        return prefix, "empty"
    lower = prefix.lower()
    open_pos = lower.rfind("<table")
    close_pos = lower.rfind("</table>")
    if open_pos > close_pos:
        prefix = prefix[:open_pos]
        boundary = "before_unclosed_html_table"
    else:
        boundary = "token_prefix"

    # If the suffix is a contiguous Markdown table block, retreat to its start.
    lines = prefix.splitlines(keepends=True)
    if lines:
        i = len(lines) - 1
        while i >= 0 and (not lines[i].strip() or lines[i].lstrip().startswith("|")):
            i -= 1
        table_start = i + 1
        suffix_lines = lines[table_start:]
        pipe_lines = [line for line in suffix_lines if line.lstrip().startswith("|")]
        separator_present = any(re.match(r"^\s*\|?\s*:?-{3,}", line) for line in pipe_lines)
        final_line_complete = bool(prefix.endswith("\n"))
        if pipe_lines and (not separator_present or not final_line_complete):
            prefix = "".join(lines[:table_start]).rstrip()
            boundary = "before_partial_markdown_table"

    if boundary == "token_prefix" and prefix.endswith("\n"):
        return prefix.rstrip(), "complete_line"

    # End on a complete paragraph/line. Do not discard more than necessary.
    for marker, name in (("\n\n", "paragraph"), ("\n", "line")):
        pos = prefix.rfind(marker)
        if pos >= max(0, len(prefix) - 6000):
            prefix = prefix[: pos + len(marker)].rstrip()
            boundary = f"{boundary}+{name}"
            break
    return prefix, boundary

def fit_sequential_content(
    content: str,
    *,
    envelope: PromptEnvelope,
    client: LLMClient,
    max_input_tokens: int | None,
    safety_margin_tokens: int = 32,
) -> Tuple[str, Dict[str, Any]]:
    """Keep the largest original-text prefix that fits without touching instructions.

    This is intentionally a sequential full-text baseline: no table priority, no
    head-tail selection, and no section routing. Only the article body is shortened.
    """

    full_system, full_user = envelope.render(content)
    full_tokens = client.count_chat_tokens(full_system, full_user)
    if not max_input_tokens or int(max_input_tokens) <= 0:
        return content, {
            "packing_mode": "sequential_full",
            "content_tokens_before": client.count_text_tokens(content),
            "content_tokens_after": client.count_text_tokens(content),
            "final_chat_tokens": full_tokens,
            "max_input_tokens": max_input_tokens,
            "safety_margin_tokens": safety_margin_tokens,
            "content_truncated": False,
            **client.token_counter_metadata(),
        }

    hard_limit = int(max_input_tokens) - max(0, int(safety_margin_tokens))
    empty_system, empty_user = envelope.render("")
    fixed_tokens = client.count_chat_tokens(empty_system, empty_user)
    if fixed_tokens > hard_limit:
        raise ValueError(
            f"fixed prompt requires {fixed_tokens} tokens, above safe input limit {hard_limit}"
        )
    if full_tokens <= hard_limit:
        return content, {
            "packing_mode": "sequential_full",
            "fixed_prompt_tokens": fixed_tokens,
            "content_tokens_before": client.count_text_tokens(content),
            "content_tokens_after": client.count_text_tokens(content),
            "final_chat_tokens": full_tokens,
            "max_input_tokens": int(max_input_tokens),
            "safe_input_limit": hard_limit,
            "safety_margin_tokens": safety_margin_tokens,
            "content_chars_before": len(content),
            "content_chars_after": len(content),
            "content_truncated": False,
            **client.token_counter_metadata(),
        }

    # Character-level binary search preserves the exact document prefix and does
    # not require decoding partial tokenizer byte sequences.
    low, high = 0, len(content)
    best = ""
    best_tokens = fixed_tokens
    while low <= high:
        mid = (low + high) // 2
        candidate = content[:mid]
        system, user = envelope.render(candidate)
        tokens = client.count_chat_tokens(system, user)
        if tokens <= hard_limit:
            best = candidate
            best_tokens = tokens
            low = mid + 1
        else:
            high = mid - 1

    raw_best = best
    best, safe_boundary = _retreat_to_safe_document_boundary(raw_best)
    final_system, final_user = envelope.render(best)
    best_tokens = client.count_chat_tokens(final_system, final_user)
    if best_tokens > hard_limit:
        raise ValueError(f"safe-boundary full prompt exceeds budget: {best_tokens} > {hard_limit}")
    return best, {
        "packing_mode": "safe_sequential_full_v3",
        "fixed_prompt_tokens": fixed_tokens,
        "content_tokens_before": client.count_text_tokens(content),
        "content_tokens_after": client.count_text_tokens(best),
        "final_chat_tokens": best_tokens,
        "max_input_tokens": int(max_input_tokens),
        "safe_input_limit": hard_limit,
        "safety_margin_tokens": safety_margin_tokens,
        "content_chars_before": len(content),
        "content_chars_after": len(best),
        "raw_token_prefix_chars": len(raw_best),
        "safe_boundary_chars_dropped": len(raw_best) - len(best),
        "safe_boundary": safe_boundary,
        "content_truncated": len(best) < len(content),
        "retained_content_fraction": round(len(best) / max(1, len(content)), 6),
        **client.token_counter_metadata(),
    }


def run_single_prompt(
    paper_id: str,
    evidence_text: str,
    client: LLMClient,
    gen: GenerationConfig,
    prompts_dir: Path,
    prompt_variant: str,
    enable_grounding_filter: bool = False,
    require_inhibition_efficiency: bool = False,
    dedup_mode: str = "full_record",
    debug_trace: Optional[DebugTrace] = None,
    stage: str = "single",
    prompt_template: str | None = None,
    policy_variant_override: str | None = None,
    prepared_messages: Tuple[str, str] | None = None,
) -> Tuple[List[Dict[str, Any]], str, str | None]:
    """Run one extraction prompt without permitting client-side prompt truncation."""

    if prepared_messages is None:
        envelope = prepare_prompt_envelope(
            paper_id=paper_id,
            prompts_dir=prompts_dir,
            prompt_variant=prompt_variant,
            prompt_template=prompt_template,
            policy_variant_override=policy_variant_override,
        )
        system_prompt, user_prompt = envelope.render(evidence_text)
        template_name = envelope.template_name
        policy_name = envelope.policy_name
    else:
        system_prompt, user_prompt = prepared_messages
        template_name = prompt_template or prompt_variant
        policy_name = policy_variant_override or prompt_variant

    if debug_trace is not None:
        debug_trace.add("prompt_snapshot.jsonl", {
            "stage": stage,
            "prompt_variant": prompt_variant,
            "prompt_template": template_name,
            "policy_template": policy_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence_char_count": len(evidence_text or ""),
            "final_chat_tokens": client.count_chat_tokens(system_prompt, user_prompt),
            **client.token_counter_metadata(),
        })

    raw = client.generate(system_prompt, user_prompt, gen)
    parsed, err = parse_llm_json(raw, paper_id)
    records = normalize_records(parsed.get("records", []))
    records = split_obvious_multivalue_records(records)
    records = dedup_records(records, mode=dedup_mode)

    if debug_trace is not None:
        debug_trace.add("parsed_before_filter.jsonl", {
            "stage": stage,
            "parse_error": err,
            "record_count": len(records),
            "records": records,
            "generation": dict(getattr(client, "last_generation_metadata", {}) or {}),
        })

    records, filter_report = filter_records_with_report(
        records,
        evidence_text,
        enabled=enable_grounding_filter,
        require_inhibition_efficiency=require_inhibition_efficiency,
    )
    records = dedup_records(records, mode=dedup_mode)

    if debug_trace is not None:
        filter_report = dict(filter_report)
        filter_report.update({
            "stage": stage,
            "parse_error": err,
            "output_count_after_dedup": len(records),
            "dedup_mode": dedup_mode,
        })
        debug_trace.add("grounding_filter_report.jsonl", filter_report)

    return records, raw, err


def prepare_chunks_and_tables(
    paper_id: str,
    text: str,
    retrieval_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunks = split_document(text, paper_id=paper_id, retrieval_cfg=retrieval_cfg)
    rows = extract_markdown_tables(
        text,
        paper_id=paper_id,
        context_chars=int(retrieval_cfg.get("table_context_chars", 1200)),
        max_rows=int(retrieval_cfg.get("max_table_rows", 120)),
    )
    return chunks + rows


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _evidence_cache_path(
    paper_id: str,
    text: str,
    fields_cfg: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    include_groups: List[str],
) -> Path | None:
    if not bool(retrieval_cfg.get("enable_persistent_evidence_cache", True)):
        return None
    root_value = retrieval_cfg.get("_persistent_cache_root")
    if not root_value:
        return None
    module_root = Path(__file__).resolve().parents[1]
    code_files = [
        Path(__file__).resolve(),
        module_root / "retrieval" / "legacy_hybrid.py",
        module_root / "retrieval" / "legacy_exact_chunking.py",
        module_root / "retrieval" / "chunker_factory.py",
        module_root / "retrieval" / "local.py",
        module_root / "schemas.py",
    ]
    cfg_for_hash = {
        k: v for k, v in retrieval_cfg.items()
        if k not in {"_persistent_cache_root", "enable_debug_trace", "debug_trace_level"}
    }
    payload = {
        "paper_id": paper_id,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "include_groups": list(include_groups),
        "field_groups": fields_cfg.get("field_groups", {}),
        "retrieval_config": cfg_for_hash,
        "code_sha256": {p.name: _sha256_file(p) for p in code_files if p.exists()},
        "cache_schema": "8b-opt-evidence-candidates-1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return Path(str(root_value)) / digest[:2] / f"{paper_id}__{digest}.json"


def _load_cached_evidence(path: Path) -> List[EvidenceUnit] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj.get("evidence", [])
        return [EvidenceUnit(**row) for row in rows]
    except Exception:
        return None


def _write_cached_evidence(path: Path, evidence: List[EvidenceUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps({"evidence": [e.to_dict() for e in evidence]}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_local_evidence(
    paper_id: str,
    text: str,
    fields_cfg: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    include_groups: List[str],
) -> List[EvidenceUnit]:
    """Build retrieval candidates; final token packing is strategy-specific."""

    cache_path = _evidence_cache_path(paper_id, text, fields_cfg, retrieval_cfg, include_groups)
    if cache_path is not None:
        cached = _load_cached_evidence(cache_path)
        if cached is not None:
            return cached

    backend = str((retrieval_cfg or {}).get("backend", "legacy_exact")).lower().strip()
    if backend in {"legacy", "legacy_hybrid", "legacy_exact", "old_q1_exact", "hybrid", "old_q1"}:
        evidence = build_legacy_high_recall_evidence(
            paper_id=paper_id,
            text=text,
            fields_cfg=fields_cfg,
            retrieval_cfg=retrieval_cfg,
            include_groups=include_groups,
        )
    else:
        units = prepare_chunks_and_tables(paper_id, text, retrieval_cfg)
        evidence = retrieve_units(
            paper_id=paper_id,
            units=units,
            field_groups=fields_cfg.get("field_groups", {}),
            include_groups=include_groups,
            top_k_per_group=int(retrieval_cfg.get("top_k_per_group", 8)),
            max_chars=int(retrieval_cfg.get("max_evidence_chars", 32000)),
        )
    if cache_path is not None:
        _write_cached_evidence(cache_path, evidence)
        canonical = _load_cached_evidence(cache_path)
        if canonical is not None:
            return canonical
    return evidence
