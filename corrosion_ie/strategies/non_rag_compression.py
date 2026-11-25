from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..debug.trace import DebugTrace, feature_flags
from ..llm.clients import GenerationConfig, LLMClient
from ..retrieval.chunker_factory import split_document
from ..schemas import EvidenceUnit, ExtractionResult
from .base import Strategy
from .common import prepare_prompt_envelope, run_single_prompt
from .token_packing import pack_evidence_for_prompt


_METHOD_RE = re.compile(r"\b(methods?|materials?|experimental|procedure|measurement|testing)\b", re.I)
_RESULT_RE = re.compile(r"\b(results?|discussion|performance|electrochemical|weight\s+loss)\b", re.I)
_CONCLUSION_RE = re.compile(r"\b(conclusions?|summary)\b", re.I)
_ABSTRACT_RE = re.compile(r"\b(abstract|introduction|background)\b", re.I)


def _canonical_model_key(model_name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", str(model_name or "").lower()).strip("_")
    for size in ("14b", "7b", "3b"):
        if size in compact:
            return f"qwen2_5_{size}"
    return compact


def _resolve_project_file(path_value: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _load_token_budget(
    manifest_path: Path,
    *,
    model_name: str,
    reference_strategy: str,
    paper_id: str,
) -> int:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read token-budget manifest {manifest_path}: {exc}") from exc
    model_key = _canonical_model_key(model_name)
    try:
        value = payload["budgets"][model_key][reference_strategy][paper_id]
    except KeyError as exc:
        raise KeyError(
            "token-budget entry missing for "
            f"model={model_key}, strategy={reference_strategy}, paper={paper_id} "
            f"in {manifest_path}"
        ) from exc
    target = int(value)
    if target <= 0:
        raise ValueError(f"invalid token budget {target} for {model_key}/{reference_strategy}/{paper_id}")
    return target


def _chunk_to_evidence(paper_id: str, chunk: Dict[str, Any], rank: int) -> EvidenceUnit:
    unit_id = str(chunk.get("unit_id") or chunk.get("chunk_id") or f"{paper_id}#chunk_{rank - 1}")
    metadata = {k: v for k, v in chunk.items() if k not in {"text", "columns"}}
    metadata.update({
        "direct_hit": True,
        "non_rag_control": True,
        "selection_score": float(max(1, 1_000_000 - rank)),
    })
    return EvidenceUnit(
        paper_id=paper_id,
        unit_id=unit_id,
        unit_type=str(chunk.get("unit_type") or "chunk"),
        text=str(chunk.get("text") or ""),
        query_group="non_rag_control",
        query="deterministic document-structure control; no semantic retrieval",
        rank=rank,
        score=float(max(1, 1_000_000 - rank)),
        section=str(chunk.get("section") or chunk.get("section_title") or ""),
        caption=str(chunk.get("caption") or ""),
        columns=dict(chunk.get("columns") or {}),
        metadata=metadata,
    )


def _uniform_spread_indices(indices: Sequence[int]) -> List[int]:
    """Return a deterministic farthest-point traversal over document indices."""

    remaining = sorted(set(int(i) for i in indices))
    if len(remaining) <= 2:
        return remaining
    selected = [remaining.pop(0)]
    selected.append(remaining.pop(-1))
    while remaining:
        best = max(
            remaining,
            key=lambda idx: (min(abs(idx - chosen) for chosen in selected), -idx),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _section_bucket(chunk: Dict[str, Any]) -> str:
    content_type = str(chunk.get("content_type") or chunk.get("chunk_type") or "").lower()
    section = str(chunk.get("section") or chunk.get("section_title") or "")
    if content_type == "table":
        return "table"
    if _METHOD_RE.search(section):
        return "methods"
    if _RESULT_RE.search(section):
        return "results"
    if _CONCLUSION_RE.search(section):
        return "conclusion"
    if _ABSTRACT_RE.search(section):
        return "abstract"
    return "other"


def _section_aware_indices(chunks: Sequence[Dict[str, Any]]) -> List[int]:
    """Round-robin fixed document sections without field-value retrieval."""

    order = ("table", "methods", "results", "conclusion", "abstract", "other")
    buckets = {
        name: _uniform_spread_indices(
            [idx for idx, chunk in enumerate(chunks) if _section_bucket(chunk) == name]
        )
        for name in order
    }
    selected: List[int] = []
    depth = 0
    while True:
        added = False
        for name in order:
            values = buckets[name]
            if depth < len(values):
                selected.append(values[depth])
                added = True
        if not added:
            break
        depth += 1
    return selected


class _TokenMatchedNonRagStrategy(Strategy):
    """Compress the target paper without dense, sparse, or rerank retrieval."""

    selection_mode = "uniform"

    def __init__(self, *args, retrieval_cfg=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retrieval_cfg = retrieval_cfg or {}
        self._evidence: List[EvidenceUnit] = []

    def _ordered_evidence(self, paper_id: str, text: str) -> List[EvidenceUnit]:
        chunks = split_document(text, paper_id=paper_id, retrieval_cfg=self.retrieval_cfg)
        if self.selection_mode == "section_aware":
            indices = _section_aware_indices(chunks)
        else:
            indices = _uniform_spread_indices(range(len(chunks)))
        return [
            _chunk_to_evidence(paper_id, chunks[idx], rank=rank)
            for rank, idx in enumerate(indices, start=1)
        ]

    def run(
        self,
        paper_id: str,
        text: str,
        client: LLMClient,
        model_name: str,
        gen: GenerationConfig,
    ) -> ExtractionResult:
        flags = feature_flags(self.retrieval_cfg, self.config)
        debug_trace = DebugTrace(
            paper_id,
            enabled=bool(flags["enable_debug_trace"]),
            level=str(flags["debug_trace_level"]),
        )
        manifest_path = _resolve_project_file(
            str(self.config.get("token_budget_manifest_path", ""))
        )
        reference_strategy = str(
            self.config.get("token_budget_reference_strategy", "q1_budgeted_rag")
        )
        target_tokens = _load_token_budget(
            manifest_path,
            model_name=model_name,
            reference_strategy=reference_strategy,
            paper_id=paper_id,
        )
        if gen.max_input_tokens:
            target_tokens = min(target_tokens, int(gen.max_input_tokens))

        prompt_template = str(self.config.get("prompt_template", "templates/q1_rag"))
        envelope = prepare_prompt_envelope(
            paper_id=paper_id,
            prompts_dir=self.prompts_dir,
            prompt_variant=self.prompt_variant,
            prompt_template=prompt_template,
        )
        ordered = self._ordered_evidence(paper_id, text)
        packed = pack_evidence_for_prompt(
            paper_id=paper_id,
            evidence=ordered,
            envelope=envelope,
            client=client,
            max_input_tokens=target_tokens,
            max_items=max(1, len(ordered)),
            max_window_chunks=0,
            safety_margin_tokens=0,
            table_aware=False,
            condition_aware=False,
        )
        self._evidence = packed.evidence
        achieved_tokens = int(packed.diagnostics.get("final_chat_tokens") or 0)
        budget_diag = {
            "selection_mode": self.selection_mode,
            "token_budget_manifest_path": str(manifest_path),
            "token_budget_reference_strategy": reference_strategy,
            "target_final_chat_tokens": target_tokens,
            "achieved_final_chat_tokens": achieved_tokens,
            "token_budget_difference": achieved_tokens - target_tokens,
            "token_budget_utilization": round(achieved_tokens / max(1, target_tokens), 6),
            "semantic_retrieval_used": False,
        }
        packed.diagnostics.update(budget_diag)
        debug_trace.add(
            "retrieval_trace.jsonl",
            {
                "strategy": self.name,
                "prompt_variant": self.prompt_variant,
                **budget_diag,
                "full_text_char_count": len(text),
                "candidate_evidence_count": len(ordered),
                "packing": packed.diagnostics,
            },
        )
        debug_trace.add_many("evidence_candidates.jsonl", [e.to_dict() for e in ordered])
        debug_trace.add_many("evidence_selected.jsonl", [e.to_dict() for e in packed.evidence])

        dedup_mode = str(self.config.get("dedup_mode", "performance_with_time"))
        records, raw, err = run_single_prompt(
            paper_id,
            packed.text,
            client,
            gen,
            self.prompts_dir,
            self.prompt_variant,
            prompt_template=prompt_template,
            prepared_messages=packed.messages,
            enable_grounding_filter=bool(flags["enable_grounding_filter"]),
            dedup_mode=dedup_mode,
            debug_trace=debug_trace,
            stage=f"{self.name}_single_prompt",
        )
        generation = dict(getattr(client, "last_generation_metadata", {}) or {})
        debug_trace.set_summary(
            strategy=self.name,
            prompt_variant=self.prompt_variant,
            model=model_name,
            records_after_filter=len(records),
            parse_error=err,
            packing=packed.diagnostics,
            generation=generation,
        )
        return ExtractionResult(
            paper_id,
            self.name,
            model_name,
            records,
            raw,
            err,
            [e.unit_id for e in packed.evidence],
            {
                **budget_diag,
                "candidate_evidence_count": len(ordered),
                "packed_evidence_count": len(packed.evidence),
                "packing": packed.diagnostics,
                "full_text_char_count": len(text),
                "context_compression_ratio": round(len(text) / max(1, len(packed.text)), 4),
                "evidence_fraction_of_full_text": round(len(packed.text) / max(1, len(text)), 4),
                "generation": generation,
                "debug_trace": debug_trace.to_dict(),
            },
        )

    def evidence_log(self) -> List[EvidenceUnit]:
        return self._evidence


class UniformTokenMatchedStrategy(_TokenMatchedNonRagStrategy):
    name = "b0_uniform_token_matched"
    selection_mode = "uniform"


class SectionAwareTokenMatchedStrategy(_TokenMatchedNonRagStrategy):
    name = "b1_section_aware_token_matched"
    selection_mode = "section_aware"
