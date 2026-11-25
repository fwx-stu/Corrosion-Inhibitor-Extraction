from __future__ import annotations

from typing import List

from ..debug.trace import DebugTrace, feature_flags
from ..llm.clients import GenerationConfig, LLMClient
from ..postprocess.normalizer import dedup_records, normalize_records
from ..retrieval.row_level_records import extract_row_level_records, make_row_level_evidence_units
from ..schemas import EvidenceUnit, ExtractionResult
from .base import Strategy
from .common import build_local_evidence, prepare_prompt_envelope, run_single_prompt
from .token_packing import pack_evidence_for_prompt


class Q1BudgetedRagStrategy(Strategy):
    """Legacy hybrid retrieval with tokenizer-aware final evidence packing."""

    name = "q1_budgeted_rag"

    def __init__(self, *args, retrieval_cfg=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retrieval_cfg = retrieval_cfg or {}
        self._evidence: List[EvidenceUnit] = []

    def run(self, paper_id: str, text: str, client: LLMClient, model_name: str, gen: GenerationConfig) -> ExtractionResult:
        flags = feature_flags(self.retrieval_cfg, self.config)
        debug_trace = DebugTrace(
            paper_id,
            enabled=bool(flags["enable_debug_trace"]),
            level=str(flags["debug_trace_level"]),
        )
        include = self.config.get("include_field_groups", ["performance", "conditions", "quantum", "adsorption"])
        base_evidence = build_local_evidence(paper_id, text, self.fields_cfg, self.retrieval_cfg, include)
        candidates = list(base_evidence)

        row_evidence: List[EvidenceUnit] = []
        if flags["enable_row_level_table_evidence"]:
            row_evidence = make_row_level_evidence_units(
                paper_id,
                text,
                max_units=int(self.retrieval_cfg.get("row_table_max_units", 120)),
            )
            candidates = row_evidence + candidates

        prompt_template = str(self.config.get("prompt_template", "templates/q1_rag"))
        envelope = prepare_prompt_envelope(
            paper_id=paper_id,
            prompts_dir=self.prompts_dir,
            prompt_variant=self.prompt_variant,
            prompt_template=prompt_template,
        )
        packed = pack_evidence_for_prompt(
            paper_id=paper_id,
            evidence=candidates,
            envelope=envelope,
            client=client,
            max_input_tokens=gen.max_input_tokens,
            max_items=int(self.config.get("max_final_chunks", self.retrieval_cfg.get("max_final_chunks", 40))),
            max_window_chunks=int(self.config.get("max_window_chunks", self.retrieval_cfg.get("max_window_chunks", 8))),
            safety_margin_tokens=int(self.config.get("input_safety_margin_tokens", 32)),
            table_aware=False,
        )
        self._evidence = packed.evidence
        dedup_mode = str(self.config.get("dedup_mode", "performance_with_time"))

        debug_trace.add("retrieval_trace.jsonl", {
            "strategy": self.name,
            "prompt_variant": self.prompt_variant,
            "backend": str(self.retrieval_cfg.get("backend", "legacy_exact")),
            "include_field_groups": include,
            "flags": flags,
            "base_evidence_count": len(base_evidence),
            "row_table_evidence_count": len(row_evidence),
            "candidate_evidence_count": len(candidates),
            "packing": packed.diagnostics,
            "full_text_char_count": len(text),
            "dedup_mode": dedup_mode,
        })
        debug_trace.add_many("evidence_candidates.jsonl", [e.to_dict() for e in candidates])
        debug_trace.add_many("evidence_selected.jsonl", [e.to_dict() for e in packed.evidence])

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
            stage="q1_budgeted_single_prompt",
        )

        row_candidates = []
        if flags["merge_row_table_candidates"]:
            row_candidates = extract_row_level_records(
                paper_id,
                text,
                max_records=int(self.retrieval_cfg.get("row_table_max_candidates", 180)),
            )
            if row_candidates:
                records = dedup_records(normalize_records(list(records) + row_candidates), mode=dedup_mode)

        generation = dict(getattr(client, "last_generation_metadata", {}) or {})
        debug_trace.set_summary(
            strategy=self.name,
            prompt_variant=self.prompt_variant,
            model=model_name,
            candidate_evidence_count=len(candidates),
            packed_evidence_count=len(packed.evidence),
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
                "candidate_evidence_count": len(candidates),
                "packed_evidence_count": len(packed.evidence),
                "row_table_evidence_count": len(row_evidence),
                "row_table_candidate_count": len(row_candidates),
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
