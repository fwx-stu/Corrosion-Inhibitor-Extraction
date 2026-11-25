from __future__ import annotations

from typing import List

from ..debug.trace import cfg_bool
from ..llm.clients import GenerationConfig, LLMClient
from ..schemas import EvidenceUnit, ExtractionResult
from .base import Strategy
from .common import fit_sequential_content, prepare_prompt_envelope, run_single_prompt


class Q0SafeFullStrategy(Strategy):
    """Sequential full-text baseline with protected extraction instructions."""

    name = "q0_safe_full"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._evidence: List[EvidenceUnit] = []

    def run(self, paper_id: str, text: str, client: LLMClient, model_name: str, gen: GenerationConfig) -> ExtractionResult:
        prompt_template = str(self.config.get("prompt_template", "templates/q0_full"))
        envelope = prepare_prompt_envelope(
            paper_id=paper_id,
            prompts_dir=self.prompts_dir,
            prompt_variant=self.prompt_variant,
            prompt_template=prompt_template,
        )
        clipped, packing = fit_sequential_content(
            text,
            envelope=envelope,
            client=client,
            max_input_tokens=gen.max_input_tokens,
            safety_margin_tokens=int(self.config.get("input_safety_margin_tokens", 32)),
        )
        messages = envelope.render(clipped)
        ev = EvidenceUnit(
            paper_id=paper_id,
            unit_id=f"{paper_id}::safe_sequential_full",
            unit_type="full_text",
            text=clipped,
            query_group="full",
            query="safe sequential full-text prefix",
            rank=1,
            score=1.0,
            metadata={
                "original_char_count": len(text),
                "submitted_char_count": len(clipped),
                "content_truncated": len(clipped) < len(text),
            },
        )
        self._evidence = [ev]
        records, raw, err = run_single_prompt(
            paper_id,
            clipped,
            client,
            gen,
            self.prompts_dir,
            self.prompt_variant,
            prompt_template=prompt_template,
            prepared_messages=messages,
            dedup_mode=str(self.config.get("dedup_mode", "performance_with_time")),
            enable_grounding_filter=cfg_bool(self.config, "enable_grounding_filter", False),
        )
        generation_meta = dict(getattr(client, "last_generation_metadata", {}) or {})
        return ExtractionResult(
            paper_id,
            self.name,
            model_name,
            records,
            raw,
            err,
            [ev.unit_id],
            {
                "full_text_char_count": len(text),
                "submitted_char_count": len(clipped),
                "content_truncated": len(clipped) < len(text),
                "packing": packing,
                "generation": generation_meta,
            },
        )

    def evidence_log(self) -> List[EvidenceUnit]:
        return self._evidence


# Backward-compatible import name. Active configs use q0_safe_full.
Q0FullStrategy = Q0SafeFullStrategy
