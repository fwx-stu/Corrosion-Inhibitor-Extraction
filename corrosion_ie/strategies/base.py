from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..llm.clients import GenerationConfig, LLMClient
from ..schemas import EvidenceUnit, ExtractionResult


class Strategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], fields_cfg: Dict[str, Any], prompts_dir: Path, prompt_variant: str) -> None:
        self.config = config
        self.fields_cfg = fields_cfg
        self.prompts_dir = prompts_dir
        self.prompt_variant = prompt_variant

    @abstractmethod
    def run(
        self,
        paper_id: str,
        text: str,
        client: LLMClient,
        model_name: str,
        gen: GenerationConfig,
    ) -> ExtractionResult:
        raise NotImplementedError

    def evidence_log(self) -> List[EvidenceUnit]:
        return []
