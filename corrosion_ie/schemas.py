from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

FIELD_NAMES = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
]

CORE_FIELDS = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "test_method",
    "inhibition_efficiency",
]

@dataclass
class EvidenceUnit:
    paper_id: str
    unit_id: str
    unit_type: str
    text: str
    query_group: str = ""
    query: str = ""
    rank: int = 0
    score: float = 0.0
    section: str = ""
    caption: str = ""
    columns: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RawModelOutput:
    paper_id: str
    strategy: str
    prompt_variant: str
    stage: str
    raw_output: str
    parse_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunIdentity:
    experiment_id: str
    model_key: str
    strategy: str
    prompt_variant: str
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.model_key}__{self.strategy}__{self.prompt_variant}__r{self.seed}"

    def output_dir(self, output_root: Path) -> Path:
        return output_root / self.experiment_id / self.model_key / self.strategy / self.prompt_variant / f"r{self.seed}"

@dataclass
class ExtractionResult:
    paper_id: str
    strategy: str
    model: str
    records: List[Dict[str, Any]]
    raw_output: str = ""
    parse_error: Optional[str] = None
    evidence_unit_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prediction_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "strategy": self.strategy,
            "model": self.model,
            "result": {"paper_id": self.paper_id, "records": self.records},
            "raw_output": self.raw_output,
            "parse_error": self.parse_error,
            "evidence_unit_ids": self.evidence_unit_ids,
            "metadata": self.metadata,
        }
