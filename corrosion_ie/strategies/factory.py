from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import Strategy
from .q0_full import Q0SafeFullStrategy
from .q1_budgeted_rag import Q1BudgetedRagStrategy
from .q2_table_aware_rag import Q2TableAwareRagStrategy
from .non_rag_compression import SectionAwareTokenMatchedStrategy, UniformTokenMatchedStrategy
from .q2_component_ablations import Q1ConditionAnchorsOnlyStrategy, Q1TableBudgetOnlyStrategy


ALIASES = {
    "q0": "q0_safe_full",
    "q0_full": "q0_safe_full",
    "q0_safe": "q0_safe_full",
    "q1": "q1_budgeted_rag",
    "q1_naive_rag": "q1_budgeted_rag",
    "q1_legacy": "q1_budgeted_rag",
    "q1_legacy_rag": "q1_budgeted_rag",
    "q2": "q2_table_aware_rag",
    "q2_table": "q2_table_aware_rag",
    "uniform_token_matched": "b0_uniform_token_matched",
    "section_aware": "b1_section_aware_token_matched",
    "q1_table_only": "a1_q1_table_budget",
    "q1_anchor_only": "a2_q1_condition_anchors",
}

STRATEGIES = {
    "q0_safe_full": Q0SafeFullStrategy,
    "q1_budgeted_rag": Q1BudgetedRagStrategy,
    "q2_table_aware_rag": Q2TableAwareRagStrategy,
    "b0_uniform_token_matched": UniformTokenMatchedStrategy,
    "b1_section_aware_token_matched": SectionAwareTokenMatchedStrategy,
    "a1_q1_table_budget": Q1TableBudgetOnlyStrategy,
    "a2_q1_condition_anchors": Q1ConditionAnchorsOnlyStrategy,
}


def normalize_strategy_name(name: str) -> str:
    key = name.lower().strip()
    return ALIASES.get(key, key)


def build_strategy(
    name: str,
    strategy_cfg: Dict[str, Any],
    fields_cfg: Dict[str, Any],
    prompts_dir: Path,
    prompt_variant: str,
    retrieval_cfg: Dict[str, Any],
) -> Strategy:
    key = normalize_strategy_name(name)
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {name}. Available: {', '.join(STRATEGIES)}")
    cls = STRATEGIES[key]
    if key in {
        "q1_budgeted_rag",
        "q2_table_aware_rag",
        "b0_uniform_token_matched",
        "b1_section_aware_token_matched",
        "a1_q1_table_budget",
        "a2_q1_condition_anchors",
    }:
        return cls(strategy_cfg, fields_cfg, prompts_dir, prompt_variant, retrieval_cfg=retrieval_cfg)
    return cls(strategy_cfg, fields_cfg, prompts_dir, prompt_variant)
