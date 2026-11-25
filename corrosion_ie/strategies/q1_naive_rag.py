"""Compatibility wrapper for historical imports.

Active experiments should use ``q1_budgeted_rag``. The old name is intentionally
mapped to the corrected token-budgeted implementation so unsafe prompt packing
cannot be reintroduced accidentally.
"""
from .q1_budgeted_rag import Q1BudgetedRagStrategy

Q1NaiveRagStrategy = Q1BudgetedRagStrategy
