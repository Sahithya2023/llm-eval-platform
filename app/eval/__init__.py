"""Evaluation layer: execution-based oracle for Text-to-SQL.

Independent of the API/UI layers (no FastAPI, Streamlit, or SQLAlchemy
imports). Callable from a CLI or a service without modification.
"""

from app.eval.comparator import (
    Comparison,
    compare,
    compare_executions,
    compare_result_sets,
    gold_requires_order,
)
from app.eval.executor import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    ExecutionResult,
    SqlExecutor,
)
from app.eval.llm_client import GenerationResult, TextToSqlClient

__all__ = [
    "SqlExecutor",
    "ExecutionResult",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_MAX_ROWS",
    "Comparison",
    "compare",
    "compare_result_sets",
    "compare_executions",
    "gold_requires_order",
    "TextToSqlClient",
    "GenerationResult",
]
