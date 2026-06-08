"""Service layer: orchestration over the evaluation components.

Stage 5 adds the evaluation runner; trace persistence and aggregate stats
arrive in a later stage.
"""

from app.services.evaluation_runner import (
    EvaluationRecord,
    EvaluationRunner,
    EvaluationSummary,
)

__all__ = [
    "EvaluationRunner",
    "EvaluationRecord",
    "EvaluationSummary",
]
