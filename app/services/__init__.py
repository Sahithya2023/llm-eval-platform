"""Service layer: orchestration over the evaluation components.

Stage 5 added the evaluation runner (orchestration only). Stage 6 adds the
persistent evaluation service, which *composes* the frozen runner with the
repository to persist runs and records without modifying Stage 5.
"""

from app.services.evaluation_runner import (
    EvaluationRecord,
    EvaluationRunner,
    EvaluationSummary,
)
from app.services.persistence_service import PersistentEvaluationService

__all__ = [
    "EvaluationRunner",
    "EvaluationRecord",
    "EvaluationSummary",
    "PersistentEvaluationService",
]
