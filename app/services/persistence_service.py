"""Service layer (Stage 6) — persistent evaluation service.

This is the wrapper that turns the in-memory Stage 5 runner into a *persistent*
platform capability, **without modifying Stage 5**. It composes two injected
collaborators:

    EvaluationRunner             (frozen Stage 5 orchestration)
    AbstractEvaluationRepository (Stage 6 persistence abstraction)

It runs an evaluation, then persists the run + records, and exposes read-through
helpers the API uses. Database logic lives entirely in the repository; this
service only coordinates. Both collaborators are constructor-injected, so the
service is fully testable with a fake runner and/or an in-memory repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from app.db.repository import (
    AbstractEvaluationRepository,
    RunMetadata,
    new_run_id,
)

if TYPE_CHECKING:
    from app.datasets.spider_loader import SpiderExample
    from app.db.eval_models import EvaluationRecordModel, EvaluationRunModel
    from app.services.evaluation_runner import (
        EvaluationRecord,
        EvaluationRunner,
        EvaluationSummary,
    )


class PersistentEvaluationService:
    """Run evaluations and persist them; composition over Stage 5."""

    def __init__(
        self,
        runner: "EvaluationRunner",
        repository: AbstractEvaluationRepository,
    ) -> None:
        self._runner = runner
        self._repository = repository

    # -- write path -------------------------------------------------------- #

    def run_and_persist(
        self,
        examples: Iterable["SpiderExample"],
        *,
        metadata: RunMetadata | None = None,
        run_id: str | None = None,
    ) -> tuple["EvaluationRunModel", list["EvaluationRecord"], "EvaluationSummary"]:
        """Evaluate ``examples`` with the frozen runner, then persist the result.

        Returns the persisted run row, the in-memory records, and the summary.
        """
        metadata = metadata or RunMetadata()
        run_id = run_id or new_run_id()

        records, summary = self._runner.run_examples(examples)
        run = self._repository.create_run(
            metadata=metadata,
            records=records,
            summary=summary,
            run_id=run_id,
        )
        return run, records, summary

    # -- read path (delegated to the repository) --------------------------- #

    def get_run(self, run_id: str) -> "EvaluationRunModel | None":
        return self._repository.get_run(run_id)

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list["EvaluationRunModel"]:
        return self._repository.list_runs(limit=limit, offset=offset)

    def get_records(self, run_id: str) -> list["EvaluationRecordModel"]:
        return self._repository.get_records(run_id)
