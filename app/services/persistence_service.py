"""Service layer (Stage 6 + 7) — persistent evaluation service.

Stage 6 introduced ``run_and_persist``, which runs an evaluation synchronously
and persists everything in one call.  That method is preserved exactly.

Stage 7 (Phase 1) adds the lifecycle half of the persistence service without
modifying the existing interface:

    create_pending(metadata, run_id)
        Insert a pending run row and return it.  Used by the async submission
        path to make the run visible before execution begins.

    execute_and_finalize(run_id, examples, metadata)
        Meant to be called by a background worker on its own session/repository.
        Drives the runner, then either finalizes (completed) or marks failed.
        No thread management here — that belongs to the JobExecutor (Phase 2).

Both collaborators remain constructor-injected so the service is fully testable
with fake runners and/or in-memory repositories.
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
    """Run evaluations and persist them; composition over Stage 5.

    All Stage 6 behaviour is preserved.  Stage 7 adds ``create_pending`` and
    ``execute_and_finalize`` without changing the existing ``run_and_persist``
    method or its callers.
    """

    def __init__(
        self,
        runner: "EvaluationRunner",
        repository: AbstractEvaluationRepository,
    ) -> None:
        self._runner = runner
        self._repository = repository

    # ------------------------------------------------------------------ #
    # Stage 6 write path (frozen)                                         #
    # ------------------------------------------------------------------ #

    def run_and_persist(
        self,
        examples: Iterable["SpiderExample"],
        *,
        metadata: RunMetadata | None = None,
        run_id: str | None = None,
    ) -> tuple["EvaluationRunModel", list["EvaluationRecord"], "EvaluationSummary"]:
        """Evaluate ``examples`` with the frozen runner, then persist the result.

        Synchronous end-to-end: evaluate → persist records + summary → return.
        The run is persisted with ``status`` taken from ``metadata.status``
        (default ``"completed"`` per ``RunMetadata``).

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

    # ------------------------------------------------------------------ #
    # Stage 7 lifecycle methods                                           #
    # ------------------------------------------------------------------ #

    def create_pending(
        self,
        metadata: RunMetadata | None = None,
        *,
        run_id: str | None = None,
    ) -> "EvaluationRunModel":
        """Insert a run row with ``status=pending`` and return it immediately.

        No examples are evaluated; records and summary columns are empty.
        The returned run row is the handle the caller uses to poll lifecycle
        state via ``GET /runs/{id}``.
        """
        metadata = metadata or RunMetadata()
        run_id = run_id or new_run_id()
        return self._repository.create_pending_run(
            metadata=metadata,
            run_id=run_id,
        )

    def execute_and_finalize(
        self,
        run_id: str,
        examples: Iterable["SpiderExample"],
    ) -> "EvaluationRunModel | None":
        """Execute evaluation for an existing pending run and persist the result.

        Intended to be called by a background worker that owns its own session
        and therefore its own ``PersistentEvaluationService`` instance.

        Lifecycle transitions performed here:
            pending → running   (before evaluation starts)
            running → completed (after successful finalization)
            running → failed    (on any exception — dataset load, DB, unexpected)

        Per-example errors (generation failures, execution errors) are *not*
        run-level failures; the runner's "never raises" contract means they are
        captured in records and the run still completes normally.

        Returns the final ``EvaluationRunModel`` (completed or failed), or
        ``None`` if the ``run_id`` no longer exists in the database.
        """
        # Transition to running.
        run = self._repository.set_status(run_id, "running")
        if run is None:
            return None

        try:
            records, summary = self._runner.run_examples(examples)
        except Exception as exc:  # noqa: BLE001
            # Infrastructure failure (dataset load, unexpected error).
            # Per-example failures are caught inside the runner and never reach
            # here; this branch handles truly unexpected exceptions only.
            error_msg = f"{type(exc).__name__}: {exc}"
            return self._repository.set_status(run_id, "failed", error=error_msg)

        try:
            return self._repository.finalize_run(
                run_id,
                records=records,
                summary=summary,
            )
        except Exception as exc:  # noqa: BLE001
            # DB / ORM failure during finalization.
            error_msg = f"{type(exc).__name__}: {exc}"
            return self._repository.set_status(run_id, "failed", error=error_msg)

    # ------------------------------------------------------------------ #
    # Read path (delegated to the repository — Stage 6, frozen)           #
    # ------------------------------------------------------------------ #

    def get_run(self, run_id: str) -> "EvaluationRunModel | None":
        return self._repository.get_run(run_id)

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list["EvaluationRunModel"]:
        return self._repository.list_runs(limit=limit, offset=offset)

    def get_records(self, run_id: str) -> list["EvaluationRecordModel"]:
        return self._repository.get_records(run_id)