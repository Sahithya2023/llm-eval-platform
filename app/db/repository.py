"""Repository layer (Stage 6 + 7) — a clean persistence abstraction.

Stage 6 introduced the core read/write API:

    save_run / save_records / create_run   (writes)
    get_run / list_runs / get_records      (reads)

Stage 7 (Phase 1) adds three lifecycle methods that support the pending →
running → completed/failed state machine without touching the Stage 6 API:

    create_pending_run   create a run row in status=pending; no records yet.
    set_status           transition a run to running/completed/failed, setting
                         started_at / finished_at / error as appropriate.
    finalize_run         write records + summary then atomically flip to
                         completed in a single unit of work.

All Stage 6 methods are preserved exactly so existing callers and tests are
unaffected.  The Protocol is extended with the three new methods; the
``FakeRepo`` in tests only needs the new methods if it exercises the new path
(existing tests don't, so FakeRepo is unchanged).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, Sequence, runtime_checkable

from app.db.eval_models import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    EvaluationRecordModel,
    EvaluationRunModel,
)

if TYPE_CHECKING:  # avoid importing heavy/ORM-adjacent symbols at runtime
    from sqlalchemy.orm import Session

    from app.services.evaluation_runner import EvaluationRecord, EvaluationSummary


# Lifecycle status literal — used for type-checking set_status calls.
RunStatus = Literal["pending", "running", "completed", "failed"]


class RunMetadata(NamedTuple):
    """Run-level configuration captured alongside the aggregate summary."""

    name: str | None = None
    notes: str | None = None
    dataset: str = "spider"
    split: str | None = None
    dataset_limit: int | None = None
    model: str | None = None
    prompt_version: str | None = None
    status: str = "completed"


@runtime_checkable
class AbstractEvaluationRepository(Protocol):
    """Persistence contract the service layer depends on.

    Stage 6 methods are listed first (frozen); Stage 7 additions follow.
    """

    # -- Stage 6 (frozen) -------------------------------------------------- #

    def save_run(
        self, *, run_id: str, metadata: RunMetadata, summary: "EvaluationSummary"
    ) -> EvaluationRunModel: ...

    def save_records(
        self, run_id: str, records: Sequence["EvaluationRecord"]
    ) -> list[EvaluationRecordModel]: ...

    def create_run(
        self,
        *,
        metadata: RunMetadata,
        records: Sequence["EvaluationRecord"],
        summary: "EvaluationSummary",
        run_id: str | None = None,
    ) -> EvaluationRunModel: ...

    def get_run(self, run_id: str) -> EvaluationRunModel | None: ...

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[EvaluationRunModel]: ...

    def get_records(self, run_id: str) -> list[EvaluationRecordModel]: ...

    # -- Stage 7 (new) ----------------------------------------------------- #

    def create_pending_run(
        self,
        *,
        metadata: RunMetadata,
        run_id: str | None = None,
    ) -> EvaluationRunModel: ...

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> EvaluationRunModel | None: ...

    def finalize_run(
        self,
        run_id: str,
        *,
        records: Sequence["EvaluationRecord"],
        summary: "EvaluationSummary",
    ) -> EvaluationRunModel | None: ...


def new_run_id() -> str:
    """Generate a short, URL-safe run identifier."""
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class SqlAlchemyEvaluationRepository:
    """SQLAlchemy-backed implementation of :class:`AbstractEvaluationRepository`.

    Holds a single :class:`~sqlalchemy.orm.Session` (request-scoped in the API,
    or an in-memory session in tests). It commits its own units of work so the
    service layer stays free of transaction plumbing.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # Stage 6 writes (frozen — do not modify)                             #
    # ------------------------------------------------------------------ #

    def save_run(
        self, *, run_id: str, metadata: RunMetadata, summary: "EvaluationSummary"
    ) -> EvaluationRunModel:
        run = EvaluationRunModel(
            id=run_id,
            name=metadata.name,
            notes=metadata.notes,
            dataset=metadata.dataset,
            split=metadata.split,
            dataset_limit=metadata.dataset_limit,
            model=metadata.model,
            prompt_version=metadata.prompt_version,
            status=metadata.status,
            total_examples=summary.total_examples,
            correct=summary.correct,
            incorrect=summary.incorrect,
            failed_generations=summary.failed_generations,
            failed_executions=summary.failed_executions,
            accuracy=summary.accuracy,
            total_prompt_tokens=summary.total_prompt_tokens,
            total_completion_tokens=summary.total_completion_tokens,
            total_tokens=summary.total_tokens,
            total_estimated_cost=summary.total_estimated_cost,
            average_latency_ms=summary.average_latency_ms,
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def save_records(
        self, run_id: str, records: Sequence["EvaluationRecord"]
    ) -> list[EvaluationRecordModel]:
        rows = [self._to_record_model(run_id, r) for r in records]
        if rows:
            self._session.add_all(rows)
            self._session.commit()
        return rows

    def create_run(
        self,
        *,
        metadata: RunMetadata,
        records: Sequence["EvaluationRecord"],
        summary: "EvaluationSummary",
        run_id: str | None = None,
    ) -> EvaluationRunModel:
        """Persist a run and all its records as one unit of work."""
        run_id = run_id or new_run_id()
        run = EvaluationRunModel(
            id=run_id,
            name=metadata.name,
            notes=metadata.notes,
            dataset=metadata.dataset,
            split=metadata.split,
            dataset_limit=metadata.dataset_limit,
            model=metadata.model,
            prompt_version=metadata.prompt_version,
            status=metadata.status,
            total_examples=summary.total_examples,
            correct=summary.correct,
            incorrect=summary.incorrect,
            failed_generations=summary.failed_generations,
            failed_executions=summary.failed_executions,
            accuracy=summary.accuracy,
            total_prompt_tokens=summary.total_prompt_tokens,
            total_completion_tokens=summary.total_completion_tokens,
            total_tokens=summary.total_tokens,
            total_estimated_cost=summary.total_estimated_cost,
            average_latency_ms=summary.average_latency_ms,
        )
        run.records = [self._to_record_model(run_id, r) for r in records]
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    # ------------------------------------------------------------------ #
    # Stage 6 reads (frozen — do not modify)                              #
    # ------------------------------------------------------------------ #

    def get_run(self, run_id: str) -> EvaluationRunModel | None:
        return self._session.get(EvaluationRunModel, run_id)

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[EvaluationRunModel]:
        from sqlalchemy import select

        stmt = (
            select(EvaluationRunModel)
            .order_by(EvaluationRunModel.created_at.desc(), EvaluationRunModel.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_records(self, run_id: str) -> list[EvaluationRecordModel]:
        from sqlalchemy import select

        stmt = (
            select(EvaluationRecordModel)
            .where(EvaluationRecordModel.run_id == run_id)
            .order_by(EvaluationRecordModel.id)
        )
        return list(self._session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------ #
    # Stage 7 writes (lifecycle)                                          #
    # ------------------------------------------------------------------ #

    def create_pending_run(
        self,
        *,
        metadata: RunMetadata,
        run_id: str | None = None,
    ) -> EvaluationRunModel:
        """Insert a run row in ``pending`` status with no records or summary yet.

        This is the first step of the Stage 7 async path: the run is visible
        immediately (via GET /runs/{id}) and callers can poll ``status`` to
        observe the lifecycle transition.  Summary columns default to 0 / 0.0
        and are only meaningful after ``finalize_run`` promotes the run to
        ``completed``.

        The ``metadata.status`` field is intentionally ignored here; the status
        is always ``pending`` so that the lifecycle is authoritative.
        """
        run_id = run_id or new_run_id()
        run = EvaluationRunModel(
            id=run_id,
            name=metadata.name,
            notes=metadata.notes,
            dataset=metadata.dataset,
            split=metadata.split,
            dataset_limit=metadata.dataset_limit,
            model=metadata.model,
            prompt_version=metadata.prompt_version,
            status=STATUS_PENDING,
            # Summary columns start at their defaults (0 / 0.0).
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> EvaluationRunModel | None:
        """Transition a run to a new lifecycle status.

        Timestamp semantics
        -------------------
        * ``running``   → sets ``started_at`` to now (once; subsequent calls are
                          no-ops on that field because ``started_at`` is already
                          set).
        * ``completed`` → sets ``finished_at`` to now.
        * ``failed``    → sets ``finished_at`` to now and stores ``error``.
        * ``pending``   → clears all lifecycle timestamps (used for reset /
                          testing; not expected in production flows).

        Returns the refreshed ``EvaluationRunModel``, or ``None`` if the run_id
        is not found (caller decides whether to treat this as an error).
        """
        run = self._session.get(EvaluationRunModel, run_id)
        if run is None:
            return None

        now = _utcnow()
        run.status = status

        if status == STATUS_RUNNING:
            # Only stamp started_at on the first running transition.
            if run.started_at is None:
                run.started_at = now
        elif status == STATUS_COMPLETED:
            run.finished_at = now
        elif status == STATUS_FAILED:
            run.finished_at = now
            run.error = error
        elif status == STATUS_PENDING:
            # Reset — used in tests or administrative tooling.
            run.started_at = None
            run.finished_at = None
            run.error = None

        self._session.commit()
        self._session.refresh(run)
        return run

    def finalize_run(
        self,
        run_id: str,
        *,
        records: Sequence["EvaluationRecord"],
        summary: "EvaluationSummary",
    ) -> EvaluationRunModel | None:
        """Write records + summary and atomically transition to ``completed``.

        This is the single unit of work that closes the async execution path.
        All record rows and the summary update are committed together so the
        database never shows a ``completed`` run with missing records or a
        partially-updated summary.

        Returns ``None`` if the run_id is not found (the background worker
        should treat this as a fatal error and log it).
        """
        run = self._session.get(EvaluationRunModel, run_id)
        if run is None:
            return None

        # Write records.
        record_rows = [self._to_record_model(run_id, r) for r in records]
        if record_rows:
            self._session.add_all(record_rows)

        # Update summary columns in-place.
        run.total_examples = summary.total_examples
        run.correct = summary.correct
        run.incorrect = summary.incorrect
        run.failed_generations = summary.failed_generations
        run.failed_executions = summary.failed_executions
        run.accuracy = summary.accuracy
        run.total_prompt_tokens = summary.total_prompt_tokens
        run.total_completion_tokens = summary.total_completion_tokens
        run.total_tokens = summary.total_tokens
        run.total_estimated_cost = summary.total_estimated_cost
        run.average_latency_ms = summary.average_latency_ms

        # Transition to completed.
        run.status = STATUS_COMPLETED
        run.finished_at = _utcnow()

        self._session.commit()
        self._session.refresh(run)
        return run

    # ------------------------------------------------------------------ #
    # Shared mapping helper                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_record_model(
        run_id: str, record: "EvaluationRecord"
    ) -> EvaluationRecordModel:
        return EvaluationRecordModel(
            run_id=run_id,
            example_id=record.example_id,
            db_id=record.db_id,
            question=record.question,
            gold_sql=record.gold_sql,
            generated_sql=record.generated_sql,
            is_correct=record.is_correct,
            generation_error=record.generation_error,
            execution_error=record.execution_error,
            comparison_reason=record.comparison_reason,
            model=record.model,
            prompt_version=record.prompt_version,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            estimated_cost=record.estimated_cost,
            latency_ms=record.latency_ms,
        )