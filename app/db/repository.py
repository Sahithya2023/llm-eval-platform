"""Repository layer (Stage 6) — a clean persistence abstraction.

This is the *only* module that knows how Stage 5 results are stored. It maps the
in-memory Stage 5 shapes (:class:`EvaluationRecord`, :class:`EvaluationSummary`)
to/from the Stage 6 ORM models and exposes a small, intention-revealing API:

    save_run / save_records / create_run   (writes)
    get_run / list_runs / get_records      (reads)

The orchestration runner (Stage 5) never imports this; persistence is composed
on top by the service layer. An :class:`AbstractEvaluationRepository` Protocol
lets the service depend on an abstraction, so it can be tested with a fake
repository and the concrete implementation can be swapped (e.g. for Postgres)
without touching callers.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, NamedTuple, Protocol, Sequence, runtime_checkable

from app.db.eval_models import EvaluationRecordModel, EvaluationRunModel

if TYPE_CHECKING:  # avoid importing heavy/ORM-adjacent symbols at runtime
    from sqlalchemy.orm import Session

    from app.services.evaluation_runner import EvaluationRecord, EvaluationSummary


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
    """Persistence contract the service layer depends on."""

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


def new_run_id() -> str:
    """Generate a short, URL-safe run identifier."""
    return uuid.uuid4().hex


class SqlAlchemyEvaluationRepository:
    """SQLAlchemy-backed implementation of :class:`AbstractEvaluationRepository`.

    Holds a single :class:`~sqlalchemy.orm.Session` (request-scoped in the API,
    or an in-memory session in tests). It commits its own units of work so the
    service layer stays free of transaction plumbing.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    # -- writes ------------------------------------------------------------ #

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

    # -- reads ------------------------------------------------------------- #

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

    # -- mapping ----------------------------------------------------------- #

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
