"""Stage 6 + 7 persistence models — evaluation runs and per-example records.

Stage 6 added two dedicated tables (``evaluation_runs`` / ``evaluation_records``)
that map 1:1 onto the Stage 5 NamedTuples.  Stage 7 (Phase 1) extends
``evaluation_runs`` with a proper job lifecycle:

* ``status``       — pending | running | completed | failed   (default: pending)
* ``started_at``   — set when the worker begins execution
* ``finished_at``  — set when the run reaches a terminal state
* ``error``        — populated only on ``failed`` runs; stores the error message

``started_at`` / ``finished_at`` / ``error`` are all nullable so that the
schema is additive: rows created by the Stage 6 path (``create_run``) that jump
straight to ``completed`` simply leave those fields ``NULL``.

ORM classes use a ``*Model`` suffix so they never shadow the Stage 5
``EvaluationRecord`` / ``EvaluationSummary`` NamedTuples or the Pydantic schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Share the declarative base so every model lives in one metadata and is
# created together by Base.metadata.create_all (used by init_db).
from app.db.models import Base


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Used as the ORM ``default`` callable for ``DateTime`` columns, replacing the
    deprecated ``datetime.utcnow`` (which returns a naive datetime and is
    scheduled for removal in a future Python version).
    """
    return datetime.now(timezone.utc)


# Valid lifecycle states for an evaluation run.
# Kept as module-level constants so other modules can reference them without
# importing magic strings.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class EvaluationRunModel(Base):
    """One evaluation run: configuration/metadata plus the aggregate summary.

    Lifecycle
    ---------
    pending   → the run row has been created but execution has not started.
    running   → a worker has claimed the run and evaluation is in progress.
    completed → execution finished successfully; summary columns are populated.
    failed    → a run-level infrastructure error occurred (dataset load, DB,
                unexpected exception).  Per-example errors do *not* trigger
                ``failed``; those are captured in ``EvaluationRecordModel``
                and the run still completes normally.

    The summary columns mirror :class:`~app.services.evaluation_runner.EvaluationSummary`
    exactly so a stored run can be returned through the API without recomputation.
    Summary columns default to 0 / 0.0 and are meaningful only when
    ``status == "completed"``.
    """

    __tablename__ = "evaluation_runs"

    # Primary key is a uuid hex string to match the existing Trace.run_id
    # convention and to be URL-safe without encoding.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # --- run metadata / configuration ---
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset: Mapped[str] = mapped_column(String(64), default="spider")
    split: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # --- lifecycle ---
    # Default is now "pending"; Stage 6's create_run passes status="completed"
    # explicitly so existing behaviour is unchanged.
    status: Mapped[str] = mapped_column(String(32), default=STATUS_PENDING)
    # Timestamps for the execution window (NULL until the relevant transition).
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Non-NULL only on failed runs; stores the exception / error message.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- aggregate summary (1:1 with EvaluationSummary) ---
    # All default to 0 / 0.0; populated atomically when status → completed.
    total_examples: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    incorrect: Mapped[int] = mapped_column(Integer, default=0)
    failed_generations: Mapped[int] = mapped_column(Integer, default=0)
    failed_executions: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    total_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    average_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # --- bookkeeping ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now(), index=True
    )

    records: Mapped[list["EvaluationRecordModel"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="EvaluationRecordModel.id",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationRunModel id={self.id!r} status={self.status!r} "
            f"accuracy={self.accuracy} total={self.total_examples}>"
        )


class EvaluationRecordModel(Base):
    """One persisted per-example outcome (1:1 with :class:`~app.services.evaluation_runner.EvaluationRecord`)."""

    __tablename__ = "evaluation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )

    # --- the task / outcome (mirror of EvaluationRecord) ---
    example_id: Mapped[str] = mapped_column(String(128))
    db_id: Mapped[str] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    gold_sql: Mapped[str] = mapped_column(Text)
    generated_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    comparison_reason: Mapped[str] = mapped_column(Text, default="")

    # --- model + cost/perf (mirror of EvaluationRecord) ---
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now()
    )

    run: Mapped["EvaluationRunModel"] = relationship(back_populates="records")

    def __repr__(self) -> str:
        return (
            f"<EvaluationRecordModel id={self.id} run_id={self.run_id!r} "
            f"is_correct={self.is_correct}>"
        )