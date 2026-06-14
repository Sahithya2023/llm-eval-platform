"""Stage 6 persistence models — evaluation runs and per-example records.

Stage 1 introduced a single ``traces`` table (the :class:`~app.db.models.Trace`
model). Stage 6 needs to persist two *new* shapes that arrive straight from the
frozen Stage 5 runner:

* an :class:`~app.services.evaluation_runner.EvaluationSummary` (run-level
  metadata + aggregate metrics), and
* a list of :class:`~app.services.evaluation_runner.EvaluationRecord`
  (per-example outcomes, including the three distinct error/reason fields the
  ``traces`` table cannot represent losslessly).

Rather than overload ``traces`` (and risk a lossy mapping), Stage 6 adds two
dedicated tables that map 1:1 onto those NamedTuples. They are registered on the
*same* declarative ``Base`` as ``Trace``, so ``init_db`` creates them with no
change to the Stage 1 init script. ``Trace`` is left exactly as-is.

ORM classes use a ``*Model`` suffix so they never shadow the Stage 5
``EvaluationRecord`` / ``EvaluationSummary`` NamedTuples or the Pydantic schemas.
"""

from __future__ import annotations

from datetime import datetime

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

# Share the Stage 1 declarative base so every model lives in one metadata and
# is created together by Base.metadata.create_all (used by init_db).
from app.db.models import Base


class EvaluationRunModel(Base):
    """One evaluation run: configuration/metadata plus the aggregate summary.

    The summary columns mirror :class:`EvaluationSummary` exactly so a stored
    run can be returned through the API without recomputation.
    """

    __tablename__ = "evaluation_runs"

    # run_id is a string (uuid hex) to match the Trace.run_id convention.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # --- run metadata / configuration ---
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset: Mapped[str] = mapped_column(String(64), default="spider")
    split: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")

    # --- aggregate summary (1:1 with EvaluationSummary) ---
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
        DateTime, default=datetime.utcnow, server_default=func.now(), index=True
    )

    records: Mapped[list["EvaluationRecordModel"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="EvaluationRecordModel.id",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationRunModel id={self.id!r} "
            f"accuracy={self.accuracy} total={self.total_examples}>"
        )


class EvaluationRecordModel(Base):
    """One persisted per-example outcome (1:1 with :class:`EvaluationRecord`)."""

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
        DateTime, default=datetime.utcnow, server_default=func.now()
    )

    run: Mapped["EvaluationRunModel"] = relationship(back_populates="records")

    def __repr__(self) -> str:
        return (
            f"<EvaluationRecordModel id={self.id} run_id={self.run_id!r} "
            f"is_correct={self.is_correct}>"
        )
