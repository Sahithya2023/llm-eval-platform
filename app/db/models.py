"""Database models.

A single `traces` table backs the whole platform. Stage 1 only creates it;
later stages populate it. Fields that are filled in by later stages are
nullable so the schema is stable from day one and never needs migrating.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by every model."""


class Trace(Base):
    """One evaluation example: a question, the generated SQL, and the verdict.

    Aggregate metrics (accuracy, latency, cost) are computed by querying over
    rows of this table, so no separate metrics table is required.
    """

    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- experiment grouping ---
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    example_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    db_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # --- the task ---
    question: Mapped[str] = mapped_column(Text)
    gold_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_sql: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- raw result sets (JSON-encoded; consumed by failure analysis later) ---
    gold_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- execution-accuracy verdict ---
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- model + reproducibility ---
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # --- cost + performance ---
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- bookkeeping ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<Trace id={self.id} run_id={self.run_id!r} "
            f"is_correct={self.is_correct}>"
        )
