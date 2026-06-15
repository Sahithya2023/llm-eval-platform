"""API schemas (Stage 6 + 7) — request, response, and validation models.

Stage 7 (Phase 1) adds three nullable lifecycle fields to ``RunResponse``:

    started_at   — when the worker began execution (None while pending)
    finished_at  — when the run reached a terminal state (None while pending/running)
    error        — non-None only on failed runs

All three are optional in the response so that Stage 6-created runs (which never
set those columns) and Stage 7 pending/running runs deserialise without error.

No other contract changes; all Stage 6 request shapes are preserved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Requests / validation                                                        #
# --------------------------------------------------------------------------- #

class RunRequest(BaseModel):
    """Request body for starting an evaluation run.

    Validation lives here: ``split`` is constrained to known Spider splits and
    ``limit`` must be a positive integer when supplied (mirrors the loader).
    """

    model_config = ConfigDict(extra="forbid")

    split: Literal["dev", "train"] = "dev"
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Cap on examples to evaluate; None means all.",
    )
    model: str | None = Field(default=None, description="Override LLM model name.")
    prompt_version: str | None = Field(default=None)
    name: str | None = Field(default=None, max_length=255)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Responses                                                                    #
# --------------------------------------------------------------------------- #

class SummaryResponse(BaseModel):
    """Aggregate metrics for a run (mirror of EvaluationSummary)."""

    model_config = ConfigDict(from_attributes=True)

    total_examples: int
    correct: int
    incorrect: int
    failed_generations: int
    failed_executions: int
    accuracy: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost: float
    average_latency_ms: float


class RunResponse(BaseModel):
    """A persisted run: metadata plus its aggregate summary.

    ``started_at``, ``finished_at``, and ``error`` are new in Stage 7 and
    are always optional so the schema is backwards-compatible with Stage 6
    run rows that have NULL in those columns.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str | None
    notes: str | None
    dataset: str
    split: str | None
    dataset_limit: int | None
    model: str | None
    prompt_version: str | None
    status: str
    created_at: datetime

    # Stage 7 lifecycle timestamps and error (all nullable).
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    # Summary fields are flattened from the same ORM row.
    # Values are 0 / 0.0 while the run is pending or running.
    total_examples: int
    correct: int
    incorrect: int
    failed_generations: int
    failed_executions: int
    accuracy: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost: float
    average_latency_ms: float


class RunListResponse(BaseModel):
    """A page of historical runs."""

    runs: list[RunResponse]
    count: int


class RecordResponse(BaseModel):
    """A single per-example outcome (mirror of EvaluationRecord)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    example_id: str
    db_id: str
    question: str
    gold_sql: str
    generated_sql: str | None
    is_correct: bool
    generation_error: str | None
    execution_error: str | None
    comparison_reason: str
    model: str | None
    prompt_version: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None
    latency_ms: int | None


class RecordListResponse(BaseModel):
    """All per-example records belonging to one run."""

    run_id: str
    records: list[RecordResponse]
    count: int