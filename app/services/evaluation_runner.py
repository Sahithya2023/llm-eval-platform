"""Evaluation runner (Stage 5) — the orchestration layer.

This is the glue that turns the standalone Stage 2-4 components into an actual
evaluation. For one example it:

    schema_provider(db_path)  ->  schema context
    TextToSqlClient           ->  generated SQL (+ tokens/cost/latency)
    SqlExecutor               ->  run predicted and gold SQL
    compare_executions        ->  execution-accuracy verdict

and packages the outcome as an :class:`EvaluationRecord`. A batch run also
produces an aggregate :class:`EvaluationSummary`.

Scope (Stage 5): orchestration only. It does **not** persist anything, expose
an API, or render a UI — and it imports no web-framework, ORM, UI, or SDK code
directly. Every collaborator is injectable so the runner can be tested
hermetically; sensible Stage 3-4 defaults are used when none are supplied.

Like the layers it coordinates, the runner never raises for ordinary model or
execution failures: those become structured records with the relevant error
field populated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple

from app.eval import SqlExecutor, TextToSqlClient, compare_executions

if TYPE_CHECKING:  # typing only; the runner duck-types on these at runtime
    from app.datasets.spider_loader import SpiderExample


SchemaProvider = Callable[[Path], str]


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #

class EvaluationRecord(NamedTuple):
    """The outcome of evaluating a single example."""

    example_id: str
    db_id: str
    question: str
    gold_sql: str
    generated_sql: str | None
    is_correct: bool
    generation_error: str | None
    execution_error: str | None
    comparison_reason: str
    model: str
    prompt_version: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None
    latency_ms: int


class EvaluationSummary(NamedTuple):
    """Aggregate statistics over a batch of :class:`EvaluationRecord`."""

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


# --------------------------------------------------------------------------- #
# Default schema provider                                                      #
# --------------------------------------------------------------------------- #

def _extract_schema(db_path: Path) -> str:
    """Build a schema string from a SQLite file via read-only introspection.

    Reads the ``CREATE TABLE`` statements out of ``sqlite_master``. Opened in
    read-only URI mode and total: on any problem it returns an empty string
    rather than raising, so a missing or unreadable database degrades to "no
    schema context" instead of crashing a run.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return ""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            )
            statements = [row[0] for row in cur.fetchall() if row[0]]
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    return "\n\n".join(statements)


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

class EvaluationRunner:
    """Coordinates generation, execution and comparison for Text-to-SQL examples.

    All collaborators are injectable for hermetic testing; when omitted, the
    standard Stage 3-4 implementations are used. The runner constructs no real
    LLM client unless one is actually needed (i.e. unless ``llm_client`` is
    omitted), so building a runner with injected fakes never touches the SDK or
    configuration.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        sql_executor: Any | None = None,
        schema_provider: SchemaProvider | None = None,
        comparator: Callable[[Any, Any, str], Any] | None = None,
    ) -> None:
        self._llm_client = llm_client if llm_client is not None else TextToSqlClient()
        self._sql_executor = sql_executor if sql_executor is not None else SqlExecutor()
        self._schema_provider = (
            schema_provider if schema_provider is not None else _extract_schema
        )
        self._comparator = comparator if comparator is not None else compare_executions

    # -- single example ---------------------------------------------------- #

    def run_example(self, example: SpiderExample) -> EvaluationRecord:
        """Evaluate one example. Never raises for ordinary failures."""
        db_path = Path(example.db_path)

        # 1. Schema context. Degrade to "" rather than fail the whole example.
        try:
            schema_context = self._schema_provider(db_path)
        except Exception:
            schema_context = ""

        # 2. Generate SQL (the client itself never raises).
        generation = self._llm_client.generate_sql(example.question, schema_context)

        # 3. Generation failure -> skip execution, return a structured record.
        if generation.error is not None or generation.sql is None:
            return self._record(
                example,
                generated_sql=generation.sql,
                is_correct=False,
                generation_error=generation.error or "generation produced no SQL",
                execution_error=None,
                comparison_reason="generation failed; execution skipped",
                generation=generation,
            )

        # 4-6. Execute predicted + gold, then compare.
        try:
            pred_exec = self._sql_executor.execute(db_path, generation.sql)
            gold_exec = self._sql_executor.execute(db_path, example.gold_sql)
            comparison = self._comparator(gold_exec, pred_exec, example.gold_sql)
            execution_error = getattr(pred_exec, "error", None)
            is_correct = bool(getattr(comparison, "is_correct", False))
            reason = getattr(comparison, "reason", "")
        except Exception as exc:  # defensive: ordinary failures never raise
            return self._record(
                example,
                generated_sql=generation.sql,
                is_correct=False,
                generation_error=None,
                execution_error=f"unexpected error: {_format_error(exc)}",
                comparison_reason="evaluation error",
                generation=generation,
            )

        return self._record(
            example,
            generated_sql=generation.sql,
            is_correct=is_correct,
            generation_error=None,
            execution_error=execution_error,
            comparison_reason=reason,
            generation=generation,
        )

    # -- batch ------------------------------------------------------------- #

    def run_examples(
        self, examples: Iterable[SpiderExample]
    ) -> tuple[list[EvaluationRecord], EvaluationSummary]:
        """Evaluate many examples and return the records plus a summary."""
        records = [self.run_example(example) for example in examples]
        return records, self._summarize(records)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _record(
        example: SpiderExample,
        *,
        generated_sql: str | None,
        is_correct: bool,
        generation_error: str | None,
        execution_error: str | None,
        comparison_reason: str,
        generation: Any,
    ) -> EvaluationRecord:
        """Assemble an EvaluationRecord, carrying metadata from the generation."""
        return EvaluationRecord(
            example_id=example.example_id,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            generated_sql=generated_sql,
            is_correct=is_correct,
            generation_error=generation_error,
            execution_error=execution_error,
            comparison_reason=comparison_reason,
            model=generation.model,
            prompt_version=generation.prompt_version,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
            estimated_cost=generation.estimated_cost,
            latency_ms=generation.latency_ms,
        )

    @staticmethod
    def _summarize(records: list[EvaluationRecord]) -> EvaluationSummary:
        total = len(records)
        correct = sum(1 for r in records if r.is_correct)
        failed_generations = sum(1 for r in records if r.generation_error is not None)
        failed_executions = sum(1 for r in records if r.execution_error is not None)

        total_prompt_tokens = sum(r.prompt_tokens or 0 for r in records)
        total_completion_tokens = sum(r.completion_tokens or 0 for r in records)
        total_tokens = sum(r.total_tokens or 0 for r in records)
        total_estimated_cost = sum(r.estimated_cost or 0.0 for r in records)

        accuracy = (correct / total) if total else 0.0
        average_latency_ms = (
            sum(r.latency_ms or 0 for r in records) / total if total else 0.0
        )

        return EvaluationSummary(
            total_examples=total,
            correct=correct,
            incorrect=total - correct,
            failed_generations=failed_generations,
            failed_executions=failed_executions,
            accuracy=accuracy,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_estimated_cost=total_estimated_cost,
            average_latency_ms=average_latency_ms,
        )
