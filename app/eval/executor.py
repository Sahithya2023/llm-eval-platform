"""Read-only SQL executor.

Standalone component: depends only on the Python standard library (``sqlite3``).
It is independent of FastAPI, SQLAlchemy, Streamlit, the LLM client, the
comparator, and the runner.

Guarantees
----------
* **Read-only** — the database is opened in SQLite read-only URI mode
  (``mode=ro``), so any attempt to write (INSERT/UPDATE/DELETE/CREATE/DROP/
  ATTACH) fails at the engine level. ``PRAGMA query_only`` is set as a
  secondary guard.
* **Bounded** — a wall-clock timeout aborts long-running queries via a SQLite
  progress handler, and a row cap guards against runaway result sets.
* **Total** — query problems never raise out of :meth:`SqlExecutor.execute`;
  they are returned as a structured :class:`ExecutionResult` with an
  ``error``/``error_type`` so callers can record the failure in a trace.

The executor deliberately returns *raw* SQLite values (``int``, ``float``,
``str``, ``bytes``, ``None``). All normalisation and equality logic lives in the
comparator, keeping the execution/judgement boundary clean.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, NamedTuple

# Defaults chosen for Spider's small per-database SQLite files.
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_ROWS = 100_000

# How often (in SQLite virtual-machine instructions) the timeout guard runs.
_PROGRESS_INSTRUCTIONS = 1000


class ExecutionResult(NamedTuple):
    """Outcome of executing one SQL statement.

    On success: ``rows`` and ``columns`` are populated, ``error`` is ``None``.
    On failure: ``rows``/``columns`` are ``None`` and ``error``/``error_type``
    describe the problem. ``error_type`` is one of ``"sql_error"``,
    ``"timeout"`` or ``"connection_error"``.
    """

    rows: list[tuple[Any, ...]] | None
    columns: tuple[str, ...] | None
    error: str | None
    error_type: str | None
    truncated: bool

    @property
    def ok(self) -> bool:
        return self.error is None


class SqlExecutor:
    """Executes SQL read-only against a SQLite file, with time/row bounds."""

    def __init__(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}.")
        if max_rows <= 0:
            raise ValueError(f"max_rows must be a positive integer, got {max_rows!r}.")
        self.timeout_s = float(timeout_s)
        self.max_rows = int(max_rows)

    def execute(self, db_path: str | Path, sql: str) -> ExecutionResult:
        """Run ``sql`` against the SQLite database at ``db_path`` (read-only)."""
        db_path = Path(db_path)
        if not db_path.exists():
            return ExecutionResult(
                None, None, f"database file not found: {db_path}", "connection_error", False
            )

        # Read-only at the file-open level: writes raise OperationalError.
        uri = f"file:{db_path.as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
        except sqlite3.Error as exc:
            return ExecutionResult(
                None, None, f"could not open database: {exc}", "connection_error", False
            )

        try:
            # Secondary guard; harmless if it is a no-op on this build.
            try:
                conn.execute("PRAGMA query_only = ON;")
            except sqlite3.Error:
                pass

            deadline = time.monotonic() + self.timeout_s
            # Returning non-zero from the progress handler aborts the query,
            # which surfaces as sqlite3.OperationalError("interrupted").
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0,
                _PROGRESS_INSTRUCTIONS,
            )

            cur = conn.cursor()
            try:
                cur.execute(sql)
            except sqlite3.OperationalError as exc:
                return self._operational_error(exc, deadline)
            except sqlite3.Error as exc:
                return ExecutionResult(None, None, str(exc), "sql_error", False)

            columns = (
                tuple(d[0] for d in cur.description) if cur.description else ()
            )

            rows: list[tuple[Any, ...]] = []
            truncated = False
            try:
                while True:
                    batch = cur.fetchmany(1000)
                    if not batch:
                        break
                    for raw in batch:
                        if len(rows) >= self.max_rows:
                            truncated = True
                            break
                        rows.append(tuple(raw))
                    if truncated:
                        break
                    if time.monotonic() > deadline:
                        return ExecutionResult(
                            None, None,
                            f"query timed out after {self.timeout_s}s",
                            "timeout", False,
                        )
            except sqlite3.OperationalError as exc:
                return self._operational_error(exc, deadline)
            except sqlite3.Error as exc:
                return ExecutionResult(None, None, str(exc), "sql_error", False)

            return ExecutionResult(rows, columns, None, None, truncated)
        finally:
            conn.close()

    def _operational_error(
        self, exc: sqlite3.OperationalError, deadline: float
    ) -> ExecutionResult:
        """Classify an OperationalError as a timeout abort or a plain SQL error."""
        if time.monotonic() > deadline:
            return ExecutionResult(
                None, None, f"query timed out after {self.timeout_s}s", "timeout", False
            )
        return ExecutionResult(None, None, str(exc), "sql_error", False)
