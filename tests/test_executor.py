"""Unit tests for the read-only SQL executor.

Hermetic: each test builds a tiny SQLite database in a temp dir, so no real
Spider download is needed. Writing the fixture uses sqlite3 directly; the
*executor* is what must be read-only, not the test setup.
"""

import sqlite3
from pathlib import Path

import pytest

from app.eval.executor import (
    DEFAULT_MAX_ROWS,
    ExecutionResult,
    SqlExecutor,
)


def _make_db(tmp_path: Path, n_rows: int = 3) -> Path:
    """Create a small SQLite db with a `people` table and return its path."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO people (id, name, age) VALUES (?, ?, ?)",
        [(i, f"person_{i}", 20 + i) for i in range(1, n_rows + 1)],
    )
    conn.commit()
    conn.close()
    return db_path


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #

def test_select_returns_rows_and_columns(tmp_path):
    db = _make_db(tmp_path, n_rows=3)
    result = SqlExecutor().execute(db, "SELECT id, name FROM people ORDER BY id")
    assert result.ok is True
    assert result.error is None
    assert result.columns == ("id", "name")
    assert result.rows == [(1, "person_1"), (2, "person_2"), (3, "person_3")]
    assert result.truncated is False


def test_aggregate_query(tmp_path):
    db = _make_db(tmp_path, n_rows=5)
    result = SqlExecutor().execute(db, "SELECT count(*) FROM people")
    assert result.ok is True
    assert result.rows == [(5,)]


def test_empty_result_set(tmp_path):
    db = _make_db(tmp_path, n_rows=2)
    result = SqlExecutor().execute(db, "SELECT id FROM people WHERE age > 999")
    assert result.ok is True
    assert result.rows == []


# --------------------------------------------------------------------------- #
# Read-only enforcement                                                        #
# --------------------------------------------------------------------------- #

def test_insert_is_blocked(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "INSERT INTO people (id, name, age) VALUES (99, 'x', 1)")
    assert result.ok is False
    assert result.error_type == "sql_error"


def test_create_table_is_blocked(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "CREATE TABLE evil (x INTEGER)")
    assert result.ok is False
    assert result.error_type == "sql_error"


def test_update_is_blocked(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "UPDATE people SET age = 0")
    assert result.ok is False
    assert result.error_type == "sql_error"


def test_readonly_does_not_mutate_db(tmp_path):
    db = _make_db(tmp_path, n_rows=3)
    SqlExecutor().execute(db, "DELETE FROM people")  # blocked, ignored
    # Re-open independently and confirm the data survived untouched.
    conn = sqlite3.connect(db)
    (count,) = conn.execute("SELECT count(*) FROM people").fetchone()
    conn.close()
    assert count == 3


# --------------------------------------------------------------------------- #
# Error handling                                                               #
# --------------------------------------------------------------------------- #

def test_syntax_error_is_reported(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "SELET * FROM people")
    assert result.ok is False
    assert result.error_type == "sql_error"
    assert result.rows is None


def test_missing_table_is_reported(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "SELECT * FROM no_such_table")
    assert result.ok is False
    assert result.error_type == "sql_error"


def test_missing_database_file_is_connection_error(tmp_path):
    missing = tmp_path / "does_not_exist.sqlite"
    result = SqlExecutor().execute(missing, "SELECT 1")
    assert result.ok is False
    assert result.error_type == "connection_error"


# --------------------------------------------------------------------------- #
# Bounds: timeout + row cap                                                    #
# --------------------------------------------------------------------------- #

def test_timeout_aborts_runaway_query(tmp_path):
    db = _make_db(tmp_path)
    # Infinite recursive CTE: spins forever until the wall-clock guard fires.
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )
    result = SqlExecutor(timeout_s=0.2).execute(db, runaway)
    assert result.ok is False
    assert result.error_type == "timeout"


def test_max_rows_truncates(tmp_path):
    db = _make_db(tmp_path, n_rows=50)
    result = SqlExecutor(max_rows=10).execute(db, "SELECT id FROM people")
    assert result.ok is True
    assert result.truncated is True
    assert len(result.rows) == 10


def test_default_max_rows_constant_is_sane():
    assert DEFAULT_MAX_ROWS >= 1000


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        SqlExecutor(timeout_s=0)
    with pytest.raises(ValueError):
        SqlExecutor(max_rows=0)


def test_execution_result_is_namedtuple_shape(tmp_path):
    db = _make_db(tmp_path)
    result = SqlExecutor().execute(db, "SELECT 1")
    assert isinstance(result, ExecutionResult)
    for field in ("rows", "columns", "error", "error_type", "truncated"):
        assert hasattr(result, field)
