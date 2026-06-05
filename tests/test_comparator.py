"""Unit tests for the result-set comparator (the execution-accuracy oracle).

These are hermetic: they operate on plain in-memory result sets (lists of
tuples), so no database and no Spider download are required.

Frozen Stage 3 semantics under test:
  1. Bag (multiset) comparison  -> row multiplicity matters.
  2. Strict positional columns  -> column order/count must match.
  3. Ordering from GOLD SQL only -> ORDER BY in gold makes order significant.
  4. Numeric normalization       -> 5 == 5.0, True == 1, float tolerance.
  5. Strict, exact strings        -> no case-folding, no trimming.
  6. NULL equals only NULL.
"""

from types import SimpleNamespace

import pytest

from app.eval.comparator import (
    Comparison,
    compare,
    compare_executions,
    compare_result_sets,
    gold_requires_order,
)


# --------------------------------------------------------------------------- #
# Core: ordering semantics                                                     #
# --------------------------------------------------------------------------- #

def test_identical_rows_unordered_is_correct():
    gold = [(1, "a"), (2, "b")]
    pred = [(1, "a"), (2, "b")]
    result = compare_result_sets(gold, pred, order_matters=False)
    assert result.is_correct is True
    assert result.order_considered is False


def test_reordered_rows_unordered_is_correct():
    gold = [(1, "a"), (2, "b")]
    pred = [(2, "b"), (1, "a")]
    assert compare_result_sets(gold, pred, order_matters=False).is_correct is True


def test_reordered_rows_ordered_is_incorrect():
    gold = [(1, "a"), (2, "b")]
    pred = [(2, "b"), (1, "a")]
    result = compare_result_sets(gold, pred, order_matters=True)
    assert result.is_correct is False
    assert result.order_considered is True


def test_same_order_ordered_is_correct():
    gold = [(1,), (2,), (3,)]
    pred = [(1,), (2,), (3,)]
    assert compare_result_sets(gold, pred, order_matters=True).is_correct is True


# --------------------------------------------------------------------------- #
# Core: multiset (bag) semantics                                               #
# --------------------------------------------------------------------------- #

def test_duplicate_rows_are_significant():
    # SELECT name  vs  SELECT DISTINCT name -> must NOT be equal.
    gold = [("x",), ("x",)]
    pred = [("x",)]
    assert compare_result_sets(gold, pred, order_matters=False).is_correct is False


def test_matching_multiset_with_duplicates_is_correct():
    gold = [("x",), ("x",), ("y",)]
    pred = [("y",), ("x",), ("x",)]
    assert compare_result_sets(gold, pred, order_matters=False).is_correct is True


# --------------------------------------------------------------------------- #
# Core: numeric normalization + float tolerance                                #
# --------------------------------------------------------------------------- #

def test_int_equals_float_same_value():
    assert compare_result_sets([(5,)], [(5.0,)], order_matters=False).is_correct is True


def test_float_tolerance_absorbs_rounding():
    # 0.1 + 0.2 == 0.30000000000000004 should match 0.3.
    assert compare_result_sets([(0.1 + 0.2,)], [(0.3,)], order_matters=False).is_correct is True


def test_bool_treated_as_int():
    assert compare_result_sets([(True,)], [(1,)], order_matters=False).is_correct is True
    assert compare_result_sets([(False,)], [(0,)], order_matters=False).is_correct is True


def test_distinct_numbers_are_incorrect():
    assert compare_result_sets([(5,)], [(6,)], order_matters=False).is_correct is False


# --------------------------------------------------------------------------- #
# Core: NULL semantics                                                         #
# --------------------------------------------------------------------------- #

def test_null_equals_null():
    assert compare_result_sets([(None,)], [(None,)], order_matters=False).is_correct is True


def test_null_does_not_equal_zero():
    assert compare_result_sets([(None,)], [(0,)], order_matters=False).is_correct is False


def test_null_does_not_equal_empty_string():
    assert compare_result_sets([(None,)], [("",)], order_matters=False).is_correct is False


# --------------------------------------------------------------------------- #
# Core: strict, exact string comparison                                        #
# --------------------------------------------------------------------------- #

def test_strings_are_case_sensitive():
    assert compare_result_sets([("M",)], [("m",)], order_matters=False).is_correct is False


def test_strings_are_whitespace_sensitive():
    assert compare_result_sets([("US",)], [("US ",)], order_matters=False).is_correct is False


# --------------------------------------------------------------------------- #
# Core: column count / shape                                                   #
# --------------------------------------------------------------------------- #

def test_column_count_mismatch_is_incorrect():
    result = compare_result_sets([(1, 2)], [(1,)], order_matters=False)
    assert result.is_correct is False
    assert "column" in result.reason.lower()


def test_strict_positional_columns():
    # Same values, swapped column positions -> incorrect under strict policy.
    gold = [(1, 2)]
    pred = [(2, 1)]
    assert compare_result_sets(gold, pred, order_matters=False).is_correct is False


def test_row_count_mismatch_is_incorrect():
    result = compare_result_sets([(1,), (2,)], [(1,)], order_matters=False)
    assert result.is_correct is False
    assert "row" in result.reason.lower()


# --------------------------------------------------------------------------- #
# Core: empty result sets                                                      #
# --------------------------------------------------------------------------- #

def test_both_empty_is_correct():
    assert compare_result_sets([], [], order_matters=False).is_correct is True
    assert compare_result_sets([], [], order_matters=True).is_correct is True


def test_empty_vs_nonempty_is_incorrect():
    assert compare_result_sets([], [(1,)], order_matters=False).is_correct is False
    assert compare_result_sets([(1,)], [], order_matters=False).is_correct is False


# --------------------------------------------------------------------------- #
# ORDER BY detection (from gold SQL only)                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT name FROM t", False),
        ("SELECT name FROM t ORDER BY age", True),
        ("select name from t order   by age desc", True),
        ("SELECT name FROM t ORDER BY age LIMIT 1", True),
        # ORDER BY only inside a subquery -> outer order is unspecified.
        ("SELECT a FROM t WHERE id IN (SELECT id FROM x ORDER BY y)", False),
        # 'order by' appearing inside a string literal must not count.
        ("SELECT 'order by' FROM t", False),
        ("", False),
    ],
)
def test_gold_requires_order(sql, expected):
    assert gold_requires_order(sql) is expected


def test_compare_uses_gold_order_by():
    # No ORDER BY in gold -> unordered -> reordered prediction is correct.
    assert compare("SELECT a FROM t", [(1,), (2,)], [(2,), (1,)]).is_correct is True
    # ORDER BY in gold -> ordered -> reordered prediction is incorrect.
    assert compare("SELECT a FROM t ORDER BY a", [(1,), (2,)], [(2,), (1,)]).is_correct is False


# --------------------------------------------------------------------------- #
# compare_executions: integrating with executor results (duck-typed)          #
# --------------------------------------------------------------------------- #

def _exec(rows=None, error=None, error_type=None):
    """Stand-in for executor.ExecutionResult (structural / duck-typed)."""
    return SimpleNamespace(rows=rows, error=error, error_type=error_type)


def test_compare_executions_both_ok_delegates():
    gold = _exec(rows=[(1,)])
    pred = _exec(rows=[(1,)])
    assert compare_executions(gold, pred, "SELECT a FROM t").is_correct is True


def test_compare_executions_prediction_failed_is_incorrect():
    gold = _exec(rows=[(1,)])
    pred = _exec(error="near \"SELET\": syntax error", error_type="sql_error")
    result = compare_executions(gold, pred, "SELECT a FROM t")
    assert result.is_correct is False
    assert "predict" in result.reason.lower()


def test_compare_executions_gold_failed_is_flagged():
    gold = _exec(error="no such table: t", error_type="sql_error")
    pred = _exec(rows=[(1,)])
    result = compare_executions(gold, pred, "SELECT a FROM t")
    assert result.is_correct is False
    assert "gold" in result.reason.lower()


def test_comparison_is_namedtuple_shape():
    result = compare_result_sets([(1,)], [(1,)], order_matters=False)
    assert isinstance(result, Comparison)
    assert hasattr(result, "is_correct")
    assert hasattr(result, "reason")
    assert hasattr(result, "order_considered")
