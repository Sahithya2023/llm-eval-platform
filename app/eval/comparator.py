"""Result-set comparator — the execution-accuracy oracle.

Standalone component: depends only on the Python standard library. It knows
nothing about FastAPI, SQLAlchemy, Streamlit, the LLM client, the executor, or
the runner. It judges correctness from *execution results*, never from SQL
strings. Two different SQL queries that produce the same result set are
considered equivalent.

Frozen Stage 3 semantics
------------------------
1. **Bag (multiset) comparison** — row multiplicity is significant, so
   ``SELECT name`` and ``SELECT DISTINCT name`` are correctly distinguished.
2. **Strict positional columns** — column count and order must match; a
   prediction that selects the same columns in a different order is incorrect.
3. **Ordering from the GOLD query only** — rows are compared as an ordered
   sequence iff the gold SQL has a top-level ``ORDER BY``; otherwise they are
   compared as an unordered multiset. The prediction's own ``ORDER BY`` does
   not change the contract.
4. **Numeric normalization** — ``int`` / ``float`` / ``Decimal`` are compared
   by value (``5 == 5.0``), ``bool`` is treated as ``int`` (``True == 1``), and
   floats are canonicalised to a fixed number of decimals so that rounding
   differences (e.g. ``0.1 + 0.2`` vs ``0.3``) compare equal.
5. **Strict, exact strings** — no case-folding and no trimming
   (``"M" != "m"``, ``"US" != "US "``).
6. **NULL equals only NULL** — ``None`` matches ``None`` and nothing else.

Known, documented limitations
------------------------------
* Float canonicalisation rounds to ``_ROUND_DECIMALS`` places (absolute), which
  is the practical equivalent of a small tolerance for Spider's numeric
  outputs. Extremely large integers may lose precision when normalised through
  ``float``; Spider does not exercise this.
* ``ORDER BY`` detection is a pragmatic textual heuristic (string literals and
  parenthesised sub-queries are stripped before searching). It is not a full
  SQL parser, but covers the Spider gold-query forms.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal
from typing import Any, NamedTuple, Sequence

# Number of decimal places used to canonicalise floats. Absorbs rounding-path
# differences between equivalent aggregate queries (AVG/SUM, etc.).
_ROUND_DECIMALS = 6

# Sentinel canonical token for SQL NULL. A tuple so it can never collide with a
# normalised string/number token.
_NULL_TOKEN = ("\x00null",)

Row = Sequence[Any]
ResultSet = Sequence[Row]


class Comparison(NamedTuple):
    """Verdict produced by the oracle.

    ``reason`` is a short human-readable explanation, suitable for logging into
    a trace and for the Phase-2 failure-analysis stage.
    """

    is_correct: bool
    reason: str
    order_considered: bool


# --------------------------------------------------------------------------- #
# Value / row canonicalisation                                                 #
# --------------------------------------------------------------------------- #

def _canonical(value: Any) -> Any:
    """Map a raw SQLite value to a hashable, type-normalised token.

    The token is what the comparison actually operates on, so all of the
    normalisation rules (numeric, bool, NULL, string, bytes) live here.
    """
    if value is None:
        return _NULL_TOKEN
    # bool is a subclass of int, so it must be checked first.
    if isinstance(value, bool):
        return ("num", round(float(int(value)), _ROUND_DECIMALS))
    if isinstance(value, (int, float)):
        return ("num", round(float(value), _ROUND_DECIMALS))
    if isinstance(value, Decimal):
        return ("num", round(float(value), _ROUND_DECIMALS))
    if isinstance(value, str):
        return ("str", value)  # strict: no case-fold, no strip
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    # Fallback for any exotic adapter output; compared by repr.
    return ("other", repr(value))


def _canonical_row(row: Row) -> tuple:
    """Canonicalise a row, preserving column order (strict positional)."""
    return tuple(_canonical(v) for v in row)


def _canonicalize(result_set: ResultSet) -> list[tuple]:
    return [_canonical_row(row) for row in result_set]


# --------------------------------------------------------------------------- #
# ORDER BY detection (gold SQL only)                                           #
# --------------------------------------------------------------------------- #

_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_INNERMOST_PAREN_RE = re.compile(r"\([^()]*\)")
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def gold_requires_order(gold_sql: str) -> bool:
    """Return ``True`` iff the gold SQL has a top-level ``ORDER BY``.

    String literals are removed first (so ``'order by'`` text does not count),
    then parenthesised sub-queries are stripped iteratively (so an ``ORDER BY``
    that only appears inside a sub-query does not make the outer result
    order-significant).
    """
    if not gold_sql:
        return False
    stripped = _STRING_LITERAL_RE.sub(" ", gold_sql)
    previous = None
    while previous != stripped:
        previous = stripped
        stripped = _INNERMOST_PAREN_RE.sub(" ", stripped)
    return bool(_ORDER_BY_RE.search(stripped))


# --------------------------------------------------------------------------- #
# Comparison                                                                   #
# --------------------------------------------------------------------------- #

def compare_result_sets(
    gold_rows: ResultSet,
    pred_rows: ResultSet,
    order_matters: bool,
) -> Comparison:
    """Compare two already-materialised result sets.

    This is the pure core of the oracle: it has no knowledge of SQL. Callers
    that have the gold SQL should prefer :func:`compare`, which derives
    ``order_matters`` for them.
    """
    gold = _canonicalize(gold_rows)
    pred = _canonicalize(pred_rows)

    gold_empty = len(gold) == 0
    pred_empty = len(pred) == 0
    if gold_empty and pred_empty:
        return Comparison(True, "both result sets are empty", order_matters)
    if gold_empty != pred_empty:
        return Comparison(
            False,
            f"row-count mismatch: gold={len(gold)} pred={len(pred)}",
            order_matters,
        )

    # Strict positional columns: arity must match.
    gold_arity = len(gold[0])
    pred_arity = len(pred[0])
    if gold_arity != pred_arity:
        return Comparison(
            False,
            f"column-count mismatch: gold={gold_arity} pred={pred_arity}",
            order_matters,
        )

    if len(gold) != len(pred):
        return Comparison(
            False,
            f"row-count mismatch: gold={len(gold)} pred={len(pred)}",
            order_matters,
        )

    if order_matters:
        for i, (g_row, p_row) in enumerate(zip(gold, pred)):
            if g_row != p_row:
                return Comparison(
                    False,
                    f"row mismatch at position {i} (ordered comparison)",
                    True,
                )
        return Comparison(True, f"exact ordered match, {len(gold)} row(s)", True)

    # Unordered: bag/multiset equality (row multiplicity preserved).
    if Counter(gold) == Counter(pred):
        return Comparison(
            True,
            f"exact multiset match, {len(gold)} row(s), unordered",
            False,
        )
    return Comparison(False, "multiset mismatch (unordered comparison)", False)


def compare(gold_sql: str, gold_rows: ResultSet, pred_rows: ResultSet) -> Comparison:
    """Compare result sets, deriving ordering significance from the gold SQL."""
    order_matters = gold_requires_order(gold_sql)
    return compare_result_sets(gold_rows, pred_rows, order_matters)


def compare_executions(gold_exec: Any, pred_exec: Any, gold_sql: str) -> Comparison:
    """Compare two execution results (e.g. ``executor.ExecutionResult``).

    Duck-typed on purpose so the comparator never imports the executor: any
    object exposing ``rows``, ``error`` and ``error_type`` attributes works.

    * If the prediction failed to execute -> incorrect, with the error noted.
    * If the gold query itself failed -> incorrect, flagged as a bad benchmark
      row (the prediction cannot be judged against a broken oracle).
    * Otherwise the two result sets are compared via :func:`compare`.
    """
    order_matters = gold_requires_order(gold_sql)

    if getattr(gold_exec, "error", None) is not None:
        return Comparison(
            False,
            f"gold query failed to execute "
            f"[{getattr(gold_exec, 'error_type', 'error')}]: {gold_exec.error}",
            order_matters,
        )
    if getattr(pred_exec, "error", None) is not None:
        return Comparison(
            False,
            f"prediction failed to execute "
            f"[{getattr(pred_exec, 'error_type', 'error')}]: {pred_exec.error}",
            order_matters,
        )

    return compare_result_sets(gold_exec.rows, pred_exec.rows, order_matters)
