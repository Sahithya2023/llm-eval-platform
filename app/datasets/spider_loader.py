"""Spider dataset loading layer.

Standalone component: depends only on the Python standard library. It knows
nothing about FastAPI, SQLAlchemy, the evaluation logic, the LLM client, or the
runner. Later stages consume it; it never imports from them.

Expected on-disk layout (the official Spider release), rooted at ``spider_dir``::

    spider_dir/
        dev.json
        train_spider.json
        train_others.json        (optional)
        tables.json
        database/
            <db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import NamedTuple


class SpiderExample(NamedTuple):
    """One Spider example.

    A NamedTuple so callers can either unpack it
    (``example_id, db_id, question, gold_sql, db_path = ex``) or use attribute
    access (``ex.db_path``).
    """

    example_id: str
    db_id: str
    question: str
    gold_sql: str
    db_path: str


# Files that make up each split in the official release. train_others.json is
# optional; the first file in each list is required.
_SPLIT_FILES: dict[str, list[str]] = {
    "dev": ["dev.json"],
    "train": ["train_spider.json", "train_others.json"],
}


class SpiderLoader:
    """Loads Spider examples and resolves each ``db_id`` to its SQLite file."""

    def __init__(
        self,
        spider_dir: str | Path,
        split: str = "dev",
        limit: int | None = None,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Expected one of: {sorted(_SPLIT_FILES)}."
            )
        if limit is not None and limit <= 0:
            raise ValueError(
                f"limit must be a positive integer or None, got {limit!r}."
            )

        self.spider_dir = Path(spider_dir)
        self.split = split
        self.limit = limit
        self.database_dir = self.spider_dir / "database"

        if not self.spider_dir.exists():
            raise FileNotFoundError(
                f"Spider directory not found: {self.spider_dir}. "
                "Download Spider and extract it there (see README, Stage 2)."
            )
        if not self.database_dir.exists():
            raise FileNotFoundError(
                f"Spider 'database/' folder not found at: {self.database_dir}. "
                "The SQLite databases are required for db_id resolution."
            )

        self._records: list[dict] = self._load_records()

    def _load_records(self) -> list[dict]:
        records: list[dict] = []
        found_any = False
        for filename in _SPLIT_FILES[self.split]:
            path = self.spider_dir / filename
            if not path.exists():
                continue
            found_any = True
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse Spider file {path}: {exc}") from exc
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON list in {path}, got {type(data).__name__}."
                )
            records.extend(data)

        if not found_any:
            expected = ", ".join(_SPLIT_FILES[self.split])
            raise FileNotFoundError(
                f"No Spider files for split {self.split!r} found in {self.spider_dir}. "
                f"Expected at least: {expected}."
            )
        return records

    def __len__(self) -> int:
        if self.limit is None:
            return len(self._records)
        return min(self.limit, len(self._records))

    def __iter__(self) -> Iterator[SpiderExample]:
        for i, record in enumerate(islice(self._records, self.limit)):
            db_id = record["db_id"]
            yield SpiderExample(
                example_id=f"{self.split}_{i}",
                db_id=db_id,
                question=record["question"],
                gold_sql=record["query"],
                db_path=str(self.resolve_db_path(db_id)),
            )

    def resolve_db_path(self, db_id: str) -> Path:
        """Return the SQLite path for ``db_id``, or raise if it is missing."""
        path = self.database_dir / db_id / f"{db_id}.sqlite"
        if not path.exists():
            raise FileNotFoundError(
                f"SQLite database not found for db_id={db_id!r}: expected at {path}."
            )
        return path

    def unique_db_ids(self) -> list[str]:
        """Distinct db_ids across the (possibly limited) examples, in order."""
        seen: list[str] = []
        seen_set: set[str] = set()
        for record in islice(self._records, self.limit):
            db_id = record["db_id"]
            if db_id not in seen_set:
                seen_set.add(db_id)
                seen.append(db_id)
        return seen

    def missing_databases(self) -> list[tuple[str, str]]:
        """Return ``(db_id, expected_path)`` for every db_id whose file is absent.

        Non-raising — intended for reporting tools such as check_spider.py.
        """
        missing: list[tuple[str, str]] = []
        for db_id in self.unique_db_ids():
            path = self.database_dir / db_id / f"{db_id}.sqlite"
            if not path.exists():
                missing.append((db_id, str(path)))
        return missing
