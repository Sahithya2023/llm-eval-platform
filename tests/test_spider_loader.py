"""Unit tests for the Spider loader.

These are hermetic: they build a tiny synthetic Spider layout in a temp dir, so
no real Spider download is needed to run them.
"""

import json
from pathlib import Path

import pytest

from app.datasets.spider_loader import SpiderExample, SpiderLoader

SAMPLE = [
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
        "query": "SELECT count(*) FROM singer",
    },
    {
        "db_id": "concert_singer",
        "question": "List all singer names.",
        "query": "SELECT name FROM singer",
    },
    {
        "db_id": "pets_1",
        "question": "How many pets are there?",
        "query": "SELECT count(*) FROM pets",
    },
]


def _make_spider(tmp_path: Path, examples, db_ids_with_files) -> Path:
    """Create a minimal valid Spider layout and return its root dir."""
    spider = tmp_path / "spider"
    (spider / "database").mkdir(parents=True)
    (spider / "dev.json").write_text(json.dumps(examples), encoding="utf-8")
    for db_id in db_ids_with_files:
        db_dir = spider / "database" / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        (db_dir / f"{db_id}.sqlite").write_bytes(b"")  # existence is all the loader checks
    return spider


def test_loads_examples(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer", "pets_1"])
    items = list(SpiderLoader(spider, split="dev"))

    assert len(items) == 3
    assert all(isinstance(x, SpiderExample) for x in items)

    first = items[0]
    assert first.example_id == "dev_0"
    assert first.db_id == "concert_singer"
    assert first.question == "How many singers are there?"
    assert first.gold_sql == "SELECT count(*) FROM singer"
    assert first.db_path.endswith("concert_singer.sqlite")


def test_limit(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer", "pets_1"])
    assert len(SpiderLoader(spider, limit=1)) == 1
    assert len(list(SpiderLoader(spider, limit=2))) == 2
    assert len(SpiderLoader(spider, limit=None)) == 3
    assert len(SpiderLoader(spider, limit=100)) == 3  # limit beyond dataset size


def test_db_id_resolution(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer", "pets_1"])
    loader = SpiderLoader(spider)
    path = loader.resolve_db_path("pets_1")
    assert path.exists()
    assert path.name == "pets_1.sqlite"


def test_missing_database_raises(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer"])  # pets_1 has no file
    loader = SpiderLoader(spider)
    with pytest.raises(FileNotFoundError):
        loader.resolve_db_path("pets_1")


def test_missing_database_reported(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer"])  # pets_1 has no file
    loader = SpiderLoader(spider)
    missing_ids = [db_id for db_id, _ in loader.missing_databases()]
    assert "pets_1" in missing_ids
    assert "concert_singer" not in missing_ids


def test_iteration_raises_on_missing_db(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer"])  # pets_1 missing
    loader = SpiderLoader(spider)
    with pytest.raises(FileNotFoundError):
        list(loader)


def test_invalid_benchmark_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        SpiderLoader(tmp_path / "does_not_exist", split="dev")


def test_invalid_split(tmp_path):
    spider = _make_spider(tmp_path, SAMPLE, ["concert_singer", "pets_1"])
    with pytest.raises(ValueError):
        SpiderLoader(spider, split="test")
