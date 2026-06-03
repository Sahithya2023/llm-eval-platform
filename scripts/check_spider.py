"""Stage 2 verification: prove Spider loads and db_id -> SQLite mapping works.

Run from the project root:

    python scripts/check_spider.py

Exits 0 on success, 1 on any failure.
"""

import sys
from itertools import islice
from pathlib import Path

# Allow running as a plain script (`python scripts/check_spider.py`) by putting
# the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.datasets.spider_loader import SpiderLoader  # noqa: E402

SPLIT = "dev"
LIMIT = 50


def main() -> int:
    settings = get_settings()
    spider_dir = settings.spider_dir

    print("=" * 60)
    print("Spider dataset check")
    print("=" * 60)
    print(f"Benchmark dir : {spider_dir}")
    print(f"Split         : {SPLIT}")
    print(f"Limit         : {LIMIT}")
    print()

    # 1. Load examples.
    try:
        loader = SpiderLoader(spider_dir, split=SPLIT, limit=LIMIT)
    except (FileNotFoundError, ValueError) as exc:
        print("FAILED to initialize the loader:")
        print(f"  {exc}")
        return 1

    # 2. Dataset statistics.
    db_ids = loader.unique_db_ids()
    print(f"Loaded examples   : {len(loader)}")
    print(f"Unique databases  : {len(db_ids)}")
    print()

    # 4 & 5. Verify db_id -> SQLite mapping and report failures clearly.
    missing = loader.missing_databases()
    if missing:
        print(f"MISSING DATABASES ({len(missing)}):")
        for db_id, path in missing:
            print(f"  - {db_id}: {path}")
        print()
        print("Resolve the missing databases before continuing to Stage 3.")
        return 1
    print(f"All {len(db_ids)} databases resolved successfully.")
    print()

    # 3. Print 3 sample examples.
    print("Sample examples:")
    print("-" * 60)
    for example in islice(loader, 3):
        print(f"  example_id : {example.example_id}")
        print(f"  db_id      : {example.db_id}")
        print(f"  question   : {example.question}")
        print(f"  gold_sql   : {example.gold_sql}")
        print(f"  db_path    : {example.db_path}")
        print("-" * 60)

    # 6. Success.
    print()
    print("Stage 2 check passed: Spider loads and db_id mapping is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
