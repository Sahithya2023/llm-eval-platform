"""Database initialization.

Creates the data directory (for the SQLite file) and all tables registered on
`Base`. Safe to run repeatedly: existing tables are left untouched.

Run with:
    python -m app.db.init_db
"""

import os

from app.core.config import get_settings
from app.db import models  # noqa: F401  (import registers models on Base)
from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    settings = get_settings()

    # The SQLite file lives under data/; make sure the directory exists.
    os.makedirs(settings.data_dir, exist_ok=True)

    Base.metadata.create_all(bind=engine)

    tables = ", ".join(sorted(Base.metadata.tables.keys()))
    print(f"[init_db] Database URL : {settings.database_url}")
    print(f"[init_db] Tables ready : {tables}")
    print("[init_db] Stage 1 initialization complete.")


if __name__ == "__main__":
    init_db()
