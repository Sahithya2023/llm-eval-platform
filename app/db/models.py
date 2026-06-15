"""Database models.

Defines the shared declarative ``Base`` that every ORM model registers on. The
concrete evaluation tables (``evaluation_runs`` / ``evaluation_records``) live in
:mod:`app.db.eval_models`; importing :mod:`app.db` registers them on this
``Base`` so ``init_db`` creates them together.

The ``traces`` table and ``Trace`` model introduced in Stage 1 have been removed:
Stage 6's dedicated tables superseded them and nothing writes to ``traces``.
``Base`` is preserved here because it is imported by ``eval_models``, ``init_db``,
``app.py``, and the test suite.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base shared by every model."""