"""Database package.

Importing :mod:`app.db` registers every ORM model on the shared declarative
``Base``. This is what allows ``init_db`` (which does ``from app.db import
models``) to create the Stage 6 tables alongside the Stage 1 ``traces`` table
*without* any change to the init script: importing the ``app.db`` package runs
this module first, which pulls in both model modules.
"""

from app.db import models  # noqa: F401  registers Trace on Base
from app.db import eval_models  # noqa: F401  registers Stage 6 models on Base

__all__ = ["models", "eval_models"]
