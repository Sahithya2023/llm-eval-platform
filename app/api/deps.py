"""API dependencies (Stage 6).

A small dependency chain that mirrors the project's existing injection style
(`get_db`). Each layer is overridable in tests via
``app.dependency_overrides``:

    get_db (Stage 1)  ->  get_repository  ->  get_service
                          get_runner      ->  get_service
                          get_example_source

Tests typically override ``get_runner`` (with a fake) and ``get_example_source``
(to avoid needing a Spider download or network), while keeping the real
repository/service wired to an in-memory database.
"""

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

from fastapi import Depends

from app.db.repository import SqlAlchemyEvaluationRepository
from app.db.session import get_db
from app.services.evaluation_runner import EvaluationRunner
from app.services.persistence_service import PersistentEvaluationService

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.datasets.spider_loader import SpiderExample
    from app.schemas.evaluation import RunRequest


# Type alias: build an iterable of examples from a validated request.
ExampleSource = Callable[["RunRequest"], Iterable["SpiderExample"]]


def get_repository(db: "Session" = Depends(get_db)) -> SqlAlchemyEvaluationRepository:
    """Request-scoped repository bound to the request's DB session."""
    return SqlAlchemyEvaluationRepository(db)


def get_runner() -> EvaluationRunner:
    """The frozen Stage 5 runner with its production defaults.

    Construction is cheap and touches neither the OpenAI SDK nor the network
    (the Stage 4 client imports lazily).
    """
    return EvaluationRunner()


def get_service(
    runner: EvaluationRunner = Depends(get_runner),
    repository: SqlAlchemyEvaluationRepository = Depends(get_repository),
) -> PersistentEvaluationService:
    return PersistentEvaluationService(runner=runner, repository=repository)


def _default_example_source(request: "RunRequest") -> Iterable["SpiderExample"]:
    """Build Spider examples from settings for a real run.

    Imported lazily so importing the API never requires the Spider dataset to
    be present; the loader only runs when a real run is actually requested.
    """
    from app.core.config import get_settings
    from app.datasets.spider_loader import SpiderLoader

    settings = get_settings()
    loader = SpiderLoader(
        settings.spider_dir,
        split=request.split,
        limit=request.limit,
    )
    return list(loader)


def get_example_source() -> ExampleSource:
    """Return the callable that turns a request into evaluation examples."""
    return _default_example_source
