"""FastAPI application factory (Stage 6).

``create_app`` builds the application and includes the evaluation router. It
also ensures the database schema exists at startup (idempotent), so a fresh
deployment has the Stage 6 tables without a separate init step. Tests build
their own app via this factory and override dependencies.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as evaluation_router
from app.core.config import get_settings


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Importing app.db registers every model on Base; create_all is a no-op for
    # tables that already exist (and harmless for an injected test DB).
    import app.db  # noqa: F401  (registers models)
    from app.db.models import Base
    from app.db.session import engine

    Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.6.0",
        description="Persistent evaluation platform for Text-to-SQL agents.",
        lifespan=_lifespan,
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(evaluation_router)
    return app
