"""API routes (Stage 6).

Four endpoints over the persistence service:

    POST   /runs                 start (run + persist) an evaluation
    GET    /runs                 list historical runs (paginated)
    GET    /runs/{run_id}        retrieve a single run (metadata + summary)
    GET    /runs/{run_id}/records retrieve per-example records for a run

Handlers stay thin: build examples, call the service, shape the response.
All persistence and orchestration live behind the injected service.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    ExampleSource,
    get_example_source,
    get_service,
)
from app.db.repository import RunMetadata
from app.schemas.evaluation import (
    RecordListResponse,
    RecordResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
)

if TYPE_CHECKING:
    from app.datasets.spider_loader import SpiderExample
    from app.services.persistence_service import PersistentEvaluationService

router = APIRouter(prefix="/runs", tags=["evaluation"])


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def start_run(
    request: RunRequest,
    service: "PersistentEvaluationService" = Depends(get_service),
    example_source: ExampleSource = Depends(get_example_source),
) -> RunResponse:
    """Start an evaluation run: evaluate examples and persist the result."""
    try:
        examples: Iterable["SpiderExample"] = example_source(request)
    except FileNotFoundError as exc:
        # Dataset not available where the server runs.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dataset unavailable: {exc}",
        ) from exc

    metadata = RunMetadata(
        name=request.name,
        notes=request.notes,
        dataset="spider",
        split=request.split,
        dataset_limit=request.limit,
        model=request.model,
        prompt_version=request.prompt_version,
        status="completed",
    )
    run, _records, _summary = service.run_and_persist(examples, metadata=metadata)
    return RunResponse.model_validate(run)


@router.get("", response_model=RunListResponse)
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: "PersistentEvaluationService" = Depends(get_service),
) -> RunListResponse:
    """List historical runs, newest first."""
    runs = service.list_runs(limit=limit, offset=offset)
    items = [RunResponse.model_validate(r) for r in runs]
    return RunListResponse(runs=items, count=len(items))


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: str,
    service: "PersistentEvaluationService" = Depends(get_service),
) -> RunResponse:
    """Retrieve a single run by id."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run not found: {run_id}",
        )
    return RunResponse.model_validate(run)


@router.get("/{run_id}/records", response_model=RecordListResponse)
def get_run_records(
    run_id: str,
    service: "PersistentEvaluationService" = Depends(get_service),
) -> RecordListResponse:
    """Retrieve the per-example records for a run (404 if the run is unknown)."""
    if service.get_run(run_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run not found: {run_id}",
        )
    rows = service.get_records(run_id)
    records = [RecordResponse.model_validate(r) for r in rows]
    return RecordListResponse(run_id=run_id, records=records, count=len(records))
