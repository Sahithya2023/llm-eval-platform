"""API tests for the Stage 6 FastAPI layer.

Hermetic: the app is built via the factory, then dependencies are overridden so
that:
  * the DB is an in-memory SQLite shared across the test,
  * the Stage 5 runner is faked (no LLM / network), and
  * the example source returns canned examples (no Spider download).

This exercises the real routes, schemas, service composition, and repository
end-to-end, isolating only the genuinely external pieces.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.db  # noqa: F401  registers models
from app.api import create_app
from app.api.deps import get_db, get_example_source, get_runner
from app.db.models import Base
from app.services.evaluation_runner import EvaluationRecord, EvaluationSummary


# --------------------------------------------------------------------------- #
# Fakes / builders                                                             #
# --------------------------------------------------------------------------- #

def _record(example_id="dev_0", is_correct=True):
    return EvaluationRecord(
        example_id=example_id,
        db_id="concert_singer",
        question="How many singers?",
        gold_sql="SELECT count(*) FROM singer",
        generated_sql="SELECT count(*) FROM singer",
        is_correct=is_correct,
        generation_error=None,
        execution_error=None,
        comparison_reason="exact match",
        model="gpt-4o-mini",
        prompt_version="v1",
        prompt_tokens=3,
        completion_tokens=5,
        total_tokens=8,
        estimated_cost=0.00075,
        latency_ms=120,
    )


class FakeRunner:
    def __init__(self, records):
        self._records = records

    def run_examples(self, examples):
        recs = self._records
        total = len(recs)
        correct = sum(1 for r in recs if r.is_correct)
        summary = EvaluationSummary(
            total_examples=total,
            correct=correct,
            incorrect=total - correct,
            failed_generations=0,
            failed_executions=0,
            accuracy=(correct / total) if total else 0.0,
            total_prompt_tokens=3 * total,
            total_completion_tokens=5 * total,
            total_tokens=8 * total,
            total_estimated_cost=0.00075 * total,
            average_latency_ms=120.0 if total else 0.0,
        )
        return recs, summary


@pytest.fixture()
def client():
    # One in-memory DB shared across all sessions in this test.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    records = [_record("dev_0", True), _record("dev_1", False)]

    def _override_runner():
        return FakeRunner(records)

    def _override_example_source():
        # Ignore the request; return canned examples (count drives nothing here,
        # because the fake runner returns fixed records).
        return lambda request: ["example-1", "example-2"]

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_runner] = _override_runner
    app.dependency_overrides[get_example_source] = _override_example_source

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    engine.dispose()


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# POST /runs                                                                   #
# --------------------------------------------------------------------------- #

def test_start_run_creates_and_returns_run(client):
    resp = client.post("/runs", json={"split": "dev", "limit": 2, "name": "smoke"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "smoke"
    assert body["split"] == "dev"
    assert body["dataset_limit"] == 2
    assert body["total_examples"] == 2
    assert body["correct"] == 1
    assert body["incorrect"] == 1
    assert body["accuracy"] == pytest.approx(0.5)
    assert "id" in body and body["id"]


def test_start_run_validation_rejects_bad_limit(client):
    resp = client.post("/runs", json={"split": "dev", "limit": 0})
    assert resp.status_code == 422


def test_start_run_validation_rejects_unknown_split(client):
    resp = client.post("/runs", json={"split": "nonsense"})
    assert resp.status_code == 422


def test_start_run_rejects_unknown_field(client):
    resp = client.post("/runs", json={"split": "dev", "bogus": 1})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# GET /runs and /runs/{id}                                                     #
# --------------------------------------------------------------------------- #

def test_get_run_roundtrip(client):
    run_id = client.post("/runs", json={"split": "dev"}).json()["id"]

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


def test_get_unknown_run_404(client):
    resp = client.get("/runs/nope")
    assert resp.status_code == 404


def test_list_runs_history(client):
    id1 = client.post("/runs", json={"split": "dev", "name": "a"}).json()["id"]
    id2 = client.post("/runs", json={"split": "dev", "name": "b"}).json()["id"]

    resp = client.get("/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    ids = {r["id"] for r in body["runs"]}
    assert ids == {id1, id2}


def test_list_runs_pagination(client):
    for _ in range(3):
        client.post("/runs", json={"split": "dev"})
    resp = client.get("/runs?limit=2&offset=0")
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


# --------------------------------------------------------------------------- #
# GET /runs/{id}/records                                                       #
# --------------------------------------------------------------------------- #

def test_get_records_for_run(client):
    run_id = client.post("/runs", json={"split": "dev"}).json()["id"]

    resp = client.get(f"/runs/{run_id}/records")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["count"] == 2
    first = body["records"][0]
    assert first["example_id"] == "dev_0"
    assert first["is_correct"] is True
    assert first["comparison_reason"] == "exact match"
    # second record is the incorrect one
    assert body["records"][1]["is_correct"] is False


def test_get_records_for_unknown_run_404(client):
    resp = client.get("/runs/nope/records")
    assert resp.status_code == 404
