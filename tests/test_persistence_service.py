"""Tests for the Stage 6 + 7 persistence service.

The service is verified two ways:
  * with a *fake* repository, to prove it composes the runner + repository and
    depends only on the repository abstraction (no DB needed); and
  * with the *real* repository on an in-memory DB, to prove the full write path.

The Stage 5 runner is always faked, so no network / LLM / Spider is touched.

Stage 7 (Phase 1) adds tests for ``create_pending`` and
``execute_and_finalize``.  All Stage 6 tests are preserved exactly.
The ``FakeRepo`` is extended with the three new lifecycle methods so the
service's new paths can be exercised without a real DB.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.db  # noqa: F401  registers models
from app.db.models import Base
from app.db.repository import RunMetadata, SqlAlchemyEvaluationRepository
from app.services.evaluation_runner import EvaluationRecord, EvaluationSummary
from app.services.persistence_service import PersistentEvaluationService


# --------------------------------------------------------------------------- #
# Fakes / builders                                                             #
# --------------------------------------------------------------------------- #

def _record(example_id="dev_0", is_correct=True):
    return EvaluationRecord(
        example_id=example_id,
        db_id="concert_singer",
        question="q",
        gold_sql="SELECT 1",
        generated_sql="SELECT 1",
        is_correct=is_correct,
        generation_error=None,
        execution_error=None,
        comparison_reason="exact match",
        model="gpt-4o-mini",
        prompt_version="v1",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        estimated_cost=0.001,
        latency_ms=50,
    )


def _summary(records):
    total = len(records)
    correct = sum(1 for r in records if r.is_correct)
    return EvaluationSummary(
        total_examples=total,
        correct=correct,
        incorrect=total - correct,
        failed_generations=0,
        failed_executions=0,
        accuracy=(correct / total) if total else 0.0,
        total_prompt_tokens=total,
        total_completion_tokens=2 * total,
        total_tokens=3 * total,
        total_estimated_cost=0.001 * total,
        average_latency_ms=50.0 if total else 0.0,
    )


class FakeRunner:
    """Stands in for the frozen Stage 5 EvaluationRunner."""

    def __init__(self, records, *, raise_on_run=False):
        self._records = records
        self._raise_on_run = raise_on_run
        self.calls = []

    def run_examples(self, examples):
        self.calls.append(list(examples))
        if self._raise_on_run:
            raise RuntimeError("simulated runner failure")
        return self._records, _summary(self._records)


class FakeRepo:
    """In-memory fake honouring AbstractEvaluationRepository.

    Extended in Stage 7 with the three new lifecycle methods so the service's
    new paths can be tested without a real DB session.
    """

    def __init__(self):
        self.runs = {}
        self.records = {}

    # -- Stage 6 methods (unchanged) --------------------------------------- #

    def save_run(self, *, run_id, metadata, summary):
        from types import SimpleNamespace

        run = SimpleNamespace(id=run_id, metadata=metadata, summary=summary)
        self.runs[run_id] = run
        return run

    def save_records(self, run_id, records):
        self.records.setdefault(run_id, []).extend(records)
        return list(records)

    def create_run(self, *, metadata, records, summary, run_id=None):
        from types import SimpleNamespace

        run_id = run_id or "fake-run"
        run = SimpleNamespace(
            id=run_id,
            status=metadata.status,
            metadata=metadata,
            summary=summary,
            records=list(records),
            started_at=None,
            finished_at=None,
            error=None,
        )
        self.runs[run_id] = run
        self.records[run_id] = list(records)
        return run

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def list_runs(self, *, limit=50, offset=0):
        return list(self.runs.values())[offset : offset + limit]

    def get_records(self, run_id):
        return self.records.get(run_id, [])

    # -- Stage 7 lifecycle methods ----------------------------------------- #

    def create_pending_run(self, *, metadata, run_id=None):
        from types import SimpleNamespace

        run_id = run_id or f"fake-{len(self.runs)}"
        run = SimpleNamespace(
            id=run_id,
            status="pending",
            name=metadata.name,
            started_at=None,
            finished_at=None,
            error=None,
            total_examples=0,
            accuracy=0.0,
        )
        self.runs[run_id] = run
        return run

    def set_status(self, run_id, status, *, error=None):
        run = self.runs.get(run_id)
        if run is None:
            return None
        run.status = status
        if error is not None:
            run.error = error
        return run

    def finalize_run(self, run_id, *, records, summary):
        run = self.runs.get(run_id)
        if run is None:
            return None
        run.status = "completed"
        run.total_examples = summary.total_examples
        run.accuracy = summary.accuracy
        self.records[run_id] = list(records)
        return run


@pytest.fixture()
def real_repo():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()
    try:
        yield SqlAlchemyEvaluationRepository(db)
    finally:
        db.close()
        engine.dispose()


# --------------------------------------------------------------------------- #
# Stage 6: composition with a fake repository (frozen)                        #
# --------------------------------------------------------------------------- #

def test_run_and_persist_invokes_runner_then_repository():
    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    runner = FakeRunner(records)
    repo = FakeRepo()
    service = PersistentEvaluationService(runner=runner, repository=repo)

    examples = ["ex-a", "ex-b"]
    run, returned_records, summary = service.run_and_persist(
        examples, metadata=RunMetadata(name="m"), run_id="r1"
    )

    # Runner was driven with the forwarded examples.
    assert runner.calls == [["ex-a", "ex-b"]]
    # Persisted under the supplied run_id.
    assert run.id == "r1"
    assert repo.get_run("r1") is run
    assert returned_records == records
    assert summary.total_examples == 2
    assert summary.correct == 1


def test_service_generates_run_id_when_absent():
    runner = FakeRunner([_record()])
    service = PersistentEvaluationService(runner=runner, repository=FakeRepo())
    run, _, _ = service.run_and_persist([], metadata=RunMetadata())
    assert run.id  # some id was assigned


def test_read_helpers_delegate_to_repository():
    repo = FakeRepo()
    runner = FakeRunner([_record()])
    service = PersistentEvaluationService(runner=runner, repository=repo)
    service.run_and_persist([], run_id="abc")

    assert service.get_run("abc") is repo.get_run("abc")
    assert service.list_runs() == repo.list_runs()
    assert service.get_records("abc") == repo.get_records("abc")


# --------------------------------------------------------------------------- #
# Stage 6: full write path against the real repository (frozen)               #
# --------------------------------------------------------------------------- #

def test_run_and_persist_writes_through_to_db(real_repo):
    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    runner = FakeRunner(records)
    service = PersistentEvaluationService(runner=runner, repository=real_repo)

    run, _, _ = service.run_and_persist(["x", "y"], metadata=RunMetadata(split="dev"))

    persisted = service.get_run(run.id)
    assert persisted is not None
    assert persisted.total_examples == 2
    assert persisted.correct == 1
    assert len(service.get_records(run.id)) == 2


# --------------------------------------------------------------------------- #
# Stage 7: create_pending                                                      #
# --------------------------------------------------------------------------- #

def test_create_pending_returns_pending_run():
    repo = FakeRepo()
    service = PersistentEvaluationService(runner=FakeRunner([]), repository=repo)

    run = service.create_pending(RunMetadata(name="async-test"), run_id="p1")

    assert run.id == "p1"
    assert run.status == "pending"
    assert repo.get_run("p1") is run


def test_create_pending_generates_run_id_when_absent():
    repo = FakeRepo()
    service = PersistentEvaluationService(runner=FakeRunner([]), repository=repo)

    run = service.create_pending()
    assert run.id  # auto-generated


def test_create_pending_against_real_repo(real_repo):
    service = PersistentEvaluationService(runner=FakeRunner([]), repository=real_repo)
    run = service.create_pending(RunMetadata(name="pending-test", split="dev"))

    assert run.status == "pending"
    assert run.name == "pending-test"
    fetched = service.get_run(run.id)
    assert fetched is not None
    assert fetched.status == "pending"


# --------------------------------------------------------------------------- #
# Stage 7: execute_and_finalize — happy path                                  #
# --------------------------------------------------------------------------- #

def test_execute_and_finalize_completes_run():
    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    runner = FakeRunner(records)
    repo = FakeRepo()
    service = PersistentEvaluationService(runner=runner, repository=repo)

    run = service.create_pending(RunMetadata(), run_id="e1")
    final = service.execute_and_finalize("e1", ["ex-a", "ex-b"])

    assert final is not None
    assert final.status == "completed"
    assert final.total_examples == 2
    assert final.accuracy == pytest.approx(0.5)
    # Runner was called with the examples.
    assert runner.calls == [["ex-a", "ex-b"]]


def test_execute_and_finalize_against_real_repo(real_repo):
    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    runner = FakeRunner(records)
    service = PersistentEvaluationService(runner=runner, repository=real_repo)

    run = service.create_pending(RunMetadata(split="dev"))
    final = service.execute_and_finalize(run.id, ["x", "y"])

    assert final is not None
    assert final.status == "completed"
    assert final.total_examples == 2
    assert len(service.get_records(run.id)) == 2


# --------------------------------------------------------------------------- #
# Stage 7: execute_and_finalize — failure paths                               #
# --------------------------------------------------------------------------- #

def test_execute_and_finalize_runner_failure_marks_failed():
    repo = FakeRepo()
    runner = FakeRunner([], raise_on_run=True)
    service = PersistentEvaluationService(runner=runner, repository=repo)

    service.create_pending(RunMetadata(), run_id="f1")
    result = service.execute_and_finalize("f1", [])

    assert result is not None
    assert result.status == "failed"
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_execute_and_finalize_unknown_run_id_returns_none():
    repo = FakeRepo()
    service = PersistentEvaluationService(runner=FakeRunner([]), repository=repo)

    result = service.execute_and_finalize("no-such-id", [])
    assert result is None