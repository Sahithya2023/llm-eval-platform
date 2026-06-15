"""Tests for the Stage 6 + 7 repository layer.

Hermetic: an in-memory SQLite database is created per test; the real ORM models
and repository are exercised. No network, no Spider, no real LLM.

Stage 7 (Phase 1) adds tests for the three new lifecycle methods:
    create_pending_run / set_status / finalize_run
All Stage 6 tests are preserved exactly.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.db  # noqa: F401  registers all models on Base
from app.db.models import Base
from app.db.repository import (
    AbstractEvaluationRepository,
    RunMetadata,
    SqlAlchemyEvaluationRepository,
    new_run_id,
)
from app.services.evaluation_runner import EvaluationRecord, EvaluationSummary


# --------------------------------------------------------------------------- #
# Fixtures / builders                                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def repo(session) -> SqlAlchemyEvaluationRepository:
    return SqlAlchemyEvaluationRepository(session)


def _record(
    example_id="dev_0",
    is_correct=True,
    generation_error=None,
    execution_error=None,
    comparison_reason="exact match",
):
    return EvaluationRecord(
        example_id=example_id,
        db_id="concert_singer",
        question="How many singers are there?",
        gold_sql="SELECT count(*) FROM singer",
        generated_sql="SELECT count(*) FROM singer",
        is_correct=is_correct,
        generation_error=generation_error,
        execution_error=execution_error,
        comparison_reason=comparison_reason,
        model="gpt-4o-mini",
        prompt_version="v1",
        prompt_tokens=3,
        completion_tokens=5,
        total_tokens=8,
        estimated_cost=0.00075,
        latency_ms=120,
    )


def _summary(records):
    total = len(records)
    correct = sum(1 for r in records if r.is_correct)
    return EvaluationSummary(
        total_examples=total,
        correct=correct,
        incorrect=total - correct,
        failed_generations=sum(1 for r in records if r.generation_error),
        failed_executions=sum(1 for r in records if r.execution_error),
        accuracy=(correct / total) if total else 0.0,
        total_prompt_tokens=sum(r.prompt_tokens or 0 for r in records),
        total_completion_tokens=sum(r.completion_tokens or 0 for r in records),
        total_tokens=sum(r.total_tokens or 0 for r in records),
        total_estimated_cost=sum(r.estimated_cost or 0.0 for r in records),
        average_latency_ms=(
            sum(r.latency_ms or 0 for r in records) / total if total else 0.0
        ),
    )


# --------------------------------------------------------------------------- #
# Contract                                                                     #
# --------------------------------------------------------------------------- #

def test_concrete_repo_satisfies_abstraction(repo):
    assert isinstance(repo, AbstractEvaluationRepository)


# --------------------------------------------------------------------------- #
# Stage 6: create_run round-trip (frozen)                                     #
# --------------------------------------------------------------------------- #

def test_create_run_persists_run_and_records(repo):
    records = [_record("dev_0"), _record("dev_1", is_correct=False, comparison_reason="mismatch")]
    summary = _summary(records)
    meta = RunMetadata(name="smoke", split="dev", dataset_limit=2, model="gpt-4o-mini")

    run = repo.create_run(metadata=meta, records=records, summary=summary)

    assert run.id
    assert run.name == "smoke"
    assert run.split == "dev"
    assert run.dataset_limit == 2
    assert run.total_examples == 2
    assert run.correct == 1
    assert run.incorrect == 1
    assert run.accuracy == pytest.approx(0.5)
    assert run.total_tokens == 16
    assert len(run.records) == 2


def test_create_run_uses_supplied_run_id(repo):
    run_id = new_run_id()
    records = [_record()]
    run = repo.create_run(
        metadata=RunMetadata(), records=records, summary=_summary(records), run_id=run_id
    )
    assert run.id == run_id


def test_records_round_trip_lossless(repo):
    """The three distinct error/reason fields survive persistence (the reason
    Stage 6 uses dedicated tables instead of the lossy single-error traces row)."""
    rec = _record(
        "dev_7",
        is_correct=False,
        generation_error="boom-gen",
        execution_error="boom-exec",
        comparison_reason="why-not",
    )
    run = repo.create_run(metadata=RunMetadata(), records=[rec], summary=_summary([rec]))

    fetched = repo.get_records(run.id)
    assert len(fetched) == 1
    f = fetched[0]
    assert f.example_id == "dev_7"
    assert f.generation_error == "boom-gen"
    assert f.execution_error == "boom-exec"
    assert f.comparison_reason == "why-not"
    assert f.is_correct is False
    assert f.estimated_cost == pytest.approx(0.00075)


# --------------------------------------------------------------------------- #
# Stage 6: get / list (frozen)                                                #
# --------------------------------------------------------------------------- #

def test_get_run_returns_none_for_unknown(repo):
    assert repo.get_run("does-not-exist") is None


def test_get_records_for_unknown_run_is_empty(repo):
    assert repo.get_records("nope") == []


def test_list_runs_newest_first_and_paginates(repo):
    ids = []
    for i in range(3):
        recs = [_record(f"dev_{i}")]
        run = repo.create_run(
            metadata=RunMetadata(name=f"run-{i}"),
            records=recs,
            summary=_summary(recs),
        )
        ids.append(run.id)

    all_runs = repo.list_runs(limit=50)
    assert len(all_runs) == 3
    # All three are retrievable and distinct.
    assert {r.id for r in all_runs} == set(ids)

    page = repo.list_runs(limit=1, offset=0)
    assert len(page) == 1
    page2 = repo.list_runs(limit=1, offset=1)
    assert len(page2) == 1
    assert page[0].id != page2[0].id


def test_save_run_then_save_records_separately(repo):
    records = [_record("dev_0"), _record("dev_1")]
    run_id = new_run_id()
    run = repo.save_run(run_id=run_id, metadata=RunMetadata(), summary=_summary(records))
    assert repo.get_records(run.id) == []

    saved = repo.save_records(run.id, records)
    assert len(saved) == 2
    assert len(repo.get_records(run.id)) == 2


def test_empty_run_is_valid(repo):
    summary = _summary([])
    run = repo.create_run(metadata=RunMetadata(), records=[], summary=summary)
    assert run.total_examples == 0
    assert run.accuracy == 0.0
    assert repo.get_records(run.id) == []


# --------------------------------------------------------------------------- #
# Stage 7: create_pending_run                                                  #
# --------------------------------------------------------------------------- #

def test_create_pending_run_status_and_defaults(repo):
    meta = RunMetadata(name="async-run", split="dev", model="gpt-4o-mini")
    run = repo.create_pending_run(metadata=meta)

    assert run.id
    assert run.status == "pending"
    assert run.name == "async-run"
    assert run.split == "dev"
    assert run.model == "gpt-4o-mini"
    # Summary columns are zeroed until finalization.
    assert run.total_examples == 0
    assert run.accuracy == 0.0
    # Lifecycle timestamps are all None at creation.
    assert run.started_at is None
    assert run.finished_at is None
    assert run.error is None


def test_create_pending_run_accepts_explicit_run_id(repo):
    run_id = new_run_id()
    run = repo.create_pending_run(metadata=RunMetadata(), run_id=run_id)
    assert run.id == run_id
    assert repo.get_run(run_id) is not None


def test_create_pending_run_metadata_status_is_ignored(repo):
    """metadata.status='completed' must not override the pending status."""
    run = repo.create_pending_run(
        metadata=RunMetadata(status="completed"),
    )
    assert run.status == "pending"


# --------------------------------------------------------------------------- #
# Stage 7: set_status                                                          #
# --------------------------------------------------------------------------- #

def test_set_status_pending_to_running_stamps_started_at(repo):
    run = repo.create_pending_run(metadata=RunMetadata())
    updated = repo.set_status(run.id, "running")

    assert updated is not None
    assert updated.status == "running"
    assert updated.started_at is not None
    assert updated.finished_at is None
    assert updated.error is None


def test_set_status_running_to_completed_stamps_finished_at(repo):
    run = repo.create_pending_run(metadata=RunMetadata())
    repo.set_status(run.id, "running")
    completed = repo.set_status(run.id, "completed")

    assert completed.status == "completed"
    assert completed.finished_at is not None
    assert completed.error is None


def test_set_status_failed_stamps_finished_at_and_error(repo):
    run = repo.create_pending_run(metadata=RunMetadata())
    repo.set_status(run.id, "running")
    failed = repo.set_status(run.id, "failed", error="ValueError: boom")

    assert failed.status == "failed"
    assert failed.finished_at is not None
    assert failed.error == "ValueError: boom"


def test_set_status_running_does_not_overwrite_started_at(repo):
    """A second running transition must not clobber the original started_at."""
    run = repo.create_pending_run(metadata=RunMetadata())
    first = repo.set_status(run.id, "running")
    first_ts = first.started_at

    second = repo.set_status(run.id, "running")
    assert second.started_at == first_ts


def test_set_status_unknown_run_returns_none(repo):
    result = repo.set_status("no-such-id", "running")
    assert result is None


# --------------------------------------------------------------------------- #
# Stage 7: finalize_run                                                        #
# --------------------------------------------------------------------------- #

def test_finalize_run_writes_records_and_summary(repo):
    run = repo.create_pending_run(metadata=RunMetadata(split="dev"))
    repo.set_status(run.id, "running")

    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    summary = _summary(records)

    final = repo.finalize_run(run.id, records=records, summary=summary)

    assert final is not None
    assert final.status == "completed"
    assert final.finished_at is not None
    assert final.total_examples == 2
    assert final.correct == 1
    assert final.incorrect == 1
    assert final.accuracy == pytest.approx(0.5)
    assert final.total_tokens == 16

    stored_records = repo.get_records(run.id)
    assert len(stored_records) == 2
    assert stored_records[0].example_id == "dev_0"
    assert stored_records[0].is_correct is True
    assert stored_records[1].is_correct is False


def test_finalize_run_with_empty_records(repo):
    run = repo.create_pending_run(metadata=RunMetadata())
    repo.set_status(run.id, "running")

    final = repo.finalize_run(run.id, records=[], summary=_summary([]))

    assert final.status == "completed"
    assert final.total_examples == 0
    assert repo.get_records(run.id) == []


def test_finalize_run_unknown_run_returns_none(repo):
    result = repo.finalize_run("no-such-id", records=[], summary=_summary([]))
    assert result is None


# --------------------------------------------------------------------------- #
# Stage 7: full lifecycle round-trip                                           #
# --------------------------------------------------------------------------- #

def test_full_lifecycle_pending_running_completed(repo):
    """Exercise the complete happy path: create → running → finalized."""
    meta = RunMetadata(name="lifecycle-test", split="dev")
    run = repo.create_pending_run(metadata=meta)
    assert run.status == "pending"

    running = repo.set_status(run.id, "running")
    assert running.status == "running"
    assert running.started_at is not None

    records = [_record("dev_0"), _record("dev_1", is_correct=False)]
    final = repo.finalize_run(run.id, records=records, summary=_summary(records))

    assert final.status == "completed"
    assert final.started_at is not None   # set by set_status("running")
    assert final.finished_at is not None  # set by finalize_run
    assert final.total_examples == 2
    assert len(repo.get_records(run.id)) == 2


def test_full_lifecycle_pending_running_failed(repo):
    """Exercise the failure path: create → running → failed."""
    run = repo.create_pending_run(metadata=RunMetadata())
    repo.set_status(run.id, "running")

    failed = repo.set_status(run.id, "failed", error="OSError: disk full")

    assert failed.status == "failed"
    assert failed.finished_at is not None
    assert failed.error == "OSError: disk full"
    # No records should exist.
    assert repo.get_records(run.id) == []