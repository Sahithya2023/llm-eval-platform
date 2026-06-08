"""Unit tests for the Stage 5 evaluation runner.

Hermetic: no network, no OpenAI, no Spider download. The runner's collaborators
(LLM client, SQL executor, comparator, schema provider) are replaced with fakes.
Where it adds fidelity, the *real* ``compare_executions`` oracle is used against
synthetic execution results — it is pure and needs no files.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval import compare_executions
from app.eval.comparator import Comparison
from app.eval.executor import ExecutionResult
from app.eval.llm_client import GenerationResult
from app.services import evaluation_runner
from app.services.evaluation_runner import (
    EvaluationRecord,
    EvaluationRunner,
    EvaluationSummary,
)


# --------------------------------------------------------------------------- #
# Builders for the real Stage 2-4 data shapes                                  #
# --------------------------------------------------------------------------- #

def _example(
    example_id="dev_0",
    db_id="concert_singer",
    question="How many singers are there?",
    gold_sql="SELECT count(*) FROM singer",
    db_path="/fake/concert_singer.sqlite",
):
    return SimpleNamespace(
        example_id=example_id,
        db_id=db_id,
        question=question,
        gold_sql=gold_sql,
        db_path=db_path,
    )


def _gen(
    sql="SELECT count(*) FROM singer",
    error=None,
    model="gpt-4o-mini",
    prompt_version="v1",
    pt=3,
    ct=5,
    tt=8,
    cost=0.00075,
    latency=120,
):
    return GenerationResult(
        sql=sql,
        model=model,
        prompt_version=prompt_version,
        prompt_hash="a" * 64,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        estimated_cost=cost,
        latency_ms=latency,
        error=error,
    )


def _exec_ok(rows=((1,),), columns=("c",)):
    return ExecutionResult(list(rows), columns, None, None, False)


def _exec_err(msg="boom", etype="sql_error"):
    return ExecutionResult(None, None, msg, etype, False)


def _cmp(is_correct=True, reason="exact match", order=False):
    return Comparison(is_correct, reason, order)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

class FakeLLM:
    def __init__(self, result=None, by_question=None):
        self.result = result
        self.by_question = by_question or {}
        self.calls: list[tuple[str, str]] = []

    def generate_sql(self, question, schema_context):
        self.calls.append((question, schema_context))
        if question in self.by_question:
            return self.by_question[question]
        return self.result


class FakeExecutor:
    def __init__(self, by_sql=None, default=None):
        self.by_sql = by_sql or {}
        self.default = default if default is not None else _exec_ok()
        self.calls: list[tuple[str, str]] = []

    def execute(self, db_path, sql):
        self.calls.append((str(db_path), sql))
        return self.by_sql.get(sql, self.default)


class FakeComparator:
    def __init__(self, result=None):
        self.result = result if result is not None else _cmp()
        self.calls: list[tuple] = []

    def __call__(self, gold_exec, pred_exec, gold_sql):
        self.calls.append((gold_exec, pred_exec, gold_sql))
        return self.result


class FakeSchema:
    def __init__(self, schema="CREATE TABLE singer (id INT, name TEXT);"):
        self.schema = schema
        self.calls: list[Path] = []

    def __call__(self, db_path):
        self.calls.append(Path(db_path))
        return self.schema


def _runner(llm=None, ex=None, schema=None, cmp=None):
    return EvaluationRunner(
        llm_client=llm if llm is not None else FakeLLM(result=_gen()),
        sql_executor=ex if ex is not None else FakeExecutor(),
        schema_provider=schema if schema is not None else FakeSchema(),
        comparator=cmp if cmp is not None else FakeComparator(),
    )


# --------------------------------------------------------------------------- #
# run_example — paths                                                          #
# --------------------------------------------------------------------------- #

def test_run_example_success():
    r = _runner().run_example(_example())
    assert r.is_correct
    assert r.generated_sql == "SELECT count(*) FROM singer"
    assert r.generation_error is None
    assert r.execution_error is None
    assert r.comparison_reason == "exact match"


def test_generation_failure_path():
    llm = FakeLLM(result=_gen(sql=None, error="APIError: boom",
                              pt=None, ct=None, tt=None, cost=None, latency=70))
    ex = FakeExecutor()
    r = _runner(llm=llm, ex=ex).run_example(_example())

    assert not r.is_correct
    assert r.generated_sql is None
    assert r.generation_error == "APIError: boom"
    assert r.execution_error is None
    assert "generation failed" in r.comparison_reason.lower()
    assert ex.calls == []  # execution skipped
    # metadata still populated from the GenerationResult
    assert r.model == "gpt-4o-mini"
    assert r.prompt_version == "v1"
    assert r.latency_ms == 70


def test_predicted_execution_failure_path():
    bad = "SELECT * FROM missing"
    llm = FakeLLM(result=_gen(sql=bad))
    ex = FakeExecutor(by_sql={
        bad: _exec_err("no such table"),
        "SELECT count(*) FROM singer": _exec_ok(),
    })
    cmp = FakeComparator(result=_cmp(is_correct=False, reason="prediction failed to execute"))
    r = _runner(llm=llm, ex=ex, cmp=cmp).run_example(_example())

    assert not r.is_correct
    assert r.execution_error == "no such table"
    assert r.generation_error is None


def test_gold_execution_failure_path():
    gold = "SELECT broken gold"
    llm = FakeLLM(result=_gen(sql="SELECT 1"))
    ex = FakeExecutor(by_sql={
        "SELECT 1": _exec_ok(rows=[(1,)]),
        gold: _exec_err("gold broke"),
    })
    # real oracle so the gold-failure reason is produced authentically
    r = _runner(llm=llm, ex=ex, cmp=compare_executions).run_example(
        _example(gold_sql=gold)
    )

    assert not r.is_correct
    assert "gold" in r.comparison_reason.lower()
    assert r.execution_error is None  # the prediction itself executed fine


def test_comparison_failure_path():
    cmp = FakeComparator(result=_cmp(is_correct=False, reason="multiset mismatch"))
    r = _runner(cmp=cmp).run_example(_example())

    assert not r.is_correct
    assert r.comparison_reason == "multiset mismatch"
    assert r.generation_error is None
    assert r.execution_error is None


# --------------------------------------------------------------------------- #
# run_example — wiring                                                          #
# --------------------------------------------------------------------------- #

def test_schema_provider_is_called():
    schema = FakeSchema()
    _runner(schema=schema).run_example(_example(db_path="/p/db.sqlite"))
    assert schema.calls == [Path("/p/db.sqlite")]


def test_generated_sql_forwarded_to_executor():
    llm = FakeLLM(result=_gen(sql="SELECT 42"))
    ex = FakeExecutor()
    _runner(llm=llm, ex=ex).run_example(_example(gold_sql="SELECT gold"))
    executed = [sql for (_, sql) in ex.calls]
    assert "SELECT 42" in executed   # predicted SQL
    assert "SELECT gold" in executed  # gold SQL


def test_comparator_receives_correct_inputs():
    pred_res = _exec_ok(rows=[(1,)])
    gold_res = _exec_ok(rows=[(2,)])
    llm = FakeLLM(result=_gen(sql="PRED"))
    ex = FakeExecutor(by_sql={"PRED": pred_res, "GOLD": gold_res})
    cmp = FakeComparator()
    _runner(llm=llm, ex=ex, cmp=cmp).run_example(_example(gold_sql="GOLD"))

    assert len(cmp.calls) == 1
    g_exec, p_exec, g_sql = cmp.calls[0]
    assert g_exec is gold_res
    assert p_exec is pred_res
    assert g_sql == "GOLD"


def test_metadata_propagated():
    llm = FakeLLM(result=_gen(model="gpt-4o", prompt_version="v2"))
    r = _runner(llm=llm).run_example(_example())
    assert r.model == "gpt-4o"
    assert r.prompt_version == "v2"


def test_token_counts_propagated():
    r = _runner(llm=FakeLLM(result=_gen(pt=11, ct=22, tt=33))).run_example(_example())
    assert (r.prompt_tokens, r.completion_tokens, r.total_tokens) == (11, 22, 33)


def test_cost_propagated():
    r = _runner(llm=FakeLLM(result=_gen(cost=0.0042))).run_example(_example())
    assert r.estimated_cost == pytest.approx(0.0042)


def test_latency_propagated():
    r = _runner(llm=FakeLLM(result=_gen(latency=250))).run_example(_example())
    assert r.latency_ms == 250


def test_dependency_injection_used():
    schema, llm = FakeSchema(), FakeLLM(result=_gen())
    ex, cmp = FakeExecutor(), FakeComparator()
    EvaluationRunner(
        llm_client=llm, sql_executor=ex, schema_provider=schema, comparator=cmp
    ).run_example(_example())
    assert schema.calls and llm.calls and ex.calls and cmp.calls


def test_runner_never_raises_on_unexpected_failure():
    class Boom:
        def execute(self, db_path, sql):
            raise RuntimeError("kaboom")

    r = EvaluationRunner(
        llm_client=FakeLLM(result=_gen(sql="X")),
        sql_executor=Boom(),
        schema_provider=FakeSchema(),
        comparator=FakeComparator(),
    ).run_example(_example())

    assert not r.is_correct
    assert r.execution_error is not None


# --------------------------------------------------------------------------- #
# run_examples — batch + summary                                               #
# --------------------------------------------------------------------------- #

def _mixed():
    """Three examples: correct, generation-failure, predicted-execution-failure."""
    gens = {
        "q1": _gen(sql="S1", pt=10, ct=10, tt=20, cost=0.001, latency=100),
        "q2": _gen(sql=None, error="boom", pt=None, ct=None, tt=None, cost=None, latency=50),
        "q3": _gen(sql="S3", pt=5, ct=5, tt=10, cost=0.002, latency=210),
    }
    llm = FakeLLM(by_question=gens)
    ex = FakeExecutor(by_sql={
        "S1": _exec_ok(rows=[(1,)]), "G1": _exec_ok(rows=[(1,)]),
        "S3": _exec_err("exec boom"), "G3": _exec_ok(rows=[(9,)]),
    })
    runner = EvaluationRunner(
        llm_client=llm, sql_executor=ex,
        schema_provider=FakeSchema(), comparator=compare_executions,
    )
    examples = [
        _example(example_id="dev_1", question="q1", gold_sql="G1", db_path="/d1"),
        _example(example_id="dev_2", question="q2", gold_sql="G2", db_path="/d2"),
        _example(example_id="dev_3", question="q3", gold_sql="G3", db_path="/d3"),
    ]
    return runner.run_examples(examples)


def test_batch_returns_record_per_example():
    records, _ = _mixed()
    assert [r.example_id for r in records] == ["dev_1", "dev_2", "dev_3"]


def test_batch_summary_counts():
    _, s = _mixed()
    assert s.total_examples == 3
    assert s.correct == 1
    assert s.incorrect == 2
    assert s.failed_generations == 1
    assert s.failed_executions == 1


def test_accuracy_calculation():
    _, s = _mixed()
    assert s.accuracy == pytest.approx(1 / 3)


def test_token_aggregation():
    _, s = _mixed()
    assert s.total_prompt_tokens == 15
    assert s.total_completion_tokens == 15
    assert s.total_tokens == 30


def test_cost_aggregation():
    _, s = _mixed()
    assert s.total_estimated_cost == pytest.approx(0.003)


def test_average_latency_calculation():
    _, s = _mixed()
    assert s.average_latency_ms == pytest.approx((100 + 50 + 210) / 3)


def test_empty_batch():
    records, s = _runner().run_examples([])
    assert records == []
    assert s.total_examples == 0
    assert s.accuracy == 0.0
    assert s.average_latency_ms == 0.0
    assert s.total_tokens == 0
    assert s.total_estimated_cost == 0.0


# --------------------------------------------------------------------------- #
# Shapes + architecture guard                                                  #
# --------------------------------------------------------------------------- #

def test_evaluation_record_shape():
    assert EvaluationRecord._fields == (
        "example_id",
        "db_id",
        "question",
        "gold_sql",
        "generated_sql",
        "is_correct",
        "generation_error",
        "execution_error",
        "comparison_reason",
        "model",
        "prompt_version",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost",
        "latency_ms",
    )


def test_evaluation_summary_shape():
    assert EvaluationSummary._fields == (
        "total_examples",
        "correct",
        "incorrect",
        "failed_generations",
        "failed_executions",
        "accuracy",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_estimated_cost",
        "average_latency_ms",
    )


def test_no_forbidden_imports():
    src = Path(evaluation_runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for forbidden in ("fastapi", "streamlit", "openai", "sqlalchemy"):
        assert forbidden not in imported, f"forbidden import: {forbidden}"
