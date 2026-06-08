"""Unit tests for the Stage 4 Text-to-SQL LLM client.

These are hermetic: they never touch the network or the real OpenAI SDK. A
``FakeClient`` mimics the modern SDK call shape
(``client.chat.completions.create(...)`` returning an object with
``choices[0].message.content`` and a ``usage`` block), so every code path —
success, malformed output, API failure, timeout — is exercised offline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval import llm_client
from app.eval.llm_client import (
    GenerationResult,
    TextToSqlClient,
    _estimate_cost,
    _prompt_hash,
)


# --------------------------------------------------------------------------- #
# Fakes: a minimal stand-in for the OpenAI client/response shape.             #
# --------------------------------------------------------------------------- #

class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _FakeCompletions:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    """Mimics the parts of ``openai.OpenAI`` that the client touches."""

    def __init__(self, content=None, usage=None, exc=None):
        response = None if exc is not None else _FakeResponse(content, usage)
        self.chat = _FakeChat(_FakeCompletions(response=response, exc=exc))


def _make_client(fake_client, model="gpt-4o-mini"):
    """Build a TextToSqlClient with injected settings + fake transport."""
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_base_url="https://example.test/v1",
        llm_model=model,
        default_temperature=0.0,
        default_prompt_version="v1",
    )
    return TextToSqlClient(settings=settings, client=fake_client)


# --------------------------------------------------------------------------- #
# Success                                                                      #
# --------------------------------------------------------------------------- #

def test_successful_generation():
    fake = FakeClient(content="SELECT * FROM singer", usage=_FakeUsage(3, 5, 8))
    result = _make_client(fake).generate_sql("How many singers?", "CREATE TABLE singer(id)")

    assert result.ok
    assert result.error is None
    assert result.sql == "SELECT * FROM singer"
    assert result.model == "gpt-4o-mini"
    assert result.prompt_version == "v1"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 5
    assert result.total_tokens == 8
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


# --------------------------------------------------------------------------- #
# SQL cleanup                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 2\n```", "SELECT 2"),
        ("   SELECT 3   ", "SELECT 3"),
        ("```sql\nSELECT a\nFROM t\n```", "SELECT a\nFROM t"),
    ],
)
def test_markdown_fence_stripping(raw, expected):
    fake = FakeClient(content=raw, usage=_FakeUsage(1, 1, 2))
    result = _make_client(fake).generate_sql("q", "schema")
    assert result.sql == expected


# --------------------------------------------------------------------------- #
# Errors                                                                       #
# --------------------------------------------------------------------------- #

def test_api_failure_returns_error():
    fake = FakeClient(exc=RuntimeError("boom"))
    result = _make_client(fake).generate_sql("q", "schema")

    assert not result.ok
    assert result.sql is None
    assert result.error is not None
    assert "boom" in result.error


def test_timeout_failure():
    fake = FakeClient(exc=TimeoutError("request timed out"))
    result = _make_client(fake).generate_sql("q", "schema")

    assert not result.ok
    assert result.sql is None
    assert "timed out" in result.error.lower()


def test_malformed_response_is_error():
    # usage present but no content -> treated as a failure, not a silent pass.
    fake = FakeClient(content=None, usage=_FakeUsage(1, 0, 1))
    result = _make_client(fake).generate_sql("q", "schema")

    assert not result.ok
    assert result.sql is None
    assert result.error is not None


def test_failure_preserves_metadata():
    fake = FakeClient(exc=RuntimeError("down"))
    result = _make_client(fake).generate_sql("q", "schema")

    # Even on failure these must be populated for attribution/reproducibility.
    assert result.model == "gpt-4o-mini"
    assert result.prompt_version == "v1"
    assert len(result.prompt_hash) == 64
    assert isinstance(result.latency_ms, int)
    assert result.sql is None
    assert result.error is not None


# --------------------------------------------------------------------------- #
# Prompt hashing                                                               #
# --------------------------------------------------------------------------- #

def test_prompt_hash_is_deterministic():
    a = _prompt_hash("q", "schema", "SYS", "v1")
    b = _prompt_hash("q", "schema", "SYS", "v1")
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_prompt_hash_is_sensitive_to_inputs():
    base = _prompt_hash("q", "schema", "SYS", "v1")
    assert _prompt_hash("q2", "schema", "SYS", "v1") != base
    assert _prompt_hash("q", "schema2", "SYS", "v1") != base
    assert _prompt_hash("q", "schema", "SYS2", "v1") != base
    assert _prompt_hash("q", "schema", "SYS", "v2") != base


def test_generate_sql_populates_hex_hash():
    fake = FakeClient(content="SELECT 1", usage=_FakeUsage(1, 1, 2))
    result = _make_client(fake).generate_sql("q", "schema")
    assert len(result.prompt_hash) == 64
    int(result.prompt_hash, 16)  # raises if not valid hex


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #

def test_latency_is_captured(monkeypatch):
    # Two perf_counter() reads: start=100.0, end=100.25 -> 250 ms.
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(llm_client.time, "perf_counter", lambda: next(ticks))

    fake = FakeClient(content="SELECT 1", usage=_FakeUsage(1, 1, 2))
    result = _make_client(fake).generate_sql("q", "schema")
    assert result.latency_ms == 250


# --------------------------------------------------------------------------- #
# Token accounting                                                             #
# --------------------------------------------------------------------------- #

def test_token_accounting_present():
    fake = FakeClient(content="SELECT 1", usage=_FakeUsage(10, 20, 30))
    result = _make_client(fake).generate_sql("q", "schema")
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (10, 20, 30)


def test_token_accounting_absent():
    fake = FakeClient(content="SELECT 1", usage=None)
    result = _make_client(fake).generate_sql("q", "schema")
    assert result.ok
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
    assert result.estimated_cost is None  # no tokens -> no cost


# --------------------------------------------------------------------------- #
# Cost estimation                                                              #
# --------------------------------------------------------------------------- #

def test_cost_estimation_known_model():
    # gpt-4o-mini priced at (0.00015, 0.00060) per 1K tokens.
    cost = _estimate_cost("gpt-4o-mini", 1000, 1000)
    assert cost == pytest.approx(0.00075)


def test_cost_estimation_unknown_model():
    assert _estimate_cost("totally-made-up-model", 1000, 1000) is None


def test_cost_estimation_missing_tokens():
    assert _estimate_cost("gpt-4o-mini", None, 5) is None
    assert _estimate_cost("gpt-4o-mini", 5, None) is None


def test_cost_flows_into_result_for_known_model():
    fake = FakeClient(content="SELECT 1", usage=_FakeUsage(1000, 1000, 2000))
    result = _make_client(fake, model="gpt-4o-mini").generate_sql("q", "schema")
    assert result.estimated_cost == pytest.approx(0.00075)


# --------------------------------------------------------------------------- #
# Result shape                                                                 #
# --------------------------------------------------------------------------- #

def test_generation_result_shape():
    assert GenerationResult._fields == (
        "sql",
        "model",
        "prompt_version",
        "prompt_hash",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost",
        "latency_ms",
        "error",
    )


def test_ok_property():
    ok_result = GenerationResult(
        sql="SELECT 1", model="m", prompt_version="v1", prompt_hash="x" * 64,
        prompt_tokens=None, completion_tokens=None, total_tokens=None,
        estimated_cost=None, latency_ms=0, error=None,
    )
    bad_result = ok_result._replace(error="failure")
    assert ok_result.ok is True
    assert bad_result.ok is False


# --------------------------------------------------------------------------- #
# Architecture guard                                                           #
# --------------------------------------------------------------------------- #

def test_llm_client_has_no_forbidden_imports():
    """llm_client.py must not import web-framework, ORM, or UI packages."""
    src = Path(llm_client.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for forbidden in ("fastapi", "sqlalchemy", "streamlit"):
        assert forbidden not in imported, f"forbidden import: {forbidden}"
