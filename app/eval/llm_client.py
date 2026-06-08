"""Text-to-SQL generation layer (Stage 4).

Standalone component: at import time it depends only on the Python standard
library. The OpenAI SDK and the application settings are imported *lazily*, so
merely importing this module (or ``app.eval``) stays lightweight and never
fails because of a missing SDK or unconfigured environment. It imports no
web-framework, ORM, UI, runner, or persistence code.

The single public entry point is :meth:`TextToSqlClient.generate_sql`, which
turns a natural-language ``question`` plus a ``schema_context`` into a
:class:`GenerationResult`. The result carries the generated SQL together with
the metadata that later stages persist to a trace: the model, prompt version, a
deterministic prompt hash, token usage, an estimated cost, and latency.

Guarantees
----------
* **Never raises** — every failure (missing SDK, bad configuration, API error,
  timeout, malformed response) is returned as a :class:`GenerationResult` with
  ``error`` populated, never raised out of :meth:`generate_sql`.
* **Always attributable** — ``model``, ``prompt_version``, ``prompt_hash`` and
  ``latency_ms`` are populated even on failure, so a failed generation can
  still be recorded and reproduced.
* **No execution** — this layer only *generates* SQL. Running it and judging
  correctness belong to the executor and comparator (Stage 3).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:  # import only for type-checkers; never at runtime
    from app.core.config import Settings

# Defaults: SQL outputs are short, so a small token budget and a modest
# wall-clock timeout are plenty.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TOKENS = 512

# Fixed system prompt. Frozen wording so the prompt hash stays stable.
SYSTEM_PROMPT = (
    "You are an expert Text-to-SQL engine. Given a database schema and a "
    "natural-language question, produce a single query that answers it.\n"
    "Rules:\n"
    "- Generate SQLite-compatible SQL only.\n"
    "- Return the SQL query and nothing else.\n"
    "- Do not include any explanation, commentary, or prose.\n"
    "- Do not use markdown formatting or code fences."
)

# Per-model pricing in USD per 1,000 tokens as ``(prompt_rate, completion_rate)``.
# Centralised so it is the single place to update when published prices change;
# figures are public list prices and should be verified periodically. Models not
# listed here yield ``estimated_cost = None`` rather than a wrong number.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4.1": (0.00200, 0.00800),
    "gpt-4.1-mini": (0.00040, 0.00160),
    "gpt-3.5-turbo": (0.00050, 0.00150),
}

# Matches a leading ```/```sql fence line and a trailing ``` fence line.
_OPEN_FENCE_RE = re.compile(r"^```[^\n]*\n?")
_CLOSE_FENCE_RE = re.compile(r"\n?```\s*$")


class GenerationResult(NamedTuple):
    """Outcome of one Text-to-SQL generation.

    On success ``sql`` is populated and ``error`` is ``None``. On failure ``sql``
    (and the token/cost fields) are ``None`` while ``error`` describes what went
    wrong; ``model``, ``prompt_version``, ``prompt_hash`` and ``latency_ms`` are
    populated in both cases.
    """

    sql: str | None
    model: str
    prompt_version: str
    prompt_hash: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None
    latency_ms: int
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_messages(
    question: str, schema_context: str, system_prompt: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Build the chat messages for one generation request."""
    user_prompt = (
        f"Database schema:\n{schema_context}\n\n"
        f"Question: {question}\n\n"
        "Return only the SQLite SQL query."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _prompt_hash(
    question: str, schema_context: str, system_prompt: str, prompt_version: str
) -> str:
    """Deterministic SHA-256 over the inputs that define the prompt.

    Depends on the question, schema context, system prompt and prompt version,
    so identical inputs hash identically and any change is detected. A NUL
    separator keeps the parts unambiguous. Returns a 64-character hex digest.
    """
    payload = "\x00".join((prompt_version, system_prompt, schema_context, question))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences and surrounding whitespace from model output.

    The system prompt forbids fences, but models sometimes add them anyway, so
    this defensively removes a leading ```` ``` ```` / ```` ```sql ```` line and
    a trailing ```` ``` ```` line, then trims.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = _OPEN_FENCE_RE.sub("", text)
        text = _CLOSE_FENCE_RE.sub("", text)
    return text.strip()


def _estimate_cost(
    model: str, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """Estimate request cost in USD, or ``None`` when it cannot be known.

    Returns ``None`` for an unknown model or when either token count is missing,
    so callers never see a confidently-wrong figure.
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        return None
    if prompt_tokens is None or completion_tokens is None:
        return None
    prompt_rate, completion_rate = pricing
    cost = (prompt_tokens / 1000.0) * prompt_rate + (
        completion_tokens / 1000.0
    ) * completion_rate
    return round(cost, 8)


def _extract_sql(response: Any) -> str:
    """Pull the SQL text out of a chat-completion response or raise on garbage."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("malformed response: no choices returned")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        raise ValueError("malformed response: empty message content")
    return content


def _extract_usage(
    response: Any,
) -> tuple[int | None, int | None, int | None]:
    """Pull token usage out of a response; missing usage is ``None`` (not error)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def _elapsed_ms(start: float) -> int:
    """Whole milliseconds elapsed since ``start`` (a ``perf_counter`` reading)."""
    return int(round((time.perf_counter() - start) * 1000))


def _format_error(exc: Exception) -> str:
    """Render an exception as a compact, log-friendly string."""
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Client                                                                       #
# --------------------------------------------------------------------------- #

class TextToSqlClient:
    """Generates SQLite SQL from a question + schema using an LLM.

    Configuration is read from the existing settings layer (loaded lazily); a
    transport ``client`` may be injected for tests. The real OpenAI client is
    built lazily on first use, so constructing this object never requires the
    SDK to be installed.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}.")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}.")

        if settings is None:
            # Lazy: avoids importing the config (and pydantic) at module import.
            from app.core.config import get_settings

            settings = get_settings()

        self._api_key = settings.openai_api_key
        self._base_url = settings.openai_base_url
        self._model = settings.llm_model
        self._temperature = settings.default_temperature
        self._prompt_version = settings.default_prompt_version

        self._timeout_s = float(timeout_s)
        self._max_tokens = int(max_tokens)
        self._client = client  # may be None; built lazily in _get_client()

    def _get_client(self) -> Any:
        """Return the transport client, constructing a real one if needed.

        Lazy import keeps module import light and turns a missing SDK or bad
        configuration into a caught exception (-> ``GenerationResult.error``)
        rather than an import-time crash.
        """
        if self._client is not None:
            return self._client
        from openai import OpenAI  # lazy, intentional

        self._client = OpenAI(
            api_key=self._api_key or None,
            base_url=self._base_url,
            timeout=self._timeout_s,
        )
        return self._client

    def generate_sql(self, question: str, schema_context: str) -> GenerationResult:
        """Generate SQL for ``question`` against ``schema_context``.

        Never raises: any problem is returned as a ``GenerationResult`` whose
        ``error`` is set, with ``model``/``prompt_version``/``prompt_hash``/
        ``latency_ms`` still populated.
        """
        messages = _build_messages(question, schema_context)
        prompt_hash = _prompt_hash(
            question, schema_context, SYSTEM_PROMPT, self._prompt_version
        )
        start = time.perf_counter()

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            sql = _clean_sql(_extract_sql(response))
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            cost = _estimate_cost(self._model, prompt_tokens, completion_tokens)
            return GenerationResult(
                sql=sql,
                model=self._model,
                prompt_version=self._prompt_version,
                prompt_hash=prompt_hash,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
                latency_ms=_elapsed_ms(start),
                error=None,
            )
        except Exception as exc:  # never raise outward
            return GenerationResult(
                sql=None,
                model=self._model,
                prompt_version=self._prompt_version,
                prompt_hash=prompt_hash,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                estimated_cost=None,
                latency_ms=_elapsed_ms(start),
                error=_format_error(exc),
            )
