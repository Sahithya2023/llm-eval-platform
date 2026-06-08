# LLM Evaluation & Observability Platform — for Text-to-SQL Agents

An evaluation and observability platform that measures how well a Text-to-SQL
agent turns natural-language questions into correct SQL. Correctness is judged
by **execution accuracy** (do the generated query's results match the gold
query's results), never by string matching.

The project is being built incrementally. **This is Stage 1: the foundation.**

## Stage 1 scope

Stage 1 establishes the skeleton everything else sits on:

- Production-style project structure
- Environment-driven configuration (`.env`)
- SQLAlchemy setup (engine + session management)
- The `traces` table schema and `Trace` model
- A database initialization script

It does **not** yet include the dataset loader, LLM client, SQL executor,
comparator, runner, API, or dashboard. Those arrive in later stages.

## Requirements

- Python 3.11+

## Setup

```bash
# 1. From the project root, create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local config (optional for Stage 1 — defaults work)
cp .env.example .env

# 4. Initialize the database (creates data/traces.db and the traces table)
python -m app.db.init_db
```

## Expected output

After step 4 you should see:

```
[init_db] Database URL : sqlite:///./data/traces.db
[init_db] Tables ready : traces
[init_db] Stage 1 initialization complete.
```

and a `data/traces.db` file will exist. You can confirm the schema with:

```bash
sqlite3 data/traces.db ".schema traces"
```

## Project layout

```
app/
  core/      configuration
  db/        models, session, init script
  datasets/  (later) dataset loader
  eval/      (later) llm client, executor, comparator, runner
  services/  (later) trace persistence + aggregate stats
  schemas/   (later) API request/response models
  api/       (later) FastAPI routes
dashboard/   (later) Streamlit app
scripts/     (later) CLI entry points
data/        SQLite database (gitignored)
tests/       tests
```

## Stage 2 — Spider dataset loader

Stage 2 adds a standalone loader for the [Spider](https://yale-lily.github.io/spider)
Text-to-SQL benchmark and a script to verify that examples load and that every
`db_id` resolves to a real SQLite file. The loader depends only on the standard
library — it does not import FastAPI, SQLAlchemy, or any evaluation code.

### Install

```bash
pip install -r requirements.txt   # now also installs pytest
```

### Download Spider

Spider is distributed by the Yale LILY group. Download the Spider 1.0 release
from the official project page (linked above; the current download link is in
the Spider GitHub README) and extract it so the contents live under
`data/benchmark/spider/`.

### Set the benchmark path

The loader defaults to `./data/benchmark/spider`. To use a different location,
set it in `.env`:

```bash
SPIDER_DIR=/absolute/path/to/spider
```

### Expected structure under `data/benchmark/`

```
data/benchmark/spider/
├── dev.json
├── train_spider.json
├── train_others.json        (optional)
├── tables.json
└── database/
    ├── concert_singer/
    │   └── concert_singer.sqlite
    ├── pets_1/
    │   └── pets_1.sqlite
    └── ...                   (one folder per db_id)
```

### Verify

```bash
python scripts/check_spider.py
```

Expected (values depend on your data):

```
Loaded examples   : 50
Unique databases  : 17
All 17 databases resolved successfully.
...
Stage 2 check passed: Spider loads and db_id mapping is valid.
```

### Run tests

The tests are hermetic (they build a tiny synthetic dataset), so no Spider
download is required:

```bash
pytest -q
```

## Stage 3 — Execution oracle (executor + comparator)

Stage 3 adds the execution-based correctness oracle. It is the foundation for
every accuracy metric the platform reports: instead of comparing SQL strings,
we **execute** the generated SQL and compare its result set to the gold result.

The evaluation layer stays independent of the API/UI layers — `app/eval` imports
only the Python standard library and can be driven from a CLI or a service
without modification.

### Components

- `app/eval/executor.py` — `SqlExecutor`: runs SQL **read-only** against a
  Spider SQLite database. The database is opened in SQLite `mode=ro`, so any
  write is rejected by the engine. Queries are bounded by a wall-clock timeout
  (via a progress handler) and a row cap. Results come back as a structured
  `ExecutionResult(rows, columns, error, error_type, truncated)`; query failures
  are returned, never raised.
- `app/eval/comparator.py` — the oracle. Compares two result sets by execution
  semantics and returns `Comparison(is_correct, reason, order_considered)`.

### Comparison semantics (frozen)

1. **Multiset (bag) comparison** — row multiplicity matters
   (`SELECT name` ≠ `SELECT DISTINCT name`).
2. **Strict positional columns** — column count and order must match.
3. **Ordering from the gold SQL only** — rows are compared in order iff the
   gold query has a top-level `ORDER BY`, otherwise as an unordered multiset.
4. **Numeric normalization** — `5 == 5.0`, `True == 1`, floats compared with a
   small tolerance.
5. **Strict, exact strings** — no case-folding, no trimming.
6. **NULL equals only NULL**.

### Usage

```python
from app.eval import SqlExecutor, compare_executions

executor = SqlExecutor(timeout_s=5.0)
gold = executor.execute(db_path, gold_sql)
pred = executor.execute(db_path, generated_sql)

verdict = compare_executions(gold, pred, gold_sql)
print(verdict.is_correct, verdict.reason)
```

### Run tests

Hermetic, as before (synthetic SQLite fixtures; no Spider download needed):

```bash
pytest -q
```

## Stage 4 — LLM client (Text-to-SQL generation)

Stage 4 adds the generation layer: it turns a natural-language question plus a
schema description into SQL, capturing the metadata later stages persist to a
trace (model, prompt version, deterministic prompt hash, token usage, estimated
cost, latency). It is standalone — at import time it depends only on the Python
standard library; the OpenAI SDK and the settings layer are imported lazily, so
importing `app.eval` stays lightweight and never fails on a missing SDK. This
stage only *generates* SQL; running and judging it remain the job of the
Stage 3 executor and comparator.

### Behaviour

- `TextToSqlClient.generate_sql(question, schema_context)` returns a
  `GenerationResult` and **never raises**: any failure (missing SDK, bad
  configuration, API error, timeout, malformed response) comes back as
  `GenerationResult.error`, with `model`, `prompt_version`, `prompt_hash` and
  `latency_ms` populated even on failure.
- Model output is defensively cleaned (stray markdown code fences stripped).
- Cost is estimated from a centralised per-model price table; an unknown model
  or missing token usage yields `estimated_cost = None` rather than a guess.

### Configuration

Read from the existing settings (`.env`): `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`LLM_MODEL`, `DEFAULT_TEMPERATURE`, `DEFAULT_PROMPT_VERSION`.

### Usage

```python
from app.eval import TextToSqlClient

client = TextToSqlClient()
result = client.generate_sql(
    question="How many singers are there?",
    schema_context="CREATE TABLE singer (id INTEGER, name TEXT);",
)
if result.ok:
    print(result.sql, result.total_tokens, result.estimated_cost)
else:
    print("generation failed:", result.error)
```

### Run tests

Hermetic — a `FakeClient` stands in for the OpenAI SDK, so no network access or
API key is needed:

```bash
pytest -q
```
