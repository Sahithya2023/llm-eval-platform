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
