-- ============================================================================
-- deployment/gcp/cloudsql_init.sql
--
-- PURPOSE:
--   The complete MOSAIC database schema for Cloud SQL (PostgreSQL 15 + pgvector).
--
-- SAFE TO RUN REPEATEDLY:
--   Every statement is idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS),
--   so running this against an existing database only fills in what is missing.
--   It never drops data.
--
-- HOW TO APPLY:
--   psql "host=$DB_HOST port=$DB_PORT dbname=$DB_NAME user=$DB_USER" \
--        -f deployment/gcp/cloudsql_init.sql
--
-- TABLE MAP:
--   studies           raw study records from ClinicalTrials.gov   (ingestion)
--   chunks            embedded text chunks for semantic search    (processing)
--   episodes          episodic memory — past agent sessions       (memory)
--   procedures        procedural memory — agent reasoning rules   (memory)
--   sponsor_profiles  semantic memory — sponsor credibility       (memory)
--   signals           every signal an agent generated             (graph/hitl)
--   hitl_reviews      the human review queue and its decisions    (graph/hitl)
-- ============================================================================

-- pgvector gives us the VECTOR column type and the <=> cosine distance
-- operator that every semantic search in MOSAIC depends on.
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid() is the default for every UUID primary key below.
-- It is BUILT IN from PostgreSQL 13 onward, so no pgcrypto extension is
-- needed. We deliberately do not CREATE EXTENSION pgcrypto here: the
-- application database user is not a superuser and the statement would
-- fail with a permission error on Cloud SQL.


-- ── STUDIES ─────────────────────────────────────────────────────────────────
-- One row per clinical trial, written by processing/vector_store.save_study().
CREATE TABLE IF NOT EXISTS studies (
    nct_id              TEXT PRIMARY KEY,
    title               TEXT,
    sponsor             TEXT,
    phase               TEXT,
    status              TEXT,
    conditions          TEXT[],
    interventions       TEXT[],
    primary_outcome     TEXT,
    secondary_outcomes  TEXT[],
    start_date          TEXT,
    completion_date     TEXT,
    results_posted      BOOLEAN,
    enrollment          INTEGER,
    gcs_path            TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS studies_sponsor_idx ON studies (sponsor);
CREATE INDEX IF NOT EXISTS studies_status_idx  ON studies (status);


-- ── CHUNKS ──────────────────────────────────────────────────────────────────
-- Embedded text chunks. 1536 dimensions = text-embedding-3-small.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id       TEXT REFERENCES studies (nct_id),
    chunk_text   TEXT,
    embedding    VECTOR(1536),
    chunk_index  INTEGER,
    source       TEXT
);

CREATE INDEX IF NOT EXISTS chunks_nct_id_idx ON chunks (nct_id);

-- NOTE ON THE MISSING ivfflat INDEX — READ BEFORE ADDING ONE.
--
-- There is deliberately NO vector index on chunks.embedding or on
-- episodes.embedding. Without one, pgvector does an exact scan, which
-- at these table sizes (hundreds of chunks, tens of episodes) takes
-- single-digit milliseconds and returns perfect recall.
--
-- An ivfflat index was tried here and actively broke search. ivfflat is
-- APPROXIMATE: it clusters the vectors into `lists` buckets and, by
-- default, only looks inside ONE bucket per query (ivfflat.probes = 1).
-- Two things went wrong:
--
--   1. lists = 100 on a table of ~146 rows means most buckets are empty
--      and a bucket holds one or two vectors. One probe therefore
--      searched a nearly empty slice of the table.
--   2. The index on episodes was built when the table was EMPTY, so its
--      cluster centroids were meaningless. PostgreSQL even warns:
--      "ivfflat index created with little data ... This will cause low
--      recall."
--
-- The symptom was silent and misleading: semantic search returned 0 or 1
-- rows where it should have returned 5, with no error anywhere. Agents
-- appeared to have no episodic memory at all.
--
-- ADD AN INDEX ONLY ONCE A TABLE HOLDS REAL VOLUME (100k+ rows), build
-- it on populated data, and size it properly:
--
--   lists ≈ rows / 1000          (for under 1M rows)
--   SET ivfflat.probes = sqrt(lists);   -- per session, at query time
--
--   CREATE INDEX chunks_embedding_idx ON chunks
--       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- Then re-check recall against known-good queries before trusting it.


-- ── EPISODES (episodic memory) ──────────────────────────────────────────────
-- One row per agent reasoning session, written by memory/episodic_store.py.
-- episode_id is generated in Python (uuid4) and passed in explicitly.
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name   TEXT NOT NULL,
    nct_id       TEXT,
    content      TEXT NOT NULL,
    outcome      TEXT,
    embedding    VECTOR(1536),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- No ivfflat index here either — see the note under the chunks table.
CREATE INDEX IF NOT EXISTS episodes_agent_idx      ON episodes (agent_name);
CREATE INDEX IF NOT EXISTS episodes_created_at_idx ON episodes (created_at DESC);


-- ── PROCEDURES (procedural memory) ──────────────────────────────────────────
-- Agent reasoning rules: 'default' rules seeded at startup, plus 'learned'
-- rules written by the HITL rejection loop.
--
-- The UNIQUE constraint matters: ProceduralStore.initialise_defaults() uses
-- INSERT ... ON CONFLICT DO NOTHING and runs on EVERY API startup. Without
-- a unique constraint on (agent_name, rule_text) there is nothing to conflict
-- on, and the default rules would be duplicated on every restart.
CREATE TABLE IF NOT EXISTS procedures (
    procedure_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name    TEXT NOT NULL,
    rule_text     TEXT NOT NULL,
    rule_type     TEXT NOT NULL DEFAULT 'default',
    source        TEXT NOT NULL DEFAULT 'default',
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT procedures_agent_rule_unique UNIQUE (agent_name, rule_text)
);

CREATE INDEX IF NOT EXISTS procedures_agent_idx ON procedures (agent_name);


-- ── SPONSOR_PROFILES (semantic memory) ──────────────────────────────────────
-- One living credibility profile per sponsor.
CREATE TABLE IF NOT EXISTS sponsor_profiles (
    sponsor            TEXT PRIMARY KEY,
    credibility_score  DOUBLE PRECISION DEFAULT 0.5,
    total_studies      INTEGER DEFAULT 0,
    results_posted     INTEGER DEFAULT 0,
    results_missing    INTEGER DEFAULT 0,
    broken_promises    INTEGER DEFAULT 0,
    avg_delay_days     DOUBLE PRECISION DEFAULT 0.0,
    last_updated       TIMESTAMPTZ DEFAULT NOW()
);

-- The original hand-built table was missing results_missing, which
-- memory/semantic_store.py both reads and writes. Add it if absent.
ALTER TABLE sponsor_profiles
    ADD COLUMN IF NOT EXISTS results_missing INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS sponsor_credibility_idx
    ON sponsor_profiles (credibility_score ASC);


-- ── SIGNALS ─────────────────────────────────────────────────────────────────
-- Every signal any agent generated. status: pending | approved | rejected.
CREATE TABLE IF NOT EXISTS signals (
    signal_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id       TEXT REFERENCES studies (nct_id),
    agent        TEXT,
    signal_type  TEXT,
    summary      TEXT,
    evidence     JSONB,
    confidence   DOUBLE PRECISION,
    status       TEXT DEFAULT 'pending',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS signals_agent_idx      ON signals (agent);
CREATE INDEX IF NOT EXISTS signals_status_idx     ON signals (status);
CREATE INDEX IF NOT EXISTS signals_created_at_idx ON signals (created_at DESC);


-- ── HITL_REVIEWS ────────────────────────────────────────────────────────────
-- The human review queue. One row per signal that needed a human decision.
-- decision: pending | approve | reject | edit
CREATE TABLE IF NOT EXISTS hitl_reviews (
    review_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id         UUID REFERENCES signals (signal_id),
    reviewer          TEXT,
    decision          TEXT DEFAULT 'pending',
    edit_summary      TEXT,
    rejection_reason  TEXT,
    reviewed_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS hitl_decision_idx ON hitl_reviews (decision);
