##############################################################################
# tests/conftest.py
#
# PURPOSE:
#   Shared setup and fixtures for the whole test suite.
#
# TWO RULES THIS SUITE FOLLOWS:
#
#   1. NO NETWORK, NO DATABASE, NO OPENAI.
#      Every test here runs offline. Tests that need a store or an LLM get
#      a fake one. This keeps the suite fast, free and deterministic — a
#      test that calls gpt-4o costs money, gives a different answer each
#      run, and fails when the internet is down. None of that tells you
#      whether the code is correct.
#
#   2. NO REAL .env REQUIRED.
#      config/settings.py validates required variables at import time and
#      raises if any are missing, so importing almost any module without a
#      .env file would fail at collection. The block below fills in dummy
#      values for anything not already set, BEFORE the first project
#      import happens. Real values in a local .env still win.
##############################################################################


import os

# ── ENVIRONMENT MUST BE SET BEFORE ANY PROJECT IMPORT ─────────────────
# Loaded first because config.settings reads these at import time.
_TEST_ENV_DEFAULTS = {
    "OPENAI_API_KEY":   "sk-test-not-a-real-key",
    "GCP_PROJECT_ID":   "test-project",
    "GCS_BUCKET_NAME":  "test-bucket",
    "DB_HOST":          "localhost",
    "DB_PORT":          "5432",
    "DB_NAME":          "test_db",
    "DB_USER":          "test_user",
    "DB_PASSWORD":      "test-password",
    "LANGSMITH_API_KEY": "",
    "API_ENV":          "development",
}

for _key, _value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

import pytest  # noqa: E402  — must come after the env defaults above


# ── RAW API FIXTURES ──────────────────────────────────────────────────

@pytest.fixture
def raw_study() -> dict:
    """
    One raw study in ClinicalTrials.gov API v2 shape.

    Trimmed to the fields ingestion/document_parser.py actually reads,
    but with the real nesting preserved — the deep nesting is the whole
    reason the parser exists, so flattening it here would test nothing.

    This study is a results-posting violation: COMPLETED in 2019 with no
    resultsSection at all.
    """
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT04788680",
                "briefTitle": "A Short Title",
                "officialTitle": "A Study of Semaglutide in Type 2 Diabetes",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2017-03"},
                "primaryCompletionDateStruct": {"date": "2019-06"},
                "completionDateStruct": {"date": "2019-08"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Novo Nordisk A/S"},
            },
            "conditionsModule": {
                "conditions": ["Type 2 Diabetes", "Obesity"],
            },
            "designModule": {
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 1200},
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Change in HbA1c at week 52"}],
                "secondaryOutcomes": [
                    {"measure": "Change in body weight"},
                    {"measure": "Proportion achieving HbA1c < 7%"},
                ],
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "Semaglutide"},
                    {"name": "Placebo"},
                    {"name": ""},
                    # Empty name included on purpose: the parser must
                    # drop it rather than storing a blank intervention.
                ],
            },
        },
        # No "resultsSection" key → results_posted must come out False.
    }


@pytest.fixture
def raw_study_with_results(raw_study) -> dict:
    """The same study, but with results posted."""
    return {
        **raw_study,
        "resultsSection": {
            "baselineCharacteristicsModule": {"populationDescription": "x"},
        },
    }
    # Built as a new dict rather than mutating raw_study — fixtures are
    # shared, and a mutated fixture is a test that passes or fails
    # depending on which test ran first.


@pytest.fixture
def parsed_study(raw_study):
    """A ParsedStudy produced by the real parser from raw_study."""
    from ingestion.document_parser import DocumentParser
    study = DocumentParser().parse_study(raw=raw_study)
    assert study is not None, "fixture setup failed: parser returned None"
    return study


@pytest.fixture
def signal() -> dict:
    """One well-formed SignalOutput, as an agent would produce it."""
    return {
        "agent":       "missing_results_agent",
        "signal_type": "missing_results",
        "nct_id":      "NCT04788680",
        "summary":     "Completed 2019-06, no results posted after 5 years.",
        "evidence":    ["status: COMPLETED", "completion: 2019-06"],
        "confidence":  0.85,
    }


# ── FAKE DATABASE ─────────────────────────────────────────────────────
#
# A minimal stand-in for an asyncpg pool. It exists so the SQL-facing
# code — graph/hitl.py and api/routers/signals.py — can be tested for
# its LOGIC: which status a signal gets, whether a rejection reaches
# procedural memory, whether a bad UUID is a 404 rather than a 500.
#
# It does NOT pretend to be a database. It records the queries it was
# given and returns pre-loaded answers, so what is under test is our
# branching, not Postgres. Whether the SQL itself is valid is proven by
# running the system against the real Cloud SQL instance, not here.

class FakeConnection:
    """Returns queued answers and records every statement it was given."""

    def __init__(self, fetchval=None, fetchrow=None, fetch=None):
        self._fetchval = list(fetchval or [])
        self._fetchrow = list(fetchrow or [])
        self._fetch    = list(fetch or [])
        self.statements: list[tuple[str, tuple]] = []

    async def fetchval(self, query, *args):
        self.statements.append((query, args))
        return self._fetchval.pop(0) if self._fetchval else None

    async def fetchrow(self, query, *args):
        self.statements.append((query, args))
        return self._fetchrow.pop(0) if self._fetchrow else None

    async def fetch(self, query, *args):
        self.statements.append((query, args))
        return self._fetch.pop(0) if self._fetch else []

    async def execute(self, query, *args):
        self.statements.append((query, args))
        return "OK"

    def transaction(self):
        return _NullAsyncContext()

    def executed_sql(self) -> str:
        """Every statement this connection saw, as one searchable string."""
        return "\n".join(sql for sql, _ in self.statements)


class FakePool:
    """Hands out the same FakeConnection to every acquire()."""

    def __init__(self, connection: "FakeConnection"):
        self.connection = connection
        self.closed = False

    def acquire(self):
        return _ConnectionContext(self.connection)

    async def close(self):
        self.closed = True


class _ConnectionContext:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc_info):
        return False


class _NullAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def fake_connection():
    """A FakeConnection with no queued answers — load them per test."""
    return FakeConnection()
