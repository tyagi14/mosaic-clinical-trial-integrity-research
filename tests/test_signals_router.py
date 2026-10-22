##############################################################################
# tests/test_signals_router.py
#
# WHAT THIS FILE TESTS:
#   api/routers/signals.py — the two read endpoints, with the shared
#   pool from api/db.py replaced by a fake.
#
# WHY SEPARATE FROM test_api.py:
#   Every other router reaches the database through an injected store, so
#   test_api.py can swap the store. These two endpoints run SQL directly
#   against the shared pool, so they need the pool faked instead. Keeping
#   the two styles in separate files keeps each one's setup honest about
#   what it is pretending to be.
#
# WHAT IS ACTUALLY BEING CHECKED:
#   Filter composition (placeholder numbering shifts as filters are
#   added, which is where an off-by-one would appear), the NULL handling
#   that cross-study signals depend on, and the 404-vs-500 distinction.
##############################################################################


import json

import pytest
from httpx import ASGITransport, AsyncClient

from api import db
from api.main import app
from tests.conftest import FakeConnection, FakePool


def signal_row(**overrides):
    row = {
        "signal_id": "123e4567-e89b-12d3-a456-426614174000",
        "nct_id": "NCT04788680",
        "agent": "missing_results_agent",
        "signal_type": "missing_results",
        "summary": "No results posted.",
        "evidence": json.dumps(["status: COMPLETED"]),
        "confidence": 0.9,
        "status": "approved",
        "created_at": "2026-10-16 12:00:00+00",
    }
    return {**row, **overrides}


@pytest.fixture
def fake_db(monkeypatch, fake_connection):
    """Replaces api.db.get_pool with a fake pool."""
    pool = FakePool(fake_connection)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(db, "get_pool", fake_get_pool)

    import api.routers.signals as signals_router
    monkeypatch.setattr(signals_router, "get_pool", fake_get_pool)
    # Patched in both places: api/db.py owns the function, and the
    # router imported it by name at import time, so it holds its own
    # reference to the original.

    return fake_connection


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestListSignals:

    async def test_returns_the_signals(self, client, fake_db):
        fake_db._fetch = [[signal_row()]]

        response = await client.get("/api/v1/signals")

        assert response.status_code == 200
        body = response.json()
        assert body[0]["nct_id"] == "NCT04788680"
        assert body[0]["confidence"] == 0.9

    async def test_no_filters_means_no_where_clause(self, client, fake_db):
        fake_db._fetch = [[]]

        await client.get("/api/v1/signals")

        sql, args = fake_db.statements[0]
        assert "WHERE" not in sql
        assert args == (50,)
        # Only the default limit is bound. An empty WHERE clause would
        # be a syntax error, and a WHERE on empty strings would match
        # nothing instead of everything.

    async def test_filters_are_numbered_in_order(self, client, fake_db):
        fake_db._fetch = [[]]

        await client.get(
            "/api/v1/signals"
            "?agent=timeline_agent&signal_type=timeline_delay&status=approved&limit=5"
        )

        sql, args = fake_db.statements[0]
        assert "agent = $1" in sql
        assert "signal_type = $2" in sql
        assert "status = $3" in sql
        assert "LIMIT $4" in sql
        assert args == ("timeline_agent", "timeline_delay", "approved", 5)
        # LIMIT's placeholder number depends on how many filters were
        # supplied. This is the one place in the project where SQL is
        # assembled dynamically, so the numbering is pinned exactly.

    async def test_one_filter_still_puts_limit_last(self, client, fake_db):
        fake_db._fetch = [[]]

        await client.get("/api/v1/signals?status=pending")

        sql, args = fake_db.statements[0]
        assert "status = $1" in sql
        assert "LIMIT $2" in sql
        assert args == ("pending", 50)

    async def test_results_are_newest_first(self, client, fake_db):
        fake_db._fetch = [[]]

        await client.get("/api/v1/signals")

        assert "ORDER BY created_at DESC" in fake_db.statements[0][0]

    async def test_a_null_nct_id_becomes_an_empty_string(self, client, fake_db):
        fake_db._fetch = [[signal_row(nct_id=None)]]

        response = await client.get("/api/v1/signals")

        assert response.json()[0]["nct_id"] == ""
        # A cross-study pattern signal has no single trial and is stored
        # with nct_id NULL by design. SignalResponse declares nct_id as a
        # string, so null would fail response validation with a 500.

    @pytest.mark.parametrize("limit", [0, -5, 500])
    async def test_an_out_of_range_limit_is_rejected(self, client, fake_db, limit):
        response = await client.get(f"/api/v1/signals?limit={limit}")

        assert response.status_code == 422

    async def test_a_database_failure_is_a_500_with_an_explanation(
        self, client, monkeypatch
    ):
        async def broken_get_pool():
            raise ConnectionError("cloud sql unreachable")

        import api.routers.signals as signals_router
        monkeypatch.setattr(signals_router, "get_pool", broken_get_pool)

        response = await client.get("/api/v1/signals")

        assert response.status_code == 500
        assert "cloud sql unreachable" in response.json()["detail"]


class TestGetOneSignal:

    async def test_returns_the_signal_with_its_evidence_decoded(
        self, client, fake_db
    ):
        fake_db._fetchrow = [signal_row(
            evidence=json.dumps(["status: COMPLETED", "no results section"])
        )]

        response = await client.get(
            "/api/v1/signals/123e4567-e89b-12d3-a456-426614174000"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["evidence"] == ["status: COMPLETED", "no results section"]
        # A real list, not the JSON string asyncpg hands back for a JSONB
        # column. The evidence IS the justification for an accusation, so
        # a caller must be able to iterate it.

    async def test_unparseable_evidence_is_wrapped_not_dropped(
        self, client, fake_db
    ):
        fake_db._fetchrow = [signal_row(evidence="not json at all")]

        response = await client.get(
            "/api/v1/signals/123e4567-e89b-12d3-a456-426614174000"
        )

        assert response.json()["evidence"] == ["not json at all"]
        # Degrading to a one-item list keeps the evidence visible. Losing
        # it silently would remove the only support for the claim.

    async def test_a_missing_signal_is_a_404(self, client, fake_db):
        fake_db._fetchrow = [None]

        response = await client.get(
            "/api/v1/signals/123e4567-e89b-12d3-a456-426614174000"
        )

        assert response.status_code == 404

    @pytest.mark.parametrize("bad_id", ["not-a-uuid", "12345", "NCT04788680"])
    async def test_a_malformed_id_is_a_404_not_a_500(self, client, fake_db, bad_id):
        response = await client.get(f"/api/v1/signals/{bad_id}")

        assert response.status_code == 404
        assert fake_db.statements == []
        # Validated before the query runs. signal_id is a UUID column,
        # and handing asyncpg a non-UUID raises a DataError that would
        # surface as a 500 — but asking for something that cannot exist
        # is a client error.
