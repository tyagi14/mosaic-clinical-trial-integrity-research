##############################################################################
# tests/test_lifecycle.py
#
# WHAT THIS FILE TESTS:
#   api/main.py — startup and shutdown
#   api/db.py   — the shared pool's create-once / close-once behaviour
#   api/dependencies.py — that the injected singletons really are single
#
# WHY STARTUP AND SHUTDOWN DESERVE TESTS:
#   Startup seeds the default reasoning rules. If that step silently
#   stopped running, agents would load zero rules on a fresh database
#   and reason with nothing to constrain them — and the API would still
#   report itself healthy.
#
#   Shutdown closes database pools. Leaked connections do not fail
#   locally; they fail later, in production, when Cloud SQL's connection
#   limit is reached and every request starts erroring at once.
##############################################################################


import pytest

from api import db, main


class FakeProceduralStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.initialised = False
        self.closed = False

    async def initialise_defaults(self):
        if self.fail:
            raise ConnectionError("procedures table unreachable")
        self.initialised = True

    async def close(self):
        self.closed = True


class FakeHITLGate:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    async def close(self):
        if self.fail:
            raise ConnectionError("pool already gone")
        self.closed = True


class TestLifespan:

    async def test_startup_seeds_the_default_reasoning_rules(self, monkeypatch):
        procedural = FakeProceduralStore()
        gate = FakeHITLGate()

        monkeypatch.setattr(main, "get_procedural_store", lambda: procedural)
        monkeypatch.setattr(main, "get_hitl_gate", lambda: gate)
        monkeypatch.setattr(main, "close_pool", _noop)

        async with main.lifespan(main.app):
            assert procedural.initialised is True
        # Seeded before the app serves a request, so the very first
        # /analyze on a fresh database still has rules to reason with.

    async def test_shutdown_closes_the_pools(self, monkeypatch):
        gate = FakeHITLGate()
        closed = []

        async def record_close():
            closed.append(True)

        monkeypatch.setattr(main, "get_procedural_store", lambda: FakeProceduralStore())
        monkeypatch.setattr(main, "get_hitl_gate", lambda: gate)
        monkeypatch.setattr(main, "close_pool", record_close)

        async with main.lifespan(main.app):
            pass

        assert gate.closed is True
        assert closed == [True]
        # Both pools. Closing one and leaking the other is the kind of
        # omission that only shows up as connection exhaustion in
        # production, days later.

    async def test_a_failed_seed_does_not_stop_the_api_from_starting(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            main, "get_procedural_store", lambda: FakeProceduralStore(fail=True)
        )
        monkeypatch.setattr(main, "get_hitl_gate", lambda: FakeHITLGate())
        monkeypatch.setattr(main, "close_pool", _noop)

        async with main.lifespan(main.app):
            pass
        # Logged, not raised. A database hiccup at boot must not leave
        # the container in a crash loop — /api/v1/health is then free to
        # report "degraded" so the operator can see what is wrong.

    async def test_a_failed_close_does_not_skip_the_other_close(
        self, monkeypatch
    ):
        closed = []

        async def record_close():
            closed.append(True)

        monkeypatch.setattr(main, "get_procedural_store", lambda: FakeProceduralStore())
        monkeypatch.setattr(main, "get_hitl_gate", lambda: FakeHITLGate(fail=True))
        monkeypatch.setattr(main, "close_pool", record_close)

        async with main.lifespan(main.app):
            pass

        assert closed == [True]
        # The gate's close raised, and the read pool was still closed.
        # Each step has its own try/except for exactly this reason.


class TestSharedPool:

    async def test_the_pool_is_created_once_and_reused(self, monkeypatch):
        created = []

        async def fake_create_pool(**kwargs):
            created.append(kwargs)
            return _FakePool()

        monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setattr(db, "_pool", None)

        first  = await db.get_pool()
        second = await db.get_pool()

        assert first is second
        assert len(created) == 1
        # One pool per process. The previous code built one per REQUEST,
        # which cost a TCP connect and auth handshake every time and
        # could exhaust the instance's connection limit under load.

        await db.close_pool()

    async def test_closing_resets_so_a_later_call_reconnects(self, monkeypatch):
        async def fake_create_pool(**kwargs):
            return _FakePool()

        monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setattr(db, "_pool", None)

        pool = await db.get_pool()
        await db.close_pool()

        assert pool.closed is True
        assert db._pool is None

    async def test_closing_an_unopened_pool_is_safe(self, monkeypatch):
        monkeypatch.setattr(db, "_pool", None)

        await db.close_pool()
        # A process that only ever served the root endpoint never opens
        # a pool, and its shutdown must not fail because of that.

    async def test_the_pool_is_configured_from_settings(self, monkeypatch):
        from config.settings import settings

        captured = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return _FakePool()

        monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setattr(db, "_pool", None)

        await db.get_pool()

        assert captured["host"] == settings.db_host
        assert captured["database"] == settings.db_name
        assert captured["max_size"] == db.POOL_MAX_SIZE

        await db.close_pool()


class TestDependencySingletons:
    """
    api/dependencies.py caches each store with lru_cache.

    The caching is the point: each store holds a connection pool, and a
    new store per request would mean a new pool per request.
    """

    def test_each_store_is_returned_as_the_same_instance(self):
        from api import dependencies

        assert dependencies.get_hitl_gate() is dependencies.get_hitl_gate()
        assert dependencies.get_episodic_store() is dependencies.get_episodic_store()
        assert dependencies.get_procedural_store() is dependencies.get_procedural_store()
        assert dependencies.get_semantic_store() is dependencies.get_semantic_store()

    def test_the_graph_is_the_one_compiled_at_import(self):
        from api import dependencies
        from graph.graph_builder import mosaic_graph

        assert dependencies.get_graph() is mosaic_graph
        # Not cached with lru_cache because it does not need to be — the
        # graph is already a module-level singleton, compiled once.


# ── HELPERS ───────────────────────────────────────────────────────────

async def _noop():
    return None


class _FakePool:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True
