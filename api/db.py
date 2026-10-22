##############################################################################
# api/db.py
#
# PURPOSE:
#   One shared PostgreSQL connection pool for the API's direct SQL reads.
#
# WHAT PROBLEM DOES THIS SOLVE:
#   The read-only endpoints — listing signals, fetching one signal, the
#   health check — query tables that no store class owns, so they used
#   raw asyncpg. Each of them built its own pool inline, per request:
#
#       pool = await asyncpg.create_pool(host=..., port=..., ...)
#       try:    ... one query ...
#       finally: await pool.close()
#
#   That is a TCP connect plus a password handshake on every single API
#   call, for a pool that is thrown away immediately. Worse, the same six
#   connection arguments were copy-pasted into three files, so Cloud SQL
#   connection settings had three places to drift apart.
#
#   Under load it is also a way to exhaust the database. Cloud SQL caps
#   total connections; a per-request pool means concurrent requests each
#   claim their own connections instead of sharing a fixed set.
#
# THE RULE THIS MODULE ENFORCES:
#   One pool per process, created on first use, closed once on shutdown.
#
# HOW TO USE:
#   from api.db import get_pool
#   pool = await get_pool()
#   async with pool.acquire() as conn:
#       rows = await conn.fetch("SELECT ...")
#
#   Never call pool.close() at a call site — the pool is shared, and
#   api/main.py's lifespan closes it on shutdown.
##############################################################################


import asyncio

import asyncpg

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 5
# Small on purpose. These endpoints run short read queries, and the three
# memory stores plus the HITL gate each hold their own pool against the
# same Cloud SQL instance. The instance-wide connection limit is the
# constraint being respected here, not this pool's own throughput.


_pool: asyncpg.Pool | None = None

_lock = asyncio.Lock()
# Guards creation. Without it, two requests arriving together would both
# see _pool is None and both create a pool — and the second would
# silently replace the first, leaking its connections for the lifetime
# of the process.


async def get_pool() -> asyncpg.Pool:
    """
    Returns the shared connection pool, creating it on first call.

    Returns:
        The process-wide asyncpg pool. Acquire connections from it with
        `async with pool.acquire() as conn`, and do not close it.
    """
    global _pool

    if _pool is not None:
        return _pool

    async with _lock:
        if _pool is not None:
            # Created by another request while this one waited.
            return _pool

        _pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
        )
        logger.info(
            f"API read pool created | "
            f"host={settings.db_host} | database={settings.db_name}"
        )
        return _pool


async def close_pool() -> None:
    """
    Closes the shared pool. Called once, from the API's shutdown handler.

    Safe to call when no pool was ever created — an API process that
    only served the root endpoint never opens one.
    """
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("API read pool closed")
