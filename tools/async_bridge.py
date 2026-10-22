##############################################################################
# tools/async_bridge.py
#
# PURPOSE:
#   One safe way to call an ASYNC function from a SYNCHRONOUS LangGraph tool.
#
# WHAT PROBLEM DOES THIS SOLVE:
#   Every @tool function in tools/ is synchronous — LangGraph calls them
#   without await. But everything underneath them is async: asyncpg queries,
#   httpx API calls, OpenAI embeddings.
#
#   The obvious bridge looks like this, and it is WRONG:
#
#       loop = asyncio.get_event_loop()
#       return loop.run_until_complete(coroutine)      # ← breaks
#
#   It breaks because the agents call their tools from INSIDE an async
#   function, which means an event loop is already running on this thread:
#
#       RuntimeError: This event loop is already running
#       RuntimeError: asyncio.run() cannot be called from a running event loop
#
#   A loop cannot run a second loop inside itself. Every single tool call
#   failed this way before this module existed.
#
# THE SOLUTION — ONE DEDICATED BACKGROUND LOOP:
#   We start exactly ONE event loop on its own daemon thread, once, and
#   hand every coroutine to it with asyncio.run_coroutine_threadsafe().
#   The calling thread then blocks on the result.
#
#       caller's loop (thread A)         bridge loop (thread B)
#       ───────────────────────          ──────────────────────
#       run_async(coro)  ──────────────► coro runs here
#       .result() blocks                 ...
#                        ◄────────────── returns value
#
# WHY NOT JUST asyncio.run() IN A THROWAWAY THREAD PER CALL:
#   Because asyncpg connection pools are BOUND to the event loop that
#   created them. A pool opened on a throwaway loop is dead the moment
#   that loop closes, and reusing it from a different loop raises
#   "got Future attached to a different loop".
#
#   The shared stores in tools/search_tools.py are module-level singletons
#   that open their pools lazily on first use. Routing every call through
#   ONE long-lived loop means every pool is created on that loop and
#   reused by every later call — which is exactly what those singletons
#   were designed for.
#
# HOW TO USE:
#   from tools.async_bridge import run_async
#   result = run_async(some_store.some_async_method(...))
##############################################################################


import asyncio
import threading
from typing import Any, Coroutine, TypeVar

from config.logging_config import setup_logging

logger = setup_logging(__name__)


DEFAULT_TIMEOUT_SECONDS = 180.0
# Hard ceiling on how long any single tool call may block.
# Agents run up to 10 reasoning iterations and each may call several tools —
# without a timeout, one hung database or API call would hang the whole
# analysis run forever. 180s is generous for a pgvector query or a
# ClinicalTrials.gov fetch, while still guaranteeing the run ends.

T = TypeVar("T")


_loop: asyncio.AbstractEventLoop | None = None
# The single shared background event loop. None until first use.

_loop_lock = threading.Lock()
# Guards creation of the loop.
# Six agent nodes run in parallel, so two threads could reach
# _ensure_loop() at the same moment. Without the lock, both would
# see _loop is None and each start its own loop — defeating the
# entire purpose of this module.


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    Returns the shared background event loop, starting it on first call.

    The loop runs on a daemon thread. "Daemon" means Python will not wait
    for this thread when the process exits — the API server can shut down
    cleanly without us having to stop the loop by hand.

    Returns:
        The one and only bridge event loop for this process.
    """
    global _loop

    if _loop is not None:
        # Fast path — the loop already exists. No lock needed to read it,
        # because once set it is never reassigned.
        return _loop

    with _loop_lock:
        if _loop is not None:
            # Another thread created it while we waited for the lock.
            return _loop

        loop = asyncio.new_event_loop()

        thread = threading.Thread(
            target=loop.run_forever,
            name="mosaic-async-bridge",
            daemon=True,
        )
        thread.start()
        # run_forever() blocks this new thread and processes whatever
        # coroutines we submit to it, for the lifetime of the process.

        _loop = loop
        logger.info("Async bridge loop started | thread=mosaic-async-bridge")
        return _loop


def run_async(
    coroutine: Coroutine[Any, Any, T],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """
    Runs an async coroutine from synchronous code and returns its result.

    Safe to call from anywhere: inside a running event loop (which is how
    LangGraph calls tools) or from plain synchronous code such as a test.

    Args:
        coroutine: An unawaited async call, e.g. store.search_episodes(...)
        timeout:   Maximum seconds to wait before giving up.

    Returns:
        Whatever the coroutine would have returned with await.

    Raises:
        TimeoutError: If the coroutine did not finish within timeout.
                      The coroutine is cancelled before this is raised,
                      so it cannot keep holding a database connection.
        Exception:    Any exception the coroutine itself raised is
                      re-raised here unchanged, so callers can handle
                      real errors normally.
    """
    loop = _ensure_loop()

    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    # Hands the coroutine to the bridge loop in a thread-safe way and
    # returns a concurrent.futures.Future — NOT an asyncio future.
    # That distinction matters: .result() on this future blocks the
    # calling thread, which is exactly the synchronous behaviour
    # LangGraph tools need.

    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        # Cancel the coroutine so it stops work and releases any
        # database connection it was holding.
        logger.error(f"Async bridge call timed out | timeout={timeout}s")
        raise
