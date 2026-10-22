##############################################################################
# api/routers/signals.py
#
# PURPOSE:
#   Read-only access to the signals table — everything the six agents
#   have ever reported, approved or otherwise.
#
# ENDPOINTS:
#   GET /api/v1/signals              list signals, with optional filters
#   GET /api/v1/signals/{signal_id}  one signal in full, including evidence
#
# WHY THESE QUERIES ARE RAW SQL AND NOT A STORE CLASS:
#   The signals table is written by graph/hitl.py and read here. There is
#   no reasoning attached to reading it — no embeddings, no scoring, no
#   learning — so a store class would add a layer without adding meaning.
#   The connection pool is shared through api/db.py.
##############################################################################


from fastapi import APIRouter, HTTPException, Query

from api.db import get_pool
from api.schemas import SignalResponse
from config.logging_config import setup_logging

logger = setup_logging(__name__)

router = APIRouter()


@router.get(
    "/signals",
    response_model=list[SignalResponse],
    summary="List all signals",
    description="Returns all generated signals with optional filters.",
    tags=["Signals"],
)
async def list_signals(
    agent: str = Query(default=None, description="Filter by agent name"),
    signal_type: str = Query(default=None, description="Filter by signal type"),
    status: str = Query(
        default=None,
        description="Filter by status: approved, pending, rejected",
    ),
    limit: int = Query(
        default=50, ge=1, le=200,
        description="Maximum signals to return",
    ),
) -> list[dict]:
    """
    Returns signals from the signals table, newest first.

    All three filters are optional and combine with AND.

    Args:
        agent:       Only this agent's signals, e.g. "timeline_agent".
        signal_type: Only this type, e.g. "missing_results".
        status:      approved | pending | rejected.
        limit:       Maximum rows to return (1-200).

    Returns:
        A list of signal dicts matching api.schemas.SignalResponse.
    """
    conditions: list[str] = []
    params: list = []
    index = 1

    # ── BUILD THE WHERE CLAUSE SAFELY ─────────────────────────
    # Only the placeholder numbers ($1, $2...) are formatted into the
    # SQL string. Every user-supplied VALUE is passed separately to
    # conn.fetch() and never interpolated, so a filter value cannot
    # alter the query. This is parameterised SQL, not string building,
    # and it is what makes the endpoint injection-proof.

    if agent:
        conditions.append(f"agent = ${index}")
        params.append(agent)
        index += 1

    if signal_type:
        conditions.append(f"signal_type = ${index}")
        params.append(signal_type)
        index += 1

    if status:
        conditions.append(f"status = ${index}")
        params.append(status)
        index += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    params.append(limit)
    # LIMIT is always the last parameter, whatever filters were used.

    query = f"""
        SELECT signal_id, nct_id, agent, signal_type,
               summary, confidence, status,
               created_at::text AS created_at
        FROM signals
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${index}
    """
    # created_at::text casts the timestamp in Postgres so it arrives as
    # a string and needs no serialisation on the way out.

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
    except Exception as e:
        logger.error(f"list_signals failed | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read signals: {e}",
        )

    return [
        {
            "signal_id":   str(row["signal_id"]),
            "nct_id":      row["nct_id"] or "",
            "agent":       row["agent"] or "",
            "signal_type": row["signal_type"] or "",
            "summary":     row["summary"] or "",
            "confidence":  float(row["confidence"] or 0.0),
            "status":      row["status"] or "",
            "created_at":  row["created_at"],
        }
        for row in rows
    ]
    # Built explicitly rather than dict(row) so a UUID arrives as a
    # string and a NULL nct_id — which a cross-study pattern signal has
    # by design — arrives as "" rather than null.


@router.get(
    "/signals/{signal_id}",
    summary="Get one signal by ID",
    description="Returns one signal in full, including its evidence list.",
    tags=["Signals"],
)
async def get_signal(signal_id: str) -> dict:
    """
    Returns the complete record for one signal.

    Args:
        signal_id: The signal's UUID.

    Returns:
        The signal as a dict.

    Raises:
        HTTPException 404: No signal with that id, or the id is not a
                           valid UUID. Both are "you asked for something
                           that does not exist", so both are a 404
                           rather than a 500.
    """
    import json
    import uuid

    try:
        signal_uuid = uuid.UUID(signal_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=404,
            detail=f"'{signal_id}' is not a valid signal id.",
        )
        # Validated before the query because signals.signal_id is a UUID
        # column: passing a non-UUID string would make asyncpg raise a
        # DataError, which would surface to the caller as a 500.

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT signal_id, nct_id, agent, signal_type,
                       summary, evidence, confidence, status,
                       created_at::text AS created_at
                FROM signals
                WHERE signal_id = $1
                """,
                signal_uuid,
            )
    except Exception as e:
        logger.error(f"get_signal failed | signal_id={signal_id} | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read signal: {e}",
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Signal {signal_id} not found.",
        )

    evidence = row["evidence"]
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except (json.JSONDecodeError, ValueError):
            evidence = [evidence]
        # evidence is a JSONB column and asyncpg hands JSONB back as a
        # raw JSON string unless a codec is registered. Decoding it here
        # means the API returns a real list instead of a string that
        # merely looks like one.

    return {
        "signal_id":   str(row["signal_id"]),
        "nct_id":      row["nct_id"] or "",
        "agent":       row["agent"] or "",
        "signal_type": row["signal_type"] or "",
        "summary":     row["summary"] or "",
        "evidence":    evidence or [],
        "confidence":  float(row["confidence"] or 0.0),
        "status":      row["status"] or "",
        "created_at":  row["created_at"],
    }
