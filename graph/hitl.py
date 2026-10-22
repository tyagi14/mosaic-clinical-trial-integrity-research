##############################################################################
# graph/hitl.py
#
# PURPOSE:
#   The Human-In-The-Loop gate — the last stop between an agent's claim
#   and a published signal, and the first step of the learning loop.
#
# WHAT PROBLEM DOES THIS SOLVE:
#   The six specialist agents accuse named sponsors of breaking federal
#   law. Some of those accusations will be wrong. Publishing all of them
#   unchecked would be reckless; sending all of them to a human would
#   make the system pointless. So every signal is routed by how sure the
#   agent is:
#
#       confidence >= 0.6   →  auto-approved, published immediately
#       confidence <  0.6   →  queued for a human analyst to decide
#
#   0.6 is the same threshold agents/supervisor.py uses to split its
#   brief into "findings" and "requires review", so the brief and the
#   queue always agree.
#
# THE LEARNING LOOP — WHY REJECTION IS THE IMPORTANT PATH:
#   When an analyst REJECTS a signal they must say why. That sentence is
#   written straight into procedural memory as a new reasoning rule for
#   the agent that produced the signal:
#
#       analyst: "Terminated trials are exempt — don't flag them."
#            ↓
#       procedures table  (agent_name, rule_text, 'learned')
#            ↓
#       every future run of that agent loads the rule into its system
#       prompt and never makes that mistake again.
#
#   One correction, typed once, permanently changes how the agent
#   reasons. That is the whole point of the gate.
#
# WHERE THIS SITS:
#   The gate runs AFTER the graph, not inside it — api/routers/analysis.py
#   calls process_signal() on each signal the graph produced. Agents stay
#   free of persistence concerns; the gate owns the signals and
#   hitl_reviews tables.
##############################################################################


import json
import re
import uuid
from typing import Any

import asyncpg

from memory.procedural_store import ProceduralStore
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


CONFIDENCE_THRESHOLD = 0.6
# At or above this, a signal is published without human review.
# Below it, a human decides. Keep this in step with the same threshold
# in agents/supervisor.py.

STATUS_APPROVED = "approved"
STATUS_PENDING  = "pending"
STATUS_REJECTED = "rejected"

DECISION_PENDING = "pending"
VALID_DECISIONS  = ("approve", "reject", "edit")

UNKNOWN_STUDY_TITLE = "Not ingested — referenced by a signal"
# Placeholder title for the stub study row described in _ensure_study().

NCT_ID_PATTERN = re.compile(r"^NCT\d{8}$")
# A real ClinicalTrials.gov identifier is "NCT" followed by exactly
# 8 digits.
#
# Agents do not always put one in the nct_id field. The pattern finder,
# whose entire job is cross-study patterns, has no single trial to name
# and writes placeholders such as "PATTERN" or "MULTIPLE". Those must
# never become rows in the studies table — see _ensure_study().

POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 5


class HITLGate:
    """
    Routes every agent signal to publication or to a human reviewer,
    and feeds human rejections back into procedural memory.

    One instance per process — api/dependencies.get_hitl_gate() caches
    it with lru_cache so the whole API shares a single connection pool.

    Usage:
        gate = HITLGate()
        await gate.process_signal(signal)          # after an analysis run
        queue = await gate.get_review_queue()      # what humans must decide
        await gate.process_human_decision(...)     # record a decision
        await gate.close()                         # on shutdown
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        # Opened lazily by _ensure_pool() — see the note there.

        self._procedural = ProceduralStore()
        # The gate owns the write side of the learning loop. It is the
        # only place in the system that turns human words into rules.

        logger.info(
            f"HITLGate initialised | threshold={CONFIDENCE_THRESHOLD}"
        )

    # ── CONNECTION MANAGEMENT ─────────────────────────────────

    async def _ensure_pool(self) -> None:
        """
        Opens the connection pool on first use.

        Deliberately NOT done in __init__: this class is instantiated at
        import time via api/dependencies.py, long before an event loop
        exists and possibly in a process that never touches the database
        (a test collecting fixtures, for example). Connecting on first
        query keeps import cheap and side-effect free.
        """
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
        )
        logger.info("HITLGate pool created")

    async def close(self) -> None:
        """
        Closes both pools this gate owns. Called on API shutdown.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("HITLGate pool closed")

        await self._procedural.close()

    # ── INBOUND: ROUTE ONE SIGNAL ─────────────────────────────

    async def process_signal(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Saves one agent signal and decides whether a human must see it.

        Args:
            signal: A SignalOutput dict from an agent, with keys
                    agent, signal_type, nct_id, summary, evidence,
                    confidence.

        Returns:
            {
              "action":     "auto_approved" | "sent_to_review" | "failed",
              "signal_id":  the new row's id (empty string on failure),
              "review_id":  the queue id, only when sent_to_review,
              "confidence": the confidence this decision was based on,
              "error":      present only when action == "failed",
            }

        Never raises. A signal that cannot be stored must not abort an
        analysis run that may hold five other valid signals, so failures
        are logged and reported in the return value instead.
        """
        confidence = self._coerce_confidence(signal.get("confidence"))
        needs_review = confidence < CONFIDENCE_THRESHOLD

        try:
            await self._ensure_pool()

            signal_id = await self._insert_signal(
                signal=signal,
                confidence=confidence,
                status=STATUS_PENDING if needs_review else STATUS_APPROVED,
            )

            if not needs_review:
                logger.info(
                    f"Signal auto-approved | "
                    f"agent={signal.get('agent')} | "
                    f"nct_id={signal.get('nct_id')} | "
                    f"confidence={confidence:.2f}"
                )
                return {
                    "action":     "auto_approved",
                    "signal_id":  signal_id,
                    "confidence": confidence,
                }

            review_id = await self._enqueue_for_review(signal_id)

            logger.info(
                f"Signal sent to human review | "
                f"agent={signal.get('agent')} | "
                f"nct_id={signal.get('nct_id')} | "
                f"confidence={confidence:.2f} | "
                f"review_id={review_id}"
            )
            return {
                "action":     "sent_to_review",
                "signal_id":  signal_id,
                "review_id":  review_id,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(
                f"process_signal failed | "
                f"agent={signal.get('agent')} | error={e}"
            )
            return {
                "action":     "failed",
                "signal_id":  "",
                "confidence": confidence,
                "error":      str(e),
            }

    async def _insert_signal(
        self,
        signal: dict[str, Any],
        confidence: float,
        status: str,
    ) -> str:
        """
        Writes one row into the signals table and returns its id.

        Args:
            signal:     The agent's signal dict.
            confidence: Already-validated confidence value.
            status:     STATUS_APPROVED or STATUS_PENDING.

        Returns:
            The new signal_id as a string.
        """
        nct_id = (signal.get("nct_id") or "").strip() or None

        if nct_id:
            nct_id = await self._ensure_study(nct_id)

        async with self._pool.acquire() as conn:
            signal_id = await conn.fetchval(
                """
                INSERT INTO signals (
                    nct_id, agent, signal_type,
                    summary, evidence, confidence, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING signal_id
                """,
                nct_id,
                signal.get("agent", "unknown"),
                signal.get("signal_type", "unknown"),
                signal.get("summary", ""),
                json.dumps(signal.get("evidence", [])),
                # evidence is a JSONB column. asyncpg wants a JSON
                # string for JSONB, not a Python list — passing the
                # list directly raises a DataError.
                confidence,
                status,
            )

        return str(signal_id)

    async def _ensure_study(self, nct_id: str) -> str | None:
        """
        Guarantees signals.nct_id can be stored for this trial.

        signals.nct_id carries a foreign key to studies(nct_id), but an
        agent can legitimately raise a signal about a trial that was
        never ingested — the clinical tools query ClinicalTrials.gov
        live, so agents see trials the database has never held.
        Inserting such a signal straight away fails the FK.

        Rather than lose the finding, we insert a minimal placeholder
        study so the reference resolves and the NCT ID stays visible in
        the API and the brief. The placeholder title makes it obvious
        the record came from a signal, not from ingestion.

        Args:
            nct_id: The trial the signal is about.

        Returns:
            The same nct_id when it is now safe to reference, or None if
            the placeholder could not be written — in which case the
            signal is still stored, just without the trial link.
        """
        if not NCT_ID_PATTERN.match(nct_id):
            logger.info(
                f"Signal carries no real trial id — storing it without a "
                f"trial link | nct_id={nct_id!r}"
            )
            return None
            # Returning None stores the signal with nct_id NULL, which
            # the foreign key accepts. Nothing is lost: the label is
            # still in the agent's summary and evidence, and the API
            # returns '' for a null nct_id.

        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM studies WHERE nct_id = $1",
                nct_id,
            )
            if exists:
                return nct_id

            try:
                await conn.execute(
                    """
                    INSERT INTO studies (nct_id, title)
                    VALUES ($1, $2)
                    ON CONFLICT (nct_id) DO NOTHING
                    """,
                    nct_id,
                    UNKNOWN_STUDY_TITLE,
                )
                logger.info(
                    f"Placeholder study row created for signal | "
                    f"nct_id={nct_id}"
                )
                return nct_id
            except Exception as e:
                logger.warning(
                    f"Could not create placeholder study — storing "
                    f"signal without trial link | nct_id={nct_id} | error={e}"
                )
                return None

    async def _enqueue_for_review(self, signal_id: str) -> str:
        """
        Adds one signal to the human review queue.

        Args:
            signal_id: The signal awaiting a decision.

        Returns:
            The review_id a human will submit their decision against.
        """
        async with self._pool.acquire() as conn:
            review_id = await conn.fetchval(
                """
                INSERT INTO hitl_reviews (signal_id, decision)
                VALUES ($1, $2)
                RETURNING review_id
                """,
                uuid.UUID(signal_id),
                DECISION_PENDING,
            )

        return str(review_id)

    # ── OUTBOUND: WHAT HUMANS MUST DECIDE ─────────────────────

    async def get_review_queue(self) -> list[dict[str, Any]]:
        """
        Returns every signal still waiting for a human decision.

        Ordered by confidence ASCENDING — the signals the agent was
        least sure about come first, because those are where a human
        judgment adds the most and where a published mistake would be
        most likely.

        Returns:
            List of dicts shaped exactly like api.schemas.ReviewQueueItem.
            Empty list when the queue is clear.
        """
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    r.review_id,
                    r.signal_id,
                    r.decision,
                    s.agent,
                    s.signal_type,
                    s.summary,
                    s.confidence,
                    COALESCE(s.nct_id, '') AS nct_id
                FROM hitl_reviews r
                JOIN signals s ON s.signal_id = r.signal_id
                WHERE r.decision = $1
                ORDER BY s.confidence ASC
                """,
                DECISION_PENDING,
            )

        queue = [
            {
                "review_id":   str(row["review_id"]),
                "signal_id":   str(row["signal_id"]),
                "agent":       row["agent"] or "unknown",
                "signal_type": row["signal_type"] or "unknown",
                "summary":     row["summary"] or "",
                "confidence":  float(row["confidence"] or 0.0),
                "nct_id":      row["nct_id"],
                "decision":    row["decision"],
            }
            for row in rows
        ]

        logger.info(f"Review queue fetched | pending={len(queue)}")
        return queue

    async def get_review_counts(self) -> dict[str, int]:
        """
        Returns how many reviews sit in each decision state.

        Lives here, next to get_review_queue(), because the gate owns the
        hitl_reviews table. api/routers/review.py previously counted these
        rows by opening its own asyncpg pool inline on every request,
        which created and tore down a database pool per API call and made
        the endpoint impossible to test without a live Postgres.

        Returns:
            {"pending": n, "approved": n, "rejected": n}
        """
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT decision, COUNT(*) AS total
                FROM hitl_reviews
                GROUP BY decision
                """
            )
            # One grouped query instead of three COUNT(*) round trips.

        counts = {row["decision"]: int(row["total"]) for row in rows}

        return {
            "pending":  counts.get(DECISION_PENDING, 0),
            "approved": counts.get("approve", 0),
            "rejected": counts.get("reject", 0),
            "edited":   counts.get("edit", 0),
        }
        # Keys are spelled out rather than returned raw so a decision
        # value that never occurred still reports 0 instead of being
        # missing from the response.

    # ── INBOUND: RECORD A HUMAN DECISION ──────────────────────

    async def process_human_decision(
        self,
        queue_id: str,
        decision: str,
        reviewer: str = "analyst",
        rejection_reason: str = "",
        edit_summary: str = "",
    ) -> dict[str, Any]:
        """
        Records an analyst's decision and runs the learning loop.

        APPROVE → signal published as-is.
        EDIT    → the analyst's corrected summary replaces the agent's,
                  then published.
        REJECT  → signal marked rejected AND, if a reason was given,
                  that reason becomes a permanent reasoning rule for the
                  agent that produced the signal.

        Args:
            queue_id:         review_id from the queue.
            decision:         "approve", "reject" or "edit".
            reviewer:         Who decided — stored for the audit trail.
            rejection_reason: Required for a reject to teach anything.
            edit_summary:     The corrected summary, for an edit.

        Returns:
            {"success": True, "signal_id": ..., "agent": ...,
             "memory_updated": bool}
            or {"success": False, "error": "..."} when the queue item
            does not exist — the router turns that into a 404.
        """
        if decision not in VALID_DECISIONS:
            return {
                "success": False,
                "error": (
                    f"Invalid decision '{decision}'. "
                    f"Expected one of {VALID_DECISIONS}."
                ),
            }

        try:
            review_uuid = uuid.UUID(queue_id)
        except (ValueError, AttributeError, TypeError):
            # A malformed id is a client mistake, not a server fault.
            # Returning success=False keeps it a 404 instead of a 500.
            return {
                "success": False,
                "error":   f"'{queue_id}' is not a valid review id.",
            }

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            review = await conn.fetchrow(
                """
                SELECT r.review_id, r.signal_id, r.decision, s.agent
                FROM hitl_reviews r
                JOIN signals s ON s.signal_id = r.signal_id
                WHERE r.review_id = $1
                """,
                review_uuid,
            )

            if review is None:
                return {
                    "success": False,
                    "error":   f"Review {queue_id} not found.",
                }

            signal_id = review["signal_id"]
            agent     = review["agent"] or "unknown"

            async with conn.transaction():
                # One transaction so the review row and the signal row
                # can never disagree about the outcome.
                await conn.execute(
                    """
                    UPDATE hitl_reviews
                    SET decision         = $1,
                        reviewer         = $2,
                        rejection_reason = $3,
                        edit_summary     = $4,
                        reviewed_at      = NOW()
                    WHERE review_id = $5
                    """,
                    decision,
                    reviewer,
                    rejection_reason,
                    edit_summary,
                    review_uuid,
                )

                await self._apply_decision_to_signal(
                    conn=conn,
                    signal_id=signal_id,
                    decision=decision,
                    edit_summary=edit_summary,
                )

        memory_updated = await self._maybe_learn(
            decision=decision,
            agent=agent,
            rejection_reason=rejection_reason,
        )

        logger.info(
            f"Human decision recorded | "
            f"review_id={queue_id} | "
            f"decision={decision} | "
            f"reviewer={reviewer} | "
            f"agent={agent} | "
            f"memory_updated={memory_updated}"
        )

        return {
            "success":        True,
            "signal_id":      str(signal_id),
            "agent":          agent,
            "decision":       decision,
            "memory_updated": memory_updated,
        }

    @staticmethod
    async def _apply_decision_to_signal(
        conn: asyncpg.Connection,
        signal_id: Any,
        decision: str,
        edit_summary: str,
    ) -> None:
        """
        Moves the signal row itself to its decided state.

        Args:
            conn:         The open transaction connection.
            signal_id:    Which signal to update.
            decision:     approve | reject | edit.
            edit_summary: Replacement summary, used only for edit.
        """
        if decision == "reject":
            await conn.execute(
                "UPDATE signals SET status = $1 WHERE signal_id = $2",
                STATUS_REJECTED,
                signal_id,
            )
            return

        if decision == "edit" and edit_summary:
            await conn.execute(
                """
                UPDATE signals
                SET status = $1, summary = $2
                WHERE signal_id = $3
                """,
                STATUS_APPROVED,
                edit_summary,
                signal_id,
            )
            return

        # approve, or an edit submitted with no replacement text
        await conn.execute(
            "UPDATE signals SET status = $1 WHERE signal_id = $2",
            STATUS_APPROVED,
            signal_id,
        )

    async def _maybe_learn(
        self,
        decision: str,
        agent: str,
        rejection_reason: str,
    ) -> bool:
        """
        Writes a rejection reason into procedural memory.

        THIS IS THE LEARNING LOOP. Everything else in this file is
        bookkeeping; this is the part that changes the system's future
        behaviour.

        Only rejections teach, and only rejections that came with a
        reason — "reject" with an empty reason records a decision but
        has nothing to teach, so no rule is written.

        A failure here is logged but never raised: the human's decision
        has already been committed, and losing the API response because
        the rule could not be stored would be worse than losing the rule.

        Args:
            decision:         The analyst's decision.
            agent:            Which agent's rules to extend.
            rejection_reason: The analyst's explanation, verbatim.

        Returns:
            True if a new rule was stored.
        """
        if decision != "reject" or not rejection_reason.strip():
            return False

        try:
            await self._procedural.update_from_feedback(
                agent_name=agent,
                rejection_reason=rejection_reason.strip(),
            )
            logger.info(
                f"LEARNING LOOP | agent={agent} | "
                f"new_rule='{rejection_reason.strip()[:80]}'"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to write learned rule | agent={agent} | error={e}"
            )
            return False

    # ── HELPERS ───────────────────────────────────────────────

    @staticmethod
    def _coerce_confidence(raw: Any) -> float:
        """
        Turns whatever the agent produced into a usable confidence score.

        The value comes from JSON that an LLM wrote, so it may be a
        string ("0.85"), missing, or out of range. A bad value must not
        be able to push a signal past the review threshold, so anything
        unparseable becomes 0.0 — the most cautious possible reading,
        which sends the signal to a human.

        Args:
            raw: The confidence field as the agent wrote it.

        Returns:
            A float clamped to 0.0 - 1.0.
        """
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Unparseable confidence {raw!r} — treating as 0.0 "
                f"so the signal goes to human review"
            )
            return 0.0

        return max(0.0, min(1.0, value))
