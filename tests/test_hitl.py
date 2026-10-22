##############################################################################
# tests/test_hitl.py
#
# WHAT THIS FILE TESTS:
#   graph/hitl.py — the routing decision and the learning loop, with the
#   database faked out (see the FakeConnection note in conftest.py).
#
# WHY THESE ARE THE HIGHEST-STAKES TESTS IN THE SUITE:
#   This class decides what gets published without a human ever seeing
#   it. Every assertion here is about one of two questions:
#
#     Did an uncertain accusation reach a human?
#     Did a human's correction actually change the agent?
#
#   Both can fail silently. A threshold comparison written the wrong way
#   round publishes exactly the signals that needed review, and the API
#   still returns 200.
##############################################################################


import json

import pytest

from graph.hitl import (
    CONFIDENCE_THRESHOLD,
    HITLGate,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
)
from tests.conftest import FakeConnection, FakePool


class FakeProceduralStore:
    """Records what the learning loop tried to write."""

    def __init__(self):
        self.learned: list[dict] = []

    async def update_from_feedback(self, agent_name, rejection_reason):
        self.learned.append(
            {"agent_name": agent_name, "rejection_reason": rejection_reason}
        )
        return "procedure-1"

    async def close(self):
        return None


@pytest.fixture
def gate(fake_connection):
    """A HITLGate wired to a fake pool and a fake procedural store."""
    gate = HITLGate()
    gate._pool = FakePool(fake_connection)
    # Set directly so _ensure_pool() returns early and never connects.
    gate._procedural = FakeProceduralStore()
    return gate


class TestSignalRouting:

    async def test_a_confident_signal_is_published_without_review(
        self, gate, fake_connection, signal
    ):
        fake_connection._fetchval = [1, "signal-uuid"]
        # The study exists, then the INSERT returns the new signal id.

        result = await gate.process_signal({**signal, "confidence": 0.9})

        assert result["action"] == "auto_approved"
        assert STATUS_APPROVED in str(fake_connection.statements)
        assert "hitl_reviews" not in fake_connection.executed_sql()
        # Nothing was queued for a human, which is the point of the
        # threshold: an analyst's attention is the scarce resource.

    async def test_an_uncertain_signal_is_queued_for_a_human(
        self, gate, fake_connection, signal
    ):
        fake_connection._fetchval = [1, "123e4567-e89b-12d3-a456-426614174000", "review-uuid"]

        result = await gate.process_signal({**signal, "confidence": 0.45})

        assert result["action"] == "sent_to_review"
        assert result["review_id"] == "review-uuid"
        assert "hitl_reviews" in fake_connection.executed_sql()
        assert STATUS_PENDING in str(fake_connection.statements)

    @pytest.mark.parametrize("confidence", [0.0, 0.3, 0.59])
    async def test_everything_below_the_threshold_goes_to_review(
        self, gate, fake_connection, signal, confidence
    ):
        fake_connection._fetchval = [1, "123e4567-e89b-12d3-a456-426614174000", "r"]

        result = await gate.process_signal({**signal, "confidence": confidence})

        assert result["action"] == "sent_to_review"

    @pytest.mark.parametrize("confidence", [0.6, 0.75, 1.0])
    async def test_the_threshold_itself_is_inclusive(
        self, gate, fake_connection, signal, confidence
    ):
        fake_connection._fetchval = [1, "sig"]

        result = await gate.process_signal({**signal, "confidence": confidence})

        assert result["action"] == "auto_approved"
        assert confidence >= CONFIDENCE_THRESHOLD
        # Exactly 0.6 publishes. Pinned because an off-by-one here is a
        # silent change in what the public sees without review.

    async def test_an_unparseable_confidence_goes_to_review(
        self, gate, fake_connection, signal
    ):
        fake_connection._fetchval = [1, "123e4567-e89b-12d3-a456-426614174000", "r"]

        result = await gate.process_signal({**signal, "confidence": "very sure"})

        assert result["action"] == "sent_to_review"
        assert result["confidence"] == 0.0
        # Fails toward human review, never toward publication.

    async def test_evidence_is_serialised_as_json_for_the_jsonb_column(
        self, gate, fake_connection, signal
    ):
        fake_connection._fetchval = [1, "sig"]

        await gate.process_signal(signal)

        insert_args = next(
            args for sql, args in fake_connection.statements
            if "INSERT INTO signals" in sql
        )
        evidence_arg = insert_args[4]

        assert isinstance(evidence_arg, str)
        assert json.loads(evidence_arg) == signal["evidence"]
        # asyncpg will not accept a Python list for a JSONB parameter —
        # it raises a DataError. The list has to be dumped to JSON first.

    async def test_a_database_failure_does_not_abort_the_run(self, gate, signal):
        class BrokenPool:
            def acquire(self):
                raise ConnectionError("cloud sql unreachable")

        gate._pool = BrokenPool()

        result = await gate.process_signal(signal)

        assert result["action"] == "failed"
        assert "cloud sql unreachable" in result["error"]
        # One unstorable signal must not take down a run holding five
        # other valid findings, so the failure is reported, not raised.

    async def test_a_non_nct_identifier_is_stored_without_a_trial_link(
        self, gate, fake_connection
    ):
        fake_connection._fetchval = ["sig"]

        await gate.process_signal({
            "agent": "pattern_finder_agent", "signal_type": "cross_study_pattern",
            "nct_id": "PATTERN", "summary": "a pattern across sponsors",
            "evidence": [], "confidence": 0.8,
        })

        insert_args = next(
            args for sql, args in fake_connection.statements
            if "INSERT INTO signals" in sql
        )

        assert insert_args[0] is None
        assert "INSERT INTO studies" not in fake_connection.executed_sql()
        # nct_id NULL, and no placeholder study invented to satisfy the
        # foreign key. "PATTERN" is a label, not a trial.


class TestHumanDecision:

    def _review_row(self):
        return {
            "review_id": "123e4567-e89b-12d3-a456-426614174000",
            "signal_id": "223e4567-e89b-12d3-a456-426614174000",
            "decision":  "pending",
            "agent":     "missing_results_agent",
        }

    async def test_a_rejection_with_a_reason_teaches_the_agent(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [self._review_row()]

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000",
            decision="reject",
            reviewer="analyst",
            rejection_reason="Terminated trials are exempt from posting.",
        )

        assert result["success"] is True
        assert result["memory_updated"] is True
        assert gate._procedural.learned[0]["agent_name"] == "missing_results_agent"
        assert gate._procedural.learned[0]["rejection_reason"].startswith("Terminated")
        assert STATUS_REJECTED in str(fake_connection.statements)
        # THE LEARNING LOOP, END TO END: the reason reaches procedural
        # memory, tagged with the agent that produced the signal — the
        # one agent whose future prompts will carry the new rule.

    async def test_a_rejection_without_a_reason_teaches_nothing(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [self._review_row()]

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000",
            decision="reject",
            rejection_reason="   ",
        )

        assert result["memory_updated"] is False
        assert gate._procedural.learned == []
        # Whitespace is not a reason. Writing it as a rule would put a
        # blank line into that agent's prompt forever.

    async def test_an_approval_publishes_the_signal(self, gate, fake_connection):
        fake_connection._fetchrow = [self._review_row()]

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000", decision="approve",
        )

        assert result["success"] is True
        assert STATUS_APPROVED in str(fake_connection.statements)
        assert gate._procedural.learned == []

    async def test_an_edit_replaces_the_summary_and_publishes(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [self._review_row()]

        await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000",
            decision="edit",
            edit_summary="Corrected: overdue by 18 months, not 5 years.",
        )

        update_args = next(
            args for sql, args in fake_connection.statements
            if "SET status = $1, summary = $2" in sql
        )

        assert update_args[1] == "Corrected: overdue by 18 months, not 5 years."

    async def test_an_edit_with_no_replacement_text_still_publishes(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [self._review_row()]

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000",
            decision="edit", edit_summary="",
        )

        assert result["success"] is True
        # An edit submitted with nothing to change is an approval. It must
        # not blank out the agent's original summary.

    @pytest.mark.parametrize("decision", ["maybe", "", "DELETE", "approve "])
    async def test_an_invalid_decision_is_refused_before_any_write(
        self, gate, fake_connection, decision
    ):
        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000", decision=decision,
        )

        assert result["success"] is False
        assert fake_connection.statements == []

    @pytest.mark.parametrize("queue_id", ["not-a-uuid", "", "12345"])
    async def test_a_malformed_review_id_is_refused_not_raised(
        self, gate, fake_connection, queue_id
    ):
        result = await gate.process_human_decision(
            queue_id=queue_id, decision="approve",
        )

        assert result["success"] is False
        assert "not a valid review id" in result["error"]
        # signals/hitl_reviews ids are UUID columns. Passing a non-UUID
        # to asyncpg raises a DataError, which the router would surface
        # as a 500 — but asking for a nonexistent thing is a 404.

    async def test_an_unknown_review_id_reports_not_found(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [None]

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000", decision="approve",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_a_failed_learning_write_does_not_lose_the_decision(
        self, gate, fake_connection
    ):
        fake_connection._fetchrow = [self._review_row()]

        class BrokenProceduralStore(FakeProceduralStore):
            async def update_from_feedback(self, agent_name, rejection_reason):
                raise ConnectionError("procedures table unreachable")

        gate._procedural = BrokenProceduralStore()

        result = await gate.process_human_decision(
            queue_id="123e4567-e89b-12d3-a456-426614174000",
            decision="reject", rejection_reason="a real reason",
        )

        assert result["success"] is True
        assert result["memory_updated"] is False
        # The human's decision was already committed. Failing the whole
        # request because the rule could not be stored would throw away
        # the decision too — and report nothing useful to the analyst.


class TestReviewQueue:

    async def test_the_queue_is_ordered_least_confident_first(
        self, gate, fake_connection
    ):
        fake_connection._fetch = [[
            {"review_id": "r1", "signal_id": "s1", "decision": "pending",
             "agent": "missing_results_agent", "signal_type": "missing_results",
             "summary": "uncertain", "confidence": 0.45, "nct_id": "NCT1"},
        ]]

        queue = await gate.get_review_queue()

        assert queue[0]["review_id"] == "r1"
        assert queue[0]["confidence"] == 0.45
        assert "ORDER BY s.confidence ASC" in fake_connection.executed_sql()
        # Ascending, so the signals the agent was least sure about are at
        # the top of the analyst's screen.

    async def test_counts_default_to_zero_for_absent_decisions(
        self, gate, fake_connection
    ):
        fake_connection._fetch = [[{"decision": "pending", "total": 3}]]

        counts = await gate.get_review_counts()

        assert counts == {"pending": 3, "approved": 0, "rejected": 0, "edited": 0}
        # A decision nobody has made yet reports 0 rather than going
        # missing from the response and breaking the dashboard.
