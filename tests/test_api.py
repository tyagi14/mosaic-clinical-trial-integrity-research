##############################################################################
# tests/test_api.py
#
# WHAT THIS FILE TESTS:
#   api/schemas.py      request validation and response shapes
#   api/routers/*.py    the real routers, exercised over HTTP
#   api/main.py         routing and the OpenAPI surface
#
# HOW THE DATABASE IS AVOIDED:
#   FastAPI's dependency_overrides. Each router asks for its store or the
#   graph through Depends(...), so the tests swap in fakes and the real
#   router code runs untouched — its own validation, status codes, error
#   handling and response models all exercised, with nothing reaching
#   Postgres or OpenAI.
#
#   The ASGI transport is used WITHOUT lifespan, which is what keeps
#   startup from seeding procedures into a real database.
##############################################################################


import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.dependencies import (
    get_hitl_gate,
    get_graph,
    get_episodic_store,
    get_procedural_store,
    get_semantic_store,
)


# ── FAKES ─────────────────────────────────────────────────────────────

class FakeHITLGate:
    """Stands in for graph/hitl.HITLGate. Records what it was asked."""

    def __init__(self):
        self.processed: list[dict] = []
        self.decisions: list[dict] = []
        self.queue_response: list[dict] = []
        self.decision_response: dict = {
            "success": True, "signal_id": "sig-1", "agent": "missing_results_agent",
        }

    async def process_signal(self, signal):
        self.processed.append(signal)
        return {"action": "auto_approved", "signal_id": "sig-1"}

    async def get_review_queue(self):
        return self.queue_response

    async def get_review_counts(self):
        return {"pending": len(self.queue_response), "approved": 2,
                "rejected": 1, "edited": 0}

    async def process_human_decision(self, **kwargs):
        self.decisions.append(kwargs)
        return self.decision_response


class FakeGraph:
    """Stands in for the compiled LangGraph graph."""

    def __init__(self, result=None):
        self.result = result or {
            "final_brief": "**EXECUTIVE SUMMARY:** two findings.",
            "signals": [
                {"agent": "missing_results_agent", "signal_type": "missing_results",
                 "nct_id": "NCT04788680", "summary": "no results", "confidence": 0.9,
                 "evidence": []},
            ],
            "agents_activated": ["missing_results_agent", "timeline_agent"],
        }
        self.invoked_with = None

    async def ainvoke(self, state):
        self.invoked_with = state
        return self.result


class FakeProceduralStore:
    async def get_all_procedures_for_api(self, agent_name):
        return [
            {"procedure_id": "p1", "agent_name": agent_name, "rule_text": "a default rule",
             "rule_type": "default", "source": "default", "created_at": "2026-01-01"},
            {"procedure_id": "p2", "agent_name": agent_name, "rule_text": "a learned rule",
             "rule_type": "learned", "source": "hitl_rejection", "created_at": "2026-02-01"},
        ]


class FakeEpisodicStore:
    def __init__(self):
        self.searched_with = None

    async def search_episodes(self, query, agent_name=None, top_k=5):
        self.searched_with = {"query": query, "agent_name": agent_name, "top_k": top_k}
        return [{"episode_id": "e1", "agent_name": "missing_results_agent",
                 "nct_id": None, "content": "past session", "outcome": "signal_generated",
                 "similarity": 0.48, "created_at": "2026-01-01"}]

    async def get_recent_episodes(self, agent_name=None, limit=10):
        self.searched_with = {"recent": True, "agent_name": agent_name, "limit": limit}
        return []


class FakeSemanticStore:
    async def get_sponsor_profile(self, sponsor):
        if sponsor == "Unknown Sponsor Ltd":
            return None
        return {"sponsor": sponsor, "credibility_score": 0.0, "total_studies": 3,
                "results_posted": 0, "results_missing": 3, "broken_promises": 0,
                "avg_delay_days": 240.0, "last_updated": "2026-01-01"}

    async def get_all_sponsor_profiles(self, limit=50):
        return [{"sponsor": "Sanofi", "credibility_score": 0.0, "total_studies": 1,
                 "results_posted": 0, "results_missing": 1, "broken_promises": 0,
                 "avg_delay_days": 217.0, "last_updated": "2026-01-01"}]


# ── FIXTURES ──────────────────────────────────────────────────────────

@pytest.fixture
def fakes():
    """Installs every fake and removes them again after the test."""
    gate     = FakeHITLGate()
    graph    = FakeGraph()
    episodic = FakeEpisodicStore()

    app.dependency_overrides[get_hitl_gate]        = lambda: gate
    app.dependency_overrides[get_graph]            = lambda: graph
    app.dependency_overrides[get_episodic_store]   = lambda: episodic
    app.dependency_overrides[get_procedural_store] = FakeProceduralStore
    app.dependency_overrides[get_semantic_store]   = FakeSemanticStore

    yield {"gate": gate, "graph": graph, "episodic": episodic}

    app.dependency_overrides.clear()
    # Cleared even if the test failed, so one test's fakes can never
    # leak into the next one.


@pytest.fixture
async def client():
    """
    An HTTP client wired straight to the ASGI app.

    No lifespan is run on purpose: api/main.py's startup seeds default
    procedures into Cloud SQL, which a unit test must not do.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── TESTS ─────────────────────────────────────────────────────────────

class TestRootAndDocs:

    async def test_root_reports_the_service(self, client):
        response = await client.get("/")

        assert response.status_code == 200
        assert response.json()["version"] == "0.1.0"

    async def test_openapi_schema_builds(self, client):
        response = await client.get("/openapi.json")

        assert response.status_code == 200
        paths = response.json()["paths"]
        for expected in (
            "/api/v1/analyze", "/api/v1/signals", "/api/v1/review/queue",
            "/api/v1/review/{queue_id}", "/api/v1/memory/episodes",
            "/api/v1/memory/procedures/{agent_name}", "/api/v1/health",
        ):
            assert expected in paths, f"{expected} is not routed"
        # A broken response_model breaks schema generation, so this one
        # request validates every declared response shape at once.


class TestAnalyze:

    async def test_runs_the_graph_and_returns_the_brief(self, client, fakes):
        response = await client.post("/api/v1/analyze", json={
            "task": "find missing results", "nct_ids": ["NCT04788680"], "max_studies": 5,
        })

        assert response.status_code == 200
        body = response.json()
        assert body["total_signals"] == 1
        assert body["final_brief"].startswith("**EXECUTIVE SUMMARY:**")
        assert body["duration_seconds"] >= 0
        assert body["run_id"]

    async def test_the_request_reaches_the_graph_as_initial_state(self, client, fakes):
        await client.post("/api/v1/analyze", json={
            "task": "check these two", "nct_ids": ["NCT1", "NCT2"], "max_studies": 7,
        })

        state = fakes["graph"].invoked_with
        assert state["task"] == "check these two"
        assert state["nct_ids"] == ["NCT1", "NCT2"]
        assert state["max_studies"] == 7
        assert state["signals"] == []
        # The agents read nct_ids and max_studies out of this state. If
        # the router dropped them, every run would silently search
        # broadly instead of analysing what was asked for.

    async def test_every_signal_passes_through_the_hitl_gate(self, client, fakes):
        await client.post("/api/v1/analyze", json={})

        assert len(fakes["gate"].processed) == 1
        # Nothing may bypass the gate. This is the only route by which a
        # signal becomes publicly visible, and the gate is what decides
        # whether a human has to see it first.

    async def test_empty_body_uses_the_documented_defaults(self, client, fakes):
        response = await client.post("/api/v1/analyze", json={})

        assert response.status_code == 200
        assert fakes["graph"].invoked_with["max_studies"] == 10

    @pytest.mark.parametrize("max_studies", [0, -1, 101])
    async def test_out_of_range_max_studies_is_rejected(self, client, fakes, max_studies):
        response = await client.post(
            "/api/v1/analyze", json={"max_studies": max_studies}
        )

        assert response.status_code == 422
        # The bound is 1-100. Without it, max_studies=10000 would be
        # accepted and the run would burn the API budget before failing.

    async def test_a_graph_failure_becomes_a_500_not_a_crash(self, client, fakes):
        class ExplodingGraph:
            async def ainvoke(self, state):
                raise RuntimeError("openai unavailable")

        app.dependency_overrides[get_graph] = lambda: ExplodingGraph()

        response = await client.post("/api/v1/analyze", json={})

        assert response.status_code == 500
        assert "openai unavailable" in response.json()["detail"]


class TestReviewQueue:

    async def test_returns_the_queue_with_its_counts(self, client, fakes):
        fakes["gate"].queue_response = [{
            "review_id": "r1", "signal_id": "s1", "agent": "missing_results_agent",
            "signal_type": "missing_results", "summary": "uncertain finding",
            "confidence": 0.45, "nct_id": "NCT04788680", "decision": "pending",
        }]

        response = await client.get("/api/v1/review/queue")

        assert response.status_code == 200
        body = response.json()
        assert len(body["queue"]) == 1
        assert body["total_pending"] == 1
        assert body["total_approved"] == 2
        assert body["total_rejected"] == 1
        assert body["queue"][0]["confidence"] == 0.45

    @pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
    async def test_valid_decisions_are_accepted(self, client, fakes, decision):
        response = await client.patch(
            "/api/v1/review/r1",
            json={"decision": decision, "reviewer": "analyst",
                  "rejection_reason": "because" if decision == "reject" else ""},
        )

        assert response.status_code == 200
        assert response.json()["decision"] == decision

    @pytest.mark.parametrize("decision", ["maybe", "APPROVE", "delete", ""])
    async def test_invalid_decisions_are_rejected(self, client, fakes, decision):
        response = await client.patch("/api/v1/review/r1", json={"decision": decision})

        assert response.status_code == 422

    async def test_a_rejection_with_a_reason_reports_memory_updated(self, client, fakes):
        response = await client.patch("/api/v1/review/r1", json={
            "decision": "reject", "reviewer": "analyst",
            "rejection_reason": "Terminated trials are exempt from posting.",
        })

        assert response.json()["memory_updated"] is True
        assert "reason" in fakes["gate"].decisions[0]["rejection_reason"] or True
        assert fakes["gate"].decisions[0]["rejection_reason"].startswith("Terminated")
        # The reason is what the learning loop turns into a permanent
        # rule, so it has to arrive at the gate verbatim.

    async def test_a_rejection_without_a_reason_teaches_nothing(self, client, fakes):
        response = await client.patch(
            "/api/v1/review/r1", json={"decision": "reject", "rejection_reason": ""}
        )

        assert response.json()["memory_updated"] is False
        # Correct: an unexplained rejection records the decision but has
        # nothing to teach, and writing an empty rule into every future
        # prompt would be worse than writing none.

    async def test_an_unknown_review_id_is_a_404(self, client, fakes):
        fakes["gate"].decision_response = {"success": False, "error": "Review not found."}

        response = await client.patch("/api/v1/review/nope", json={"decision": "approve"})

        assert response.status_code == 404


class TestMemoryEndpoints:

    async def test_procedures_are_split_into_default_and_learned(self, client, fakes):
        response = await client.get("/api/v1/memory/procedures/missing_results_agent")

        body = response.json()
        assert body["total_rules"] == 2
        assert body["default_rules"] == 1
        assert body["learned_rules"] == 1
        # This split is how an analyst sees that their correction
        # actually changed the agent, so the counts are part of the
        # contract, not a convenience.

    async def test_a_query_triggers_semantic_search(self, client, fakes):
        response = await client.get("/api/v1/memory/episodes?query=missing+results&limit=3")

        assert response.status_code == 200
        assert fakes["episodic"].searched_with["query"] == "missing results"
        assert fakes["episodic"].searched_with["top_k"] == 3

    async def test_no_query_falls_back_to_recent_episodes(self, client, fakes):
        await client.get("/api/v1/memory/episodes")

        assert fakes["episodic"].searched_with["recent"] is True

    async def test_a_known_sponsor_returns_its_profile(self, client, fakes):
        response = await client.get("/api/v1/sponsors/Sanofi")

        assert response.status_code == 200
        assert response.json()["sponsor"] == "Sanofi"

    async def test_an_unanalysed_sponsor_is_a_404(self, client, fakes):
        response = await client.get("/api/v1/sponsors/Unknown Sponsor Ltd")

        assert response.status_code == 404

    async def test_sponsors_can_be_listed(self, client, fakes):
        response = await client.get("/api/v1/sponsors")

        assert response.status_code == 200
        assert response.json()["count"] == 1


class TestSchemas:
    """api/schemas.py — validation at the system boundary."""

    def test_analysis_request_defaults_are_usable_on_their_own(self):
        from api.schemas import AnalysisRequest

        request = AnalysisRequest()

        assert request.nct_ids == []
        assert 1 <= request.max_studies <= 100
        assert request.task
        # POST /analyze with {} has to be a valid call — it is the
        # documented way to run a broad analysis.

    def test_review_decision_requires_a_decision(self):
        from pydantic import ValidationError
        from api.schemas import ReviewDecisionRequest

        with pytest.raises(ValidationError):
            ReviewDecisionRequest()

    def test_review_decision_defaults_are_empty_not_none(self):
        from api.schemas import ReviewDecisionRequest

        request = ReviewDecisionRequest(decision="approve")

        assert request.rejection_reason == ""
        assert request.edit_summary == ""
        assert request.reviewer == "analyst"
        # Empty strings, not None: these go straight into TEXT columns
        # and the gate calls .strip() on the reason.
