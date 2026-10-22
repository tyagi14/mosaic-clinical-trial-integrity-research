##############################################################################
# tests/test_tools.py
#
# WHAT THIS FILE TESTS:
#   tools/async_bridge.py   the sync→async bridge every tool depends on
#   tools/search_tools.py   the tool contract: always a JSON string
#
# WHY THE BRIDGE GETS ITS OWN TESTS:
#   It is the single point of failure for all fourteen tools. The version
#   this replaced used asyncio.get_event_loop().run_until_complete(),
#   which raises "This event loop is already running" when called from
#   inside a running loop — which is exactly how agents call tools. Every
#   tool call failed, and because tools swallow their own errors into a
#   returned string, the agents just reported finding nothing. Nothing in
#   the logs said why.
#
#   So the critical test here is the one that calls the bridge from inside
#   a running event loop.
#
# THE CONTRACT BEING PINNED FOR TOOLS:
#   A tool ALWAYS returns a JSON string and NEVER raises — not when the
#   database is down, not when a store returns nothing. A raising tool
#   kills the agent node and discards every finding gathered so far.
##############################################################################


import asyncio
import json

import pytest

from tools.async_bridge import run_async


class TestAsyncBridge:

    def test_runs_a_coroutine_from_synchronous_code(self):
        async def work():
            await asyncio.sleep(0)
            return "done"

        assert run_async(work()) == "done"

    async def test_runs_a_coroutine_from_inside_a_running_loop(self):
        async def work():
            await asyncio.sleep(0)
            return "done"

        assert run_async(work()) == "done"
        # THE REGRESSION TEST THAT MATTERS.
        #
        # This test function is itself async, so an event loop is already
        # running on this thread — the same situation as an agent node
        # calling a tool. The old implementation raised RuntimeError here.

    async def test_many_calls_share_one_background_loop(self):
        async def work(n):
            await asyncio.sleep(0)
            return n * 2

        results = [run_async(work(i)) for i in range(5)]

        assert results == [0, 2, 4, 6, 8]
        # Reusing one loop is not just efficiency: asyncpg pools are bound
        # to the loop that created them. A fresh loop per call would make
        # the second call fail with "got Future attached to a different
        # loop" when it reused a store's cached pool.

    async def test_exceptions_propagate_to_the_caller(self):
        async def failing():
            raise ValueError("the real problem")

        with pytest.raises(ValueError, match="the real problem"):
            run_async(failing())
        # The bridge must not swallow errors. Each tool decides how to
        # report its own failure; a bridge that hid exceptions would make
        # every failure look like an empty result.

    async def test_a_hung_coroutine_times_out_instead_of_blocking_forever(self):
        async def hangs():
            await asyncio.sleep(30)

        with pytest.raises(TimeoutError):
            run_async(hangs(), timeout=0.2)
        # Without this ceiling, one unresponsive API call would hang an
        # agent node, and with it the whole analysis request.


class TestSearchToolContract:
    """
    The @tool functions in tools/search_tools.py.

    The module holds its stores as private module-level singletons, so
    each test swaps in a fake. That exercises the real tool body —
    argument handling, JSON serialisation, error handling — without a
    database.
    """

    @pytest.fixture
    def tools_module(self, monkeypatch):
        import tools.search_tools as module
        return module, monkeypatch

    def test_search_returns_results_as_a_json_string(self, tools_module):
        module, monkeypatch = tools_module

        class FakeVectorStore:
            async def search(self, query_text, top_k=5, source_filter=None, nct_id_filter=None):
                return [{"nct_id": "NCT1", "chunk_text": "text", "similarity": 0.48}]

        monkeypatch.setattr(module, "_vector_store", FakeVectorStore())

        output = module.search_studies_by_meaning.invoke(
            {"query": "missing results", "top_k": 1}
        )

        parsed = json.loads(output)
        assert parsed["count"] == 1
        assert parsed["results"][0]["nct_id"] == "NCT1"
        # json.loads on the output is the assertion that matters: the
        # agent receives this as text and has to be able to read it.

    def test_empty_results_return_an_explanation_not_an_error(self, tools_module):
        module, monkeypatch = tools_module

        class EmptyVectorStore:
            async def search(self, **kwargs):
                return []

        monkeypatch.setattr(module, "_vector_store", EmptyVectorStore())

        parsed = json.loads(
            module.search_studies_by_meaning.invoke({"query": "nothing matches"})
        )

        assert parsed["results"] == []
        assert "message" in parsed
        # "Found nothing" and "something broke" must look different to the
        # agent. Conflating them makes it retry a working search, or give
        # up on a broken one.

    def test_a_failing_store_becomes_an_error_payload_not_an_exception(
        self, tools_module
    ):
        module, monkeypatch = tools_module

        class BrokenVectorStore:
            async def search(self, **kwargs):
                raise ConnectionError("cloud sql unreachable")

        monkeypatch.setattr(module, "_vector_store", BrokenVectorStore())

        output = module.search_studies_by_meaning.invoke({"query": "anything"})

        parsed = json.loads(output)
        assert "cloud sql unreachable" in parsed["error"]
        assert parsed["results"] == []
        # The tool catches, reports and returns. If it raised, the agent
        # node would die and lose every signal it had already found.

    def test_sponsor_profile_tool_reports_an_unknown_sponsor(self, tools_module):
        module, monkeypatch = tools_module

        class FakeSemanticStore:
            async def get_sponsor_profile(self, sponsor):
                return None

        monkeypatch.setattr(module, "_semantic_store", FakeSemanticStore())

        parsed = json.loads(
            module.get_sponsor_profile.invoke({"sponsor_name": "Nobody Ltd"})
        )

        assert parsed["found"] is False
        # An unanalysed sponsor is the normal case on a first run, not an
        # error. The agent needs to tell "no track record yet" apart from
        # "a track record of zero".

    def test_save_episode_tool_returns_the_new_episode_id(self, tools_module):
        module, monkeypatch = tools_module

        class FakeEpisodicStore:
            async def save_episode(self, agent_name, content, nct_id=None, outcome=None):
                return "episode-123"

        monkeypatch.setattr(module, "_episodic_store", FakeEpisodicStore())

        parsed = json.loads(module.save_episode.invoke({
            "agent_name": "missing_results_agent",
            "content": "investigated NCT1",
            "outcome": "signal_generated",
        }))

        assert parsed["success"] is True
        assert parsed["episode_id"] == "episode-123"

    def test_every_search_tool_is_registered_as_a_langchain_tool(self):
        import tools.search_tools as module

        for name in (
            "search_studies_by_meaning", "search_past_episodes", "save_episode",
            "get_sponsor_profile", "update_sponsor_profile",
            "get_low_credibility_sponsors", "search_study_chunks_by_nct_id",
            "search_papers_by_meaning",
        ):
            tool = getattr(module, name)
            assert tool.name == name
            assert tool.description, f"{name} has no description"
        # The description IS the model's instruction manual — it is the
        # only thing telling GPT-4o when to reach for this tool. A tool
        # with no description is a tool that never gets called.


class TestRemainingSearchTools:
    """
    The search tools not covered above. Same contract throughout: a JSON
    string out, never an exception, and "found nothing" distinguishable
    from "something broke".
    """

    @pytest.fixture
    def module(self):
        import tools.search_tools as module
        return module

    def test_past_episodes_are_returned_with_similarity_scores(
        self, module, monkeypatch
    ):
        class FakeEpisodicStore:
            async def search_episodes(self, query, agent_name=None, top_k=5):
                return [{
                    "episode_id": "e1", "agent_name": "missing_results_agent",
                    "nct_id": "NCT1", "content": "past investigation",
                    "outcome": "signal_generated", "similarity": 0.52,
                    "created_at": "2026-01-01",
                }]

        monkeypatch.setattr(module, "_episodic_store", FakeEpisodicStore())

        parsed = json.loads(module.search_past_episodes.invoke({
            "query": "trials with no results", "agent_name": "missing_results_agent",
        }))

        assert parsed["count"] == 1
        assert parsed["episodes"][0]["similarity"] == 0.52

    def test_no_past_episodes_is_reported_as_a_first_investigation(
        self, module, monkeypatch
    ):
        class EmptyEpisodicStore:
            async def search_episodes(self, **kwargs):
                return []

        monkeypatch.setattr(module, "_episodic_store", EmptyEpisodicStore())

        parsed = json.loads(module.search_past_episodes.invoke(
            {"query": "anything", "agent_name": "missing_results_agent"}
        ))

        assert parsed["episodes"] == []
        assert "message" in parsed
        # On a first run there is no history, and that is normal. The
        # agent must not read it as a broken memory system.

    def test_updating_a_sponsor_profile_confirms_what_it_recorded(
        self, module, monkeypatch
    ):
        recorded = {}

        class FakeSemanticStore:
            async def update_sponsor_knowledge(
                self, sponsor, results_posted=False,
                had_broken_promise=False, delay_days=0,
            ):
                recorded.update({
                    "sponsor": sponsor, "results_posted": results_posted,
                    "had_broken_promise": had_broken_promise,
                    "delay_days": delay_days,
                })

        monkeypatch.setattr(module, "_semantic_store", FakeSemanticStore())

        parsed = json.loads(module.update_sponsor_profile.invoke({
            "sponsor_name": "Novo Nordisk A/S", "results_posted": False,
            "had_broken_promise": True, "delay_days": 200,
        }))

        assert parsed["success"] is True
        assert recorded["sponsor"] == "Novo Nordisk A/S"
        assert recorded["results_posted"] is False
        assert recorded["had_broken_promise"] is True
        assert recorded["delay_days"] == 200
        # Every argument has to arrive intact: these counters are the
        # whole basis of a sponsor's credibility score, and the score
        # feeds back into future agents' confidence.

    def test_low_credibility_sponsors_are_listed(self, module, monkeypatch):
        class FakeSemanticStore:
            async def get_low_credibility_sponsors(self, threshold=0.6, min_studies=3):
                return [{"sponsor": "A Sponsor", "credibility_score": 0.2,
                         "total_studies": 5}]

        monkeypatch.setattr(module, "_semantic_store", FakeSemanticStore())

        parsed = json.loads(
            module.get_low_credibility_sponsors.invoke({"threshold": 0.6})
        )

        assert parsed["count"] == 1
        assert parsed["sponsors"][0]["credibility_score"] == 0.2

    def test_no_low_credibility_sponsors_is_a_clean_result(
        self, module, monkeypatch
    ):
        class FakeSemanticStore:
            async def get_low_credibility_sponsors(self, **kwargs):
                return []

        monkeypatch.setattr(module, "_semantic_store", FakeSemanticStore())

        parsed = json.loads(module.get_low_credibility_sponsors.invoke({}))

        assert parsed.get("count", 0) == 0
        assert "error" not in parsed

    def test_all_chunks_of_a_study_are_returned_when_no_query_is_given(
        self, module, monkeypatch
    ):
        class FakeVectorStore:
            async def get_chunks_for_study(self, nct_id):
                return [
                    {"nct_id": nct_id, "chunk_text": "part one", "chunk_index": 0},
                    {"nct_id": nct_id, "chunk_text": "part two", "chunk_index": 1},
                ]

            async def search(self, **kwargs):
                raise AssertionError("should not search when no query was given")

        monkeypatch.setattr(module, "_vector_store", FakeVectorStore())

        parsed = json.loads(
            module.search_study_chunks_by_nct_id.invoke({"nct_id": "NCT1"})
        )

        assert parsed["count"] == 2
        assert parsed["chunks"][0]["chunk_index"] == 0
        # No query means "show me the whole record", which is a different
        # question from "which part of it is about X" — and the reason
        # VectorStore needed a get_chunks_for_study method at all.

    def test_a_query_narrows_the_search_to_that_one_study(
        self, module, monkeypatch
    ):
        captured = {}

        class FakeVectorStore:
            async def search(self, query_text, top_k=5, nct_id_filter=None, **kwargs):
                captured.update({"query_text": query_text, "nct_id_filter": nct_id_filter})
                return [{"nct_id": "NCT1", "chunk_text": "relevant part",
                         "similarity": 0.5}]

        monkeypatch.setattr(module, "_vector_store", FakeVectorStore())

        parsed = json.loads(module.search_study_chunks_by_nct_id.invoke(
            {"nct_id": "NCT1", "query": "adverse events"}
        ))

        assert parsed["count"] == 1
        assert captured["nct_id_filter"] == "NCT1"
        # Scoped to the one trial. Without the filter the agent would get
        # the most similar chunks from the whole database and could
        # attribute another trial's text to this one.

    def test_a_study_with_no_chunks_says_it_was_not_processed(
        self, module, monkeypatch
    ):
        class EmptyVectorStore:
            async def get_chunks_for_study(self, nct_id):
                return []

        monkeypatch.setattr(module, "_vector_store", EmptyVectorStore())

        parsed = json.loads(
            module.search_study_chunks_by_nct_id.invoke({"nct_id": "NCT_MISSING"})
        )

        assert parsed["chunks"] == []
        assert "not have been processed" in parsed["message"]
        # Tells the agent the right next step: this trial is absent from
        # the local snapshot, so use the live ClinicalTrials.gov tools.

    def test_paper_search_is_restricted_to_paper_chunks(
        self, module, monkeypatch
    ):
        captured = {}

        class FakeVectorStore:
            async def search(self, query_text, top_k=5, source_filter=None, **kwargs):
                captured["source_filter"] = source_filter
                return [{"nct_id": "NCT1", "chunk_text": "a published finding",
                         "similarity": 0.44}]

        monkeypatch.setattr(module, "_vector_store", FakeVectorStore())

        parsed = json.loads(
            module.search_papers_by_meaning.invoke({"query": "adverse events"})
        )

        assert parsed["count"] == 1
        assert captured["source_filter"] == "paper"
        # The source filter is the only thing separating a peer-reviewed
        # paper from the sponsor's own filing. Conflating them would let
        # an agent cite a sponsor's own words as independent evidence
        # against it.

    def test_every_tool_reports_store_failures_as_data(self, module, monkeypatch):
        class BrokenStore:
            def __getattr__(self, name):
                async def fail(*args, **kwargs):
                    raise ConnectionError("cloud sql unreachable")
                return fail

        monkeypatch.setattr(module, "_episodic_store", BrokenStore())
        monkeypatch.setattr(module, "_semantic_store", BrokenStore())
        monkeypatch.setattr(module, "_vector_store", BrokenStore())

        calls = [
            (module.search_past_episodes, {"query": "x", "agent_name": "a"}),
            (module.save_episode, {"agent_name": "a", "content": "c"}),
            (module.get_sponsor_profile, {"sponsor_name": "s"}),
            (module.update_sponsor_profile, {"sponsor_name": "s"}),
            (module.get_low_credibility_sponsors, {}),
            (module.search_study_chunks_by_nct_id, {"nct_id": "NCT1"}),
            (module.search_papers_by_meaning, {"query": "x"}),
        ]

        for tool, args in calls:
            parsed = json.loads(tool.invoke(args))
            assert "error" in parsed or parsed.get("success") is False, tool.name
        # Checked across every tool at once, because the contract only
        # holds if it holds everywhere: one tool that raises instead of
        # returning kills the agent node that called it.
