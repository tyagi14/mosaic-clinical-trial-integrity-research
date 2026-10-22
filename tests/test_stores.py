##############################################################################
# tests/test_stores.py
#
# WHAT THIS FILE TESTS:
#   memory/episodic_store.py    episodic memory — past agent sessions
#   memory/procedural_store.py  procedural memory — reasoning rules
#   memory/semantic_store.py    semantic memory — sponsor credibility
#   processing/vector_store.py  the chunk store and semantic search
#
# HOW, WITHOUT A DATABASE:
#   Each store's pool is replaced with the FakeConnection/FakePool pair
#   from conftest.py, and the OpenAI embedding call is replaced with a
#   stub. What is under test is the shape of each query and how results
#   are turned into Python values — not whether Postgres works.
#
#   The SQL's validity is proven separately, by running the system
#   against the real Cloud SQL instance.
##############################################################################


import pytest

from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore
from processing.vector_store import VectorStore
from tests.conftest import FakeConnection, FakePool


def store_with(store, connection):
    """Attaches a fake pool so _ensure_pool() never connects."""
    store._pool = FakePool(connection)
    return store


class TestEpisodicStore:

    @pytest.fixture
    def store(self, fake_connection, monkeypatch):
        store = store_with(EpisodicStore(), fake_connection)

        async def fake_embed(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(EpisodicStore, "_embed", fake_embed)
        # Stubbed so no OpenAI call is made. The dimension is kept at
        # 1536 because that is the declared width of the VECTOR column.
        return store

    async def test_saving_an_episode_writes_every_column(self, store, fake_connection):
        episode_id = await store.save_episode(
            agent_name="missing_results_agent",
            content="investigated NCT00011531",
            nct_id="NCT00011531",
            outcome="signal_generated",
        )

        sql, args = fake_connection.statements[0]
        assert "INSERT INTO episodes" in sql
        assert args[1] == "missing_results_agent"
        assert args[2] == "NCT00011531"
        assert args[4] == "signal_generated"
        assert len(args[5]) == 1536
        assert episode_id == args[0]
        # The returned id is the one that was written, so a caller can
        # reference the episode it just created.

    async def test_searching_filters_by_similarity_and_returns_scores(
        self, store, fake_connection
    ):
        fake_connection._fetch = [[{
            "episode_id": "e1", "agent_name": "missing_results_agent",
            "nct_id": "NCT1", "content": "past finding",
            "outcome": "signal_generated", "created_at": "2026-01-01",
            "similarity": 0.48,
        }]]

        episodes = await store.search_episodes(query="missing results", top_k=3)

        assert episodes[0]["similarity"] == 0.48
        sql = fake_connection.executed_sql()
        assert "1 - (embedding <=> $1)" in sql
        # Cosine DISTANCE subtracted from 1 gives SIMILARITY, so higher
        # means more alike. Returning the raw distance would invert the
        # ranking every caller applies.

    async def test_searching_can_be_scoped_to_one_agent(self, store, fake_connection):
        fake_connection._fetch = [[]]

        await store.search_episodes(
            query="anything", agent_name="timeline_agent",
        )

        _, args = fake_connection.statements[0]
        assert "timeline_agent" in args
        # Agents consult their OWN history. Without the filter, the
        # timeline agent would be reasoning from the missing-results
        # agent's conclusions.

    async def test_recent_episodes_are_ordered_newest_first(
        self, store, fake_connection
    ):
        fake_connection._fetch = [[]]

        await store.get_recent_episodes(limit=5)

        assert "ORDER BY created_at DESC" in fake_connection.executed_sql()

    async def test_counting_episodes_returns_an_integer(self, store, fake_connection):
        fake_connection._fetchval = [42]

        assert await store.count_episodes() == 42


class TestProceduralStore:

    @pytest.fixture
    def store(self, fake_connection):
        return store_with(ProceduralStore(), fake_connection)

    async def test_seeding_defaults_is_idempotent_by_construction(
        self, store, fake_connection
    ):
        await store.initialise_defaults()

        sql = fake_connection.executed_sql()
        assert "INSERT INTO procedures" in sql
        assert "ON CONFLICT DO NOTHING" in sql
        # This runs on EVERY API startup. Without ON CONFLICT DO NOTHING
        # — and the UNIQUE (agent_name, rule_text) constraint it relies
        # on — each restart would add another copy of all 24 default
        # rules to every agent's prompt.

    async def test_seeding_covers_all_six_agents(self, store, fake_connection):
        from memory.procedural_store import DEFAULT_RULES

        await store.initialise_defaults()

        inserts = [
            args for sql, args in fake_connection.statements
            if "INSERT INTO procedures" in sql
        ]
        expected = sum(len(rules) for rules in DEFAULT_RULES.values())

        assert len(inserts) == expected
        assert {args[0] for args in inserts} == set(DEFAULT_RULES)

    async def test_rules_load_oldest_first(self, store, fake_connection):
        fake_connection._fetch = [[
            {"rule_text": "first rule"}, {"rule_text": "second rule"},
        ]]

        rules = await store.get_procedures("missing_results_agent")

        assert rules == ["first rule", "second rule"]
        assert "ORDER BY created_at ASC" in fake_connection.executed_sql()
        # Oldest first means the built-in defaults come before rules
        # learned later, so a correction reads as an amendment to them.

    async def test_a_learned_rule_is_tagged_with_its_origin(
        self, store, fake_connection
    ):
        fake_connection._fetchval = ["procedure-1"]

        await store.update_from_feedback(
            agent_name="missing_results_agent",
            rejection_reason="Terminated trials are exempt.",
        )

        _, args = fake_connection.statements[0]
        assert args[0] == "missing_results_agent"
        assert args[1] == "Terminated trials are exempt."
        assert args[2] == "learned"
        assert args[3] == "hitl_rejection"
        # The rule_type and source columns are what let the API show an
        # analyst that their correction landed, and let anyone audit why
        # an agent reasons the way it does.

    async def test_the_api_view_returns_full_metadata(self, store, fake_connection):
        fake_connection._fetch = [[{
            "procedure_id": "p1", "agent_name": "missing_results_agent",
            "rule_text": "a rule", "rule_type": "learned",
            "source": "hitl_rejection", "created_at": "2026-01-01",
        }]]

        procedures = await store.get_all_procedures_for_api("missing_results_agent")

        assert procedures[0]["rule_type"] == "learned"
        assert isinstance(procedures[0]["procedure_id"], str)
        # Stringified because procedure_id is a UUID column, and a raw
        # UUID object is not JSON serialisable.


class TestSemanticStore:

    @pytest.fixture
    def store(self, fake_connection):
        return store_with(SemanticStore(), fake_connection)

    def _profile_row(self, **overrides):
        row = {
            "sponsor": "Novo Nordisk A/S", "credibility_score": 0.45,
            "total_studies": 10, "results_posted": 6, "results_missing": 4,
            "broken_promises": 1, "avg_delay_days": 120.0,
            "last_updated": "2026-01-01",
        }
        return {**row, **overrides}

    async def test_a_known_sponsor_returns_typed_values(self, store, fake_connection):
        fake_connection._fetchrow = [self._profile_row()]

        profile = await store.get_sponsor_profile("Novo Nordisk A/S")

        assert profile["credibility_score"] == 0.45
        assert isinstance(profile["total_studies"], int)
        assert isinstance(profile["last_updated"], str)

    async def test_an_unknown_sponsor_returns_none(self, store, fake_connection):
        fake_connection._fetchrow = [None]

        assert await store.get_sponsor_profile("Nobody Ltd") is None
        # None, not an empty profile. "We have never analysed them" and
        # "they have a track record of zero" mean opposite things to an
        # agent assessing credibility.

    async def test_nulls_in_a_profile_become_zeros(self, store, fake_connection):
        fake_connection._fetchrow = [self._profile_row(
            credibility_score=None, total_studies=None, avg_delay_days=None,
        )]

        profile = await store.get_sponsor_profile("Novo Nordisk A/S")

        assert profile["credibility_score"] == 0.0
        assert profile["total_studies"] == 0
        # A newly inserted profile has NULLs before its first update, and
        # arithmetic on None raises. Agents must receive numbers.

    async def test_updating_a_sponsor_upserts_and_rescores(
        self, store, fake_connection
    ):
        await store.update_sponsor_knowledge(
            sponsor="Novo Nordisk A/S", results_posted=False,
            had_broken_promise=True, delay_days=200,
        )

        sql = fake_connection.executed_sql()
        assert "ON CONFLICT (sponsor) DO UPDATE" in sql
        assert "credibility_score" in sql
        # UPSERT rather than select-then-insert, because six agents run
        # in parallel and two of them updating the same sponsor would
        # otherwise race into a duplicate key error.

    async def test_a_missing_result_increments_the_right_counter(
        self, store, fake_connection
    ):
        await store.update_sponsor_knowledge(
            sponsor="Novo Nordisk A/S", results_posted=False,
        )

        _, args = fake_connection.statements[0]
        assert args[1] == 0
        assert args[2] == 1
        # results_posted 0, results_missing 1. These two counters are the
        # entire basis of the credibility score, so swapping them would
        # invert every sponsor's rating.

    async def test_low_credibility_sponsors_are_filtered_and_ordered(
        self, store, fake_connection
    ):
        fake_connection._fetch = [[self._profile_row(credibility_score=0.2)]]

        sponsors = await store.get_low_credibility_sponsors(
            threshold=0.6, min_studies=3,
        )

        assert sponsors[0]["credibility_score"] == 0.2
        sql = fake_connection.executed_sql()
        assert "credibility_score" in sql and "total_studies" in sql
        # min_studies matters: a sponsor with one unanalysed study would
        # otherwise show a 0.0 score and look like the worst offender in
        # the database.

    async def test_all_profiles_are_listed_worst_first(self, store, fake_connection):
        fake_connection._fetch = [[self._profile_row()]]

        await store.get_all_sponsor_profiles(limit=10)

        assert "ORDER BY credibility_score ASC" in fake_connection.executed_sql()

    async def test_sponsor_existence_is_a_boolean(self, store, fake_connection):
        fake_connection._fetchval = [1]
        assert await store.sponsor_exists("Novo Nordisk A/S") is True

        fake_connection._fetchval = [None]
        assert await store.sponsor_exists("Nobody Ltd") is False


class TestVectorStore:

    @pytest.fixture
    def store(self, fake_connection, monkeypatch):
        store = store_with(VectorStore(), fake_connection)

        async def fake_embed_query(self, text):
            return [0.1] * 1536

        monkeypatch.setattr(VectorStore, "_embed_query", fake_embed_query)
        return store

    async def test_search_by_text_embeds_then_queries(self, store, fake_connection):
        fake_connection._fetch = [[{
            "nct_id": "NCT1", "chunk_text": "text", "chunk_index": 0,
            "source": "study", "distance": 0.52, "similarity": 0.48,
        }]]

        results = await store.search(query_text="missing results", top_k=1)

        assert results[0]["similarity"] == 0.48
        assert "embedding <=> $1" in fake_connection.executed_sql()

    async def test_filters_are_applied_as_bound_parameters(
        self, store, fake_connection
    ):
        fake_connection._fetch = [[]]

        await store.search(
            query_text="x", top_k=5, source_filter="paper", nct_id_filter="NCT1",
        )

        sql, args = fake_connection.statements[0]
        assert "source = $2" in sql
        assert "nct_id = $3" in sql
        assert args[1] == "paper" and args[2] == "NCT1" and args[3] == 5
        # Placeholder numbers in the SQL, values passed separately. The
        # filter values are agent-supplied, so they must never be
        # interpolated into the query text.

    async def test_a_prepared_embedding_skips_embedding(self, store, fake_connection):
        fake_connection._fetch = [[]]

        await store.search(query_embedding=[0.2] * 1536, top_k=1)

        _, args = fake_connection.statements[0]
        assert args[0][0] == 0.2
        # The processing pipeline already holds vectors and must not pay
        # for a second embedding call to search with them.

    async def test_chunks_for_a_study_come_back_in_document_order(
        self, store, fake_connection
    ):
        fake_connection._fetch = [[
            {"nct_id": "NCT1", "chunk_text": "a", "chunk_index": 0, "source": "study"},
            {"nct_id": "NCT1", "chunk_text": "b", "chunk_index": 1, "source": "study"},
        ]]

        chunks = await store.get_chunks_for_study("NCT1")

        assert [c["chunk_index"] for c in chunks] == [0, 1]
        sql = fake_connection.executed_sql()
        assert "ORDER BY chunk_index ASC" in sql
        assert "embedding" not in sql.split("FROM")[0]
        # The embedding column is deliberately not selected: 1536 floats
        # per chunk would flood the agent's context with numbers it
        # cannot read.

    async def test_saving_chunks_skips_duplicates_rather_than_failing(
        self, store, fake_connection
    ):
        from processing.embedder import EmbeddedChunk

        chunk = EmbeddedChunk(
            chunk_id="NCT1_chunk_0", nct_id="NCT1", chunk_text="text",
            chunk_index=0, source="study", word_count=1,
            embedding=[0.1] * 1536,
        )

        saved = await store.save_embedded_chunks([chunk])

        assert saved == 1
        assert "ON CONFLICT DO NOTHING" in fake_connection.executed_sql()
        # Processing is re-runnable: a second run over the same studies
        # must not error, so duplicates are skipped.

    async def test_saving_an_empty_list_is_a_no_op(self, store, fake_connection):
        assert await store.save_embedded_chunks([]) == 0
        assert fake_connection.statements == []

    async def test_a_study_is_upserted_so_re_ingestion_refreshes_it(
        self, store, fake_connection
    ):
        await store.save_study({"nct_id": "NCT1", "title": "A trial"})

        sql = fake_connection.executed_sql()
        assert "INSERT INTO studies" in sql
        assert "ON CONFLICT (nct_id) DO UPDATE" in sql
        # Re-ingesting a trial must refresh its status and results flag.
        # DO NOTHING would freeze a study as COMPLETED-without-results
        # forever, even after the sponsor posted them.

    async def test_study_existence_check_is_a_boolean(self, store, fake_connection):
        fake_connection._fetchval = [3]
        assert await store.study_exists("NCT1") is True

        fake_connection._fetchval = [0]
        assert await store.study_exists("NCT1") is False
