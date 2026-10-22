##############################################################################
# tests/test_processing.py
#
# WHAT THIS FILE TESTS:
#   processing/chunker.py       ParsedStudy → overlapping TextChunks
#   processing/embedder.py      batching and the single-text query path
#   processing/vector_store.py  argument validation on search()
#
# NO OPENAI CALLS AND NO DATABASE:
#   Embedding and querying are exercised against fakes. What matters in
#   this layer is the splitting, batching and argument handling — all of
#   which is our own logic and all of which is deterministic.
##############################################################################


import pytest

from processing.chunker import Chunker, TextChunk, CHUNK_SIZE, OVERLAP_SIZE
from processing.embedder import Embedder, EmbeddedChunk, BATCH_SIZE
from processing.vector_store import VectorStore


class TestChunkStudy:
    """Chunker.chunk_study — one study to a list of embeddable chunks."""

    def test_produces_at_least_one_chunk(self, parsed_study):
        chunks = Chunker().chunk_study(parsed_study)

        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_every_chunk_carries_its_studys_identity(self, parsed_study):
        chunks = Chunker().chunk_study(parsed_study)

        for chunk in chunks:
            assert chunk.nct_id == parsed_study.nct_id
            assert chunk.source == "study"
            assert chunk.chunk_text.strip()
        # Identity travels with every chunk because search returns chunks,
        # not studies. A chunk that lost its nct_id is an unattributable
        # quote — useless as evidence in a signal.

    def test_chunk_indexes_are_sequential_from_zero(self, parsed_study):
        chunks = Chunker().chunk_study(parsed_study)

        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # get_chunks_for_study() orders by chunk_index to reassemble the
        # document, so a gap or a repeat would reorder the text.

    def test_chunk_ids_are_unique(self, parsed_study):
        chunks = Chunker().chunk_study(parsed_study)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_the_text_includes_the_fields_agents_reason_about(self, parsed_study):
        combined = " ".join(c.chunk_text for c in Chunker().chunk_study(parsed_study))

        assert parsed_study.nct_id in combined
        assert parsed_study.sponsor in combined
        assert parsed_study.status in combined
        # An agent searching "completed trial, sponsor never posted
        # results" can only match these words if they are in the embedded
        # text. Leave the status out of the chunk and semantic search for
        # compliance questions stops working.

    def test_word_count_matches_the_text(self, parsed_study):
        for chunk in Chunker().chunk_study(parsed_study):
            assert chunk.word_count == len(chunk.chunk_text.split())


class TestSplitIntoChunks:
    """
    Chunker._split_into_chunks — the sliding window itself.

    Tested directly with generated text because a real study is too
    short to produce multiple chunks, and the overlap behaviour only
    shows up across a chunk boundary.
    """

    def _split(self, text: str) -> list[TextChunk]:
        return Chunker()._split_into_chunks(text, nct_id="NCT1", source="study")

    def test_short_text_stays_as_one_chunk(self):
        assert len(self._split("one two three four five")) == 1

    def test_long_text_is_split_into_multiple_chunks(self):
        text = " ".join(f"word{i}" for i in range(CHUNK_SIZE * 3))

        chunks = self._split(text)

        assert len(chunks) > 1
        assert all(len(c.chunk_text.split()) <= CHUNK_SIZE for c in chunks)

    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"word{i}" for i in range(CHUNK_SIZE * 2))

        chunks = self._split(text)
        first_words  = chunks[0].chunk_text.split()
        second_words = chunks[1].chunk_text.split()

        assert set(first_words[-OVERLAP_SIZE:]) & set(second_words)
        # WHY OVERLAP MATTERS:
        # Chunks are split on a fixed word count, which lands mid-sentence
        # and can cut a fact in half — "...results were never" | "posted
        # for this trial...". Neither chunk then matches a search for the
        # whole fact. The overlap repeats the boundary words so the
        # complete statement survives in at least one chunk.

    def test_indexes_and_ids_stay_consistent_across_many_chunks(self):
        chunks = self._split(" ".join(f"word{i}" for i in range(CHUNK_SIZE * 3)))

        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.chunk_id == f"NCT1_chunk_{c.chunk_index}" for c in chunks)
        assert all(c.source == "study" for c in chunks)

    def test_empty_text_produces_no_chunks(self):
        assert self._split("") == []
        assert self._split("    ") == []


class TestEmbedderBatching:
    """
    Embedder._create_batches — how chunks are grouped into API calls.

    One OpenAI call per chunk would be hundreds of round trips for one
    study set. Batching is what makes ingestion affordable, and an
    off-by-one here either exceeds the request limit or silently drops
    the tail of the list.
    """

    def _chunks(self, count: int) -> list[TextChunk]:
        return [
            TextChunk(
                chunk_id=f"NCT1_chunk_{i}", nct_id="NCT1",
                chunk_text=f"text {i}", chunk_index=i,
                source="study", word_count=2,
            )
            for i in range(count)
        ]

    def test_fewer_chunks_than_the_batch_size_make_one_batch(self):
        batches = Embedder()._create_batches(self._chunks(10))

        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_batches_respect_the_batch_size_limit(self):
        batches = Embedder()._create_batches(self._chunks(BATCH_SIZE * 2 + 5))

        assert len(batches) == 3
        assert all(len(b) <= BATCH_SIZE for b in batches)

    def test_no_chunk_is_lost_or_duplicated_by_batching(self):
        chunks = self._chunks(BATCH_SIZE * 2 + 7)

        flattened = [c for batch in Embedder()._create_batches(chunks) for c in batch]

        assert [c.chunk_id for c in flattened] == [c.chunk_id for c in chunks]

    def test_empty_input_produces_no_batches(self):
        assert Embedder()._create_batches([]) == []


class TestEmbedText:
    """Embedder.embed_text — the query-time single-text path."""

    async def test_empty_text_is_rejected_before_the_api_call(self):
        with pytest.raises(ValueError):
            await Embedder().embed_text("")

        with pytest.raises(ValueError):
            await Embedder().embed_text("   ")
        # OpenAI rejects empty input with an opaque 400. Failing here
        # names the actual problem, and costs nothing.


class TestVectorStoreSearchArguments:
    """
    VectorStore.search — the query_text / query_embedding contract.

    search() accepts either a ready-made vector or plain text that it
    embeds itself. Both paths are used in production, so the validation
    between them is worth pinning down.
    """

    async def test_search_without_a_query_raises_before_touching_the_database(self):
        with pytest.raises(ValueError, match="query_embedding or query_text"):
            await VectorStore().search()
        # Deliberately ordered so this check runs BEFORE _ensure_pool().
        # A caller that forgot the query gets a precise message instead
        # of a confusing connection attempt, and no pool is opened.

    def test_a_fresh_store_holds_no_pool_or_embedder(self):
        store = VectorStore()

        assert store._pool is None
        assert store._embedder is None
        # Construction must stay free of side effects: tools/search_tools
        # builds one of these at import time, and importing a module must
        # not open a database connection or require an OpenAI key.


class TestEmbeddedChunk:
    """The EmbeddedChunk record written to the database."""

    def test_carries_the_chunk_fields_plus_the_vector(self):
        chunk = EmbeddedChunk(
            chunk_id="NCT1_chunk_0", nct_id="NCT1", chunk_text="text",
            chunk_index=0, source="study", word_count=1,
            embedding=[0.1] * 1536,
        )

        assert chunk.nct_id == "NCT1"
        assert len(chunk.embedding) == 1536
        # 1536 is the dimension of text-embedding-3-small and the
        # declared width of the VECTOR(1536) columns. A vector of any
        # other length is rejected by Postgres on insert.


##############################################################################
# PAPER CHUNKING AND BATCH EMBEDDING
#
# Added alongside the study path above: papers go through the same
# chunker and embedder, and the embedder's retry logic is the part of
# ingestion most likely to be exercised in anger — OpenAI rate limits
# are routine on a large run.
##############################################################################


class TestChunkPaper:
    """Chunker.chunk_paper — the PubMed counterpart to chunk_study."""

    def _paper(self):
        from ingestion.document_parser import ParsedPaper

        return ParsedPaper(
            pmid="31234567",
            title="Outcomes of a semaglutide trial",
            abstract="Three patients were hospitalised with serious events. " * 20,
            journal="NEJM",
            pub_date="2019-06",
            authors=["Smith J", "Doe J"],
            nct_ids_referenced=["NCT04788680"],
            word_count=140,
            parsed_at="2026-01-01T00:00:00",
        )

    def test_chunks_are_tagged_as_papers(self):
        chunks = Chunker().chunk_paper(self._paper())

        assert chunks
        assert all(c.source == "paper" for c in chunks)
        # The source tag is what makes source_filter="paper" work in
        # vector search. A paper chunk labelled "study" would be returned
        # as if it were an official filing — exactly the confusion the
        # side effect agent exists to resolve.

    def test_the_paper_is_linked_to_the_trial_it_reports_on(self):
        chunks = Chunker().chunk_paper(self._paper())

        assert all(c.nct_id == "NCT04788680" for c in chunks)
        # Keyed by the referenced TRIAL, not by the paper's own PMID.
        # chunks.nct_id carries a foreign key to studies(nct_id), so a
        # PMID there is rejected on insert and the paper's chunks are
        # dropped. It is also the wrong join: comparing what a sponsor
        # filed against what was published needs both sides keyed by
        # the same trial.

    def test_paper_chunk_ids_stay_traceable_to_the_paper(self):
        chunks = Chunker().chunk_paper(self._paper())

        assert all(c.chunk_id.startswith("PMID31234567_chunk_") for c in chunks)
        # Keyed by trial, identified by paper. Without the PMID prefix a
        # paper's chunk ids would collide with the trial filing's own.

    def test_a_paper_referencing_no_trial_falls_back_to_its_pmid(self):
        paper = self._paper().model_copy(update={"nct_ids_referenced": []})
        # model_copy because ParsedPaper is a pydantic model, not a
        # dataclass — and a copy rather than an edit, so the fixture
        # cannot leak a mutation into another test.

        chunks = Chunker().chunk_paper(paper)

        assert all(c.nct_id == "31234567" for c in chunks)

    def test_the_safety_language_survives_chunking(self):
        combined = " ".join(c.chunk_text for c in Chunker().chunk_paper(self._paper()))

        assert "hospitalised" in combined
        assert "NEJM" in combined

    def test_a_batch_of_papers_chunks_to_a_flat_list(self):
        chunks = Chunker().chunk_papers([self._paper(), self._paper()])

        assert all(isinstance(c, TextChunk) for c in chunks)
        assert len(chunks) >= 2


class TestEmbedderRetry:
    """
    Embedder.embed_chunks — the batch path, with the OpenAI call faked.

    Retry behaviour matters here: a rate limit halfway through a large
    ingestion run would otherwise lose every chunk after it.
    """

    def _chunks(self, count=3):
        return [
            TextChunk(
                chunk_id=f"NCT1_chunk_{i}", nct_id="NCT1",
                chunk_text=f"text {i}", chunk_index=i,
                source="study", word_count=2,
            )
            for i in range(count)
        ]

    class _FakeEmbeddings:
        def __init__(self, fail_times=0):
            self.fail_times = fail_times
            self.calls = 0

        async def create(self, model, input):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("rate limit exceeded")

            class Item:
                def __init__(self, embedding):
                    self.embedding = embedding

            class Response:
                def __init__(self, count):
                    self.data = [Item([0.1] * 1536) for _ in range(count)]

            count = len(input) if isinstance(input, list) else 1
            return Response(count)

    def _install(self, embedder, fake):
        class FakeClient:
            embeddings = fake

        embedder._client = FakeClient()
        return embedder

    async def test_every_chunk_comes_back_embedded(self):
        embedder = self._install(Embedder(), self._FakeEmbeddings())

        embedded = await embedder.embed_chunks(self._chunks(3))

        assert len(embedded) == 3
        assert all(len(e.embedding) == 1536 for e in embedded)
        assert [e.chunk_id for e in embedded] == [c.chunk_id for c in self._chunks(3)]
        # Identity and order are preserved: the embedding has to end up
        # attached to the chunk it was computed from, or every search
        # result afterwards points at the wrong text.

    async def test_a_transient_failure_is_retried(self):
        fake = self._FakeEmbeddings(fail_times=1)
        embedder = self._install(Embedder(), fake)

        embedded = await embedder.embed_chunks(self._chunks(2))

        assert len(embedded) == 2
        assert fake.calls == 2
        # One failure, one retry, then success. OpenAI rate limits are
        # routine on a large ingestion run and must not be fatal.

    async def test_embedding_nothing_returns_nothing(self):
        embedder = self._install(Embedder(), self._FakeEmbeddings())

        assert await embedder.embed_chunks([]) == []
