##############################################################################
# tests/test_memory.py
#
# WHAT THIS FILE TESTS:
#   memory/procedural_store.py  the DEFAULT_RULES that shape every agent
#   memory/episodic_store.py    construction, and the similarity threshold
#   memory/semantic_store.py    construction
#   plus a cross-module consistency check between the rules and the graph
#
# WHY THE SQL ITSELF IS NOT TESTED HERE:
#   Executing it needs a live Postgres with pgvector, which is an
#   integration test, not a unit test. What IS tested is everything that
#   can be wrong without a database: the content of the default rules,
#   the lazy-construction contract every store promises, and the
#   agent-name agreement that the whole memory system depends on.
##############################################################################


from memory.procedural_store import DEFAULT_RULES, ProceduralStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore


class TestDefaultRules:
    """
    DEFAULT_RULES — the reasoning rules every agent starts life with.

    These are seeded into the procedures table on every API startup and
    loaded into each agent's system prompt on every run. They are not
    documentation: they are the agents' operating instructions.
    """

    def test_all_six_specialists_have_rules(self):
        assert len(DEFAULT_RULES) == 6

        for agent_name, rules in DEFAULT_RULES.items():
            assert rules, f"{agent_name} has no default rules"
        # An agent with no rules gets an empty "YOUR REASONING RULES"
        # section in its prompt and falls back on whatever the model
        # feels like doing — which is exactly the unaccountable
        # behaviour the procedures table exists to prevent.

    def test_rules_are_non_empty_strings(self):
        for agent_name, rules in DEFAULT_RULES.items():
            for rule in rules:
                assert isinstance(rule, str)
                assert len(rule.strip()) > 20, f"{agent_name}: rule too short"

    def test_no_duplicate_rules_within_an_agent(self):
        for agent_name, rules in DEFAULT_RULES.items():
            assert len(rules) == len(set(rules)), f"{agent_name} repeats a rule"
        # The procedures table has UNIQUE (agent_name, rule_text), so a
        # duplicate here would be silently dropped by ON CONFLICT and the
        # seeded rule count would not match this file.

    def test_the_terminated_trial_exemption_is_present(self):
        rules = " ".join(DEFAULT_RULES["missing_results_agent"]).upper()

        assert "TERMINATED" in rules
        # This specific rule is load-bearing. Terminated trials are not
        # uniformly required to post results, so without it the missing
        # results agent generates confident false accusations against
        # every sponsor who stopped a trial early.

    def test_the_covid_delay_exemption_is_present(self):
        rules = " ".join(DEFAULT_RULES["timeline_agent"]).upper()

        assert "COVID" in rules
        # Same reasoning: 2020-2022 delays have an obvious legitimate
        # cause, and flagging them all would bury the real signals.


class TestRuleAndGraphAgreement:
    """
    The one name every layer has to agree on.

    An agent's name is used as a dict key in DEFAULT_RULES, as a node
    name in the graph, as the agent_name column in procedures and
    episodes, and as the agent field on every signal. If any two of
    those disagree, nothing raises — the agent simply loads zero rules
    and its learned rules are written against a name nobody reads.
    """

    def test_default_rules_cover_exactly_the_graphs_specialists(self):
        from graph.graph_builder import SPECIALIST_NODES

        assert set(DEFAULT_RULES) == set(SPECIALIST_NODES)

    def test_each_agent_module_declares_the_same_name_it_is_keyed_by(self):
        import importlib

        for agent_name in DEFAULT_RULES:
            module = importlib.import_module(f"agents.{agent_name}")
            assert module.AGENT_NAME == agent_name
        # This is the link that closes the loop: the HITL gate reads
        # signal["agent"] (set from AGENT_NAME) to decide which agent to
        # teach when a human rejects a signal. A mismatch would write the
        # learned rule where that agent never looks.


class TestStoreConstruction:
    """
    Every store opens its database pool lazily, on first query.

    This is a real contract, not an implementation detail: these stores
    are instantiated at module import time in tools/search_tools.py and
    in each agent module. If construction connected to Postgres, then
    importing an agent — or collecting these tests — would require a
    reachable database.
    """

    def test_no_store_connects_on_construction(self):
        assert ProceduralStore()._pool is None
        assert EpisodicStore()._pool is None
        assert SemanticStore()._pool is None

    def test_episodic_store_is_configured_from_settings(self):
        from config.settings import settings

        store = EpisodicStore()

        assert store._embedding_model == settings.openai_embedding_model
        # The query embedding and the stored embeddings must come from
        # one model. Two models produce two unrelated vector spaces, and
        # similarity scores across them are meaningless.


class TestSimilarityThreshold:
    """
    The default min_similarity on episodic search.

    Documented as a test because the value is empirical, not arbitrary,
    and raising it back to an intuitive-looking 0.5 silently breaks
    episodic recall — the symptom is an empty list, with nothing in the
    logs to explain it.
    """

    def test_default_threshold_is_calibrated_to_the_embedding_model(self):
        import inspect

        signature = inspect.signature(EpisodicStore.search_episodes)
        default = signature.parameters["min_similarity"].default

        assert default <= 0.35, (
            "Measured similarity for clearly relevant episodes from "
            "text-embedding-3-small is 0.39-0.65, and for some phrasings "
            "as low as 0.40. A threshold above ~0.35 discards real "
            "matches and agents lose their episodic memory silently."
        )
