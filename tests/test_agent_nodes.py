##############################################################################
# tests/test_agent_nodes.py
#
# WHAT THIS FILE TESTS:
#   The six specialist agent nodes — the reasoning loop itself, driven by
#   a fake LLM and fake memory stores.
#
# WHY A FAKE LLM AND NOT THE REAL ONE:
#   A test that calls gpt-4o costs money, needs the network, and returns
#   something different every run. It also would not test our code: the
#   model's answer is the input to this loop, not its output.
#
#   What IS ours, and what is tested here:
#     - loading reasoning rules into the prompt before reasoning starts
#     - looping while the model asks for tools, stopping when it does not
#     - respecting the iteration ceiling so a loop cannot run forever
#     - parsing signals out of the final answer
#     - saving an episode whatever the outcome
#     - reporting the agent as activated even when it fails
#
# WHY ALL SIX ARE TESTED THE SAME WAY:
#   The six nodes are the same loop with different prompts and toolsets.
#   Parametrising over all of them means a change to one agent that
#   breaks the shared contract is caught, rather than only being caught
#   for whichever agent happened to have a test.
##############################################################################


import importlib

import pytest
from langchain_core.messages import AIMessage


AGENT_MODULES = [
    "agents.missing_results_agent",
    "agents.broken_promises_agent",
    "agents.track_record_agent",
    "agents.pattern_finder_agent",
    "agents.side_effect_agent",
    "agents.timeline_agent",
]

NODE_FUNCTIONS = {
    "agents.missing_results_agent": "missing_results_node",
    "agents.broken_promises_agent": "broken_promises_node",
    "agents.track_record_agent":    "track_record_node",
    "agents.pattern_finder_agent":  "pattern_finder_node",
    "agents.side_effect_agent":     "side_effect_node",
    "agents.timeline_agent":        "timeline_node",
}


# ── FAKES ─────────────────────────────────────────────────────────────

class FakeLLM:
    """
    Returns a scripted sequence of model responses.

    Each scripted item is either a string (a final answer, no tool calls)
    or a list of tool calls (the model asking to use tools). The loop
    under test is supposed to keep going on the second and stop on the
    first.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.last_messages = None

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_messages = messages

        item = self.script.pop(0) if self.script else "NO_SIGNALS_FOUND"

        if isinstance(item, str):
            return AIMessage(content=item)

        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(item)
        ])


class FakeProceduralStore:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else ["always check the status first"]
        self.asked_for = None

    async def get_procedures(self, agent_name):
        self.asked_for = agent_name
        return self.rules


class FakeEpisodicStore:
    def __init__(self):
        self.saved = []

    async def save_episode(self, agent_name, content, nct_id=None, outcome=None):
        self.saved.append(
            {"agent_name": agent_name, "content": content, "outcome": outcome}
        )
        return "episode-1"


SIGNAL_ANSWER = """I confirmed the violation.
{"nct_id": "NCT00011531", "signal_type": "missing_results",
 "summary": "Completed 2001-12, no results posted 23 years later.",
 "evidence": ["status COMPLETED", "no results section"],
 "confidence": 0.9}
"""


@pytest.fixture
def agent_env(monkeypatch):
    """
    Installs a fake LLM and fake stores into one agent module.

    Returns a function: configure(module_name, script) → (node, fakes)
    """
    def configure(module_name: str, script):
        module = importlib.import_module(module_name)

        llm        = FakeLLM(script)
        procedural = FakeProceduralStore()
        episodic   = FakeEpisodicStore()

        monkeypatch.setattr(module, "_llm", llm)
        monkeypatch.setattr(module, "_procedural", procedural)
        monkeypatch.setattr(module, "_episodic", episodic)
        # monkeypatch undoes all of this after the test, so the module's
        # real singletons are restored and no other test is affected.

        node = getattr(module, NODE_FUNCTIONS[module_name])

        return node, {"llm": llm, "procedural": procedural, "episodic": episodic}

    return configure


def base_state(**overrides) -> dict:
    state = {
        "task": "find integrity signals",
        "nct_ids": ["NCT00011531"],
        "max_studies": 3,
        "messages": [],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
        "final_brief": "",
        "run_complete": False,
        "run_id": "test-run",
    }
    return {**state, **overrides}


# ── TESTS ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", AGENT_MODULES)
class TestEveryAgentNode:
    """The contract all six nodes share."""

    async def test_reports_itself_as_activated(self, agent_env, module_name):
        node, _ = agent_env(module_name, ["NO_SIGNALS_FOUND"])

        result = await node(base_state())

        module = importlib.import_module(module_name)
        assert module.AGENT_NAME in result["agents_activated"]
        # The brief lists which agents ran, so a node that forgets to say
        # so makes the run look like it skipped a specialist.

    async def test_loads_its_reasoning_rules_before_reasoning(
        self, agent_env, module_name
    ):
        node, fakes = agent_env(module_name, ["NO_SIGNALS_FOUND"])

        await node(base_state())

        module = importlib.import_module(module_name)
        assert fakes["procedural"].asked_for == module.AGENT_NAME

        system_prompt = fakes["llm"].last_messages[0].content
        assert "always check the status first" in system_prompt
        # THE LEARNING LOOP'S READ SIDE. A rule a human taught the agent
        # only has any effect if it reaches this prompt. If this broke,
        # rejections would still be stored and would simply never be
        # applied — the system would look like it learns and would not.

    async def test_stops_as_soon_as_the_model_stops_asking_for_tools(
        self, agent_env, module_name
    ):
        node, fakes = agent_env(module_name, ["NO_SIGNALS_FOUND"])

        await node(base_state())

        assert fakes["llm"].calls == 1

    async def test_keeps_going_while_the_model_asks_for_tools(
        self, agent_env, module_name
    ):
        script = [
            [("nonexistent_tool", {})],
            [("nonexistent_tool", {})],
            SIGNAL_ANSWER,
        ]
        node, fakes = agent_env(module_name, script)

        await node(base_state())

        assert fakes["llm"].calls == 3
        # An unknown tool name is used deliberately: execute_tool returns
        # an error string for it rather than touching the database, so
        # this exercises the loop without any I/O.

    async def test_the_iteration_ceiling_is_enforced(self, agent_env, module_name):
        node, fakes = agent_env(
            module_name, [[("nonexistent_tool", {})]] * 50
        )

        await node(base_state())

        module = importlib.import_module(module_name)
        assert fakes["llm"].calls == module.MAX_ITERATIONS
        # Asserted against the agent's own declared ceiling rather than a
        # hardcoded number, because the ceiling legitimately differs:
        # pattern_finder_agent gets 12 turns for cross-study comparison
        # where the others get 10.
        #
        # Without a ceiling, a model that keeps requesting tools would
        # loop until the request timed out, burning tokens the whole way.

    async def test_saves_an_episode_even_when_it_finds_nothing(
        self, agent_env, module_name
    ):
        node, fakes = agent_env(module_name, ["NO_SIGNALS_FOUND"])

        await node(base_state())

        assert len(fakes["episodic"].saved) == 1
        assert fakes["episodic"].saved[0]["outcome"] == "no_signal"
        # "I looked and found nothing" is worth remembering: it stops a
        # later session re-investigating the same trial from scratch.

    async def test_a_store_failure_is_logged_without_killing_the_run(
        self, agent_env, module_name, monkeypatch
    ):
        node, fakes = agent_env(module_name, ["NO_SIGNALS_FOUND"])

        module = importlib.import_module(module_name)

        class BrokenProceduralStore:
            async def get_procedures(self, agent_name):
                raise ConnectionError("procedures table unreachable")

        monkeypatch.setattr(module, "_procedural", BrokenProceduralStore())

        result = await node(base_state())

        assert any("unreachable" in entry for entry in result["error_log"])
        assert module.AGENT_NAME in result["agents_activated"]
        # The node must return a state update, not raise. A raising node
        # fails the whole LangGraph run and discards the other five
        # agents' findings along with its own.


class TestSignalProduction:
    """
    Signal extraction, checked on one agent.

    Not parametrised over all six because the signal type differs per
    agent; the shared parser is covered on its own in test_agents.py.
    """

    async def test_a_signal_in_the_final_answer_reaches_the_state(self, agent_env):
        node, _ = agent_env("agents.missing_results_agent", [SIGNAL_ANSWER])

        result = await node(base_state())

        assert len(result["signals"]) == 1
        signal = result["signals"][0]
        assert signal["nct_id"] == "NCT00011531"
        assert signal["agent"] == "missing_results_agent"
        assert signal["confidence"] == 0.9

    async def test_the_episode_records_that_a_signal_was_generated(self, agent_env):
        node, fakes = agent_env("agents.missing_results_agent", [SIGNAL_ANSWER])

        await node(base_state())

        assert fakes["episodic"].saved[0]["outcome"] == "signal_generated"

    async def test_the_task_and_trial_list_reach_the_model(self, agent_env):
        node, fakes = agent_env("agents.missing_results_agent", ["NO_SIGNALS_FOUND"])

        await node(base_state(task="check NCT00011531 specifically",
                              nct_ids=["NCT00011531", "NCT00135941"]))

        human_prompt = fakes["llm"].last_messages[1].content
        assert "check NCT00011531 specifically" in human_prompt
        assert "NCT00135941" in human_prompt
        # The agent can only investigate what the request actually hands
        # it. If the router's nct_ids stopped reaching this prompt, every
        # run would quietly search broadly instead.

    async def test_findings_do_not_overwrite_another_agents_findings(self, agent_env):
        node, _ = agent_env("agents.missing_results_agent", [SIGNAL_ANSWER])

        result = await node(base_state(signals=[]))

        assert len(result["signals"]) == 1
        # Each node returns only its OWN signals, and graph/state.py's
        # _merge_lists reducer appends them. A node that returned the
        # state's existing signals too would have them counted twice,
        # once by itself and once by the reducer.
