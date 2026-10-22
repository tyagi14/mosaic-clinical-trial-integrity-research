##############################################################################
# tests/test_supervisor.py
#
# WHAT THIS FILE TESTS:
#   agents/supervisor.py — the run's entry node and its exit node.
#
#   supervisor_route    stamps the run and clears per-run state
#   supervisor_compile  turns raw signals into the brief a human reads
#
# WHY THE BRIEF MATTERS AS MUCH AS THE SIGNALS:
#   The brief is the product. A correct signal that is left out of the
#   brief, or described there with the wrong confidence, is as good as
#   not found — nobody reads the signals table. These tests pin what
#   reaches the page, including what happens when the LLM that writes it
#   is unavailable.
##############################################################################


import pytest
from langchain_core.messages import AIMessage

from agents import supervisor


class FakeLLM:
    def __init__(self, content="**EXECUTIVE SUMMARY:** one finding.", fail=False):
        self.content = content
        self.fail = fail
        self.prompts = None

    async def ainvoke(self, messages):
        self.prompts = messages
        if self.fail:
            raise RuntimeError("openai rate limit exceeded")
        return AIMessage(content=self.content)


def signals(*confidences) -> list[dict]:
    return [
        {
            "agent": "missing_results_agent", "signal_type": "missing_results",
            "nct_id": f"NCT0000000{i}", "summary": f"finding {i}",
            "evidence": [], "confidence": c,
        }
        for i, c in enumerate(confidences)
    ]


class TestSupervisorRoute:

    async def test_stamps_a_unique_run_id(self):
        first  = await supervisor.supervisor_route({"task": "t"})
        second = await supervisor.supervisor_route({"task": "t"})

        assert first["run_id"] != second["run_id"]

    async def test_clears_per_run_state_before_the_agents_start(self):
        result = await supervisor.supervisor_route({
            "task": "t", "signals": [{"stale": True}],
            "agents_activated": ["from_a_previous_run"],
        })

        assert result["signals"] == []
        assert result["agents_activated"] == []
        assert result["run_complete"] is False
        # Returning empty lists here is what makes the appending reducer
        # safe: the six agents all fan out from this node, so each one
        # sees a cleared list and returns only its own findings.


class TestSupervisorCompile:

    async def test_writes_a_brief_from_the_signals(self, monkeypatch):
        llm = FakeLLM()
        monkeypatch.setattr(supervisor, "llm", llm)

        result = await supervisor.supervisor_compile({
            "signals": signals(0.9, 0.8),
            "agents_activated": ["missing_results_agent"],
            "task": "find violations",
        })

        assert result["final_brief"] == "**EXECUTIVE SUMMARY:** one finding."
        assert result["run_complete"] is True
        assert result["total_signals"] == 2

    async def test_the_signals_actually_reach_the_writing_prompt(self, monkeypatch):
        llm = FakeLLM()
        monkeypatch.setattr(supervisor, "llm", llm)

        await supervisor.supervisor_compile({
            "signals": signals(0.9),
            "agents_activated": ["missing_results_agent"],
            "task": "find violations",
        })

        human_prompt = llm.prompts[1].content
        assert "NCT00000000" in human_prompt
        assert "finding 0" in human_prompt
        assert "find violations" in human_prompt
        # The brief is only as factual as what is handed to the model. If
        # the NCT IDs stopped reaching this prompt, the model would write
        # a confident brief with invented or missing identifiers.

    async def test_signals_are_split_at_the_review_threshold(self, monkeypatch):
        monkeypatch.setattr(supervisor, "llm", FakeLLM())

        result = await supervisor.supervisor_compile({
            "signals": signals(0.9, 0.7, 0.5, 0.3),
            "agents_activated": ["missing_results_agent"],
            "task": "t",
        })

        assert result["signals_requiring_review"] == 2
        # 0.5 and 0.3 are below 0.6, which is the same threshold
        # graph/hitl.py routes on. The brief and the review queue have to
        # agree on which signals a human still has to look at.

    async def test_no_signals_produces_a_clean_brief_without_calling_the_llm(
        self, monkeypatch
    ):
        llm = FakeLLM()
        monkeypatch.setattr(supervisor, "llm", llm)

        result = await supervisor.supervisor_compile({
            "signals": [], "agents_activated": ["missing_results_agent"], "task": "t",
        })

        assert "No significant research integrity signals" in result["final_brief"]
        assert result["run_complete"] is True
        assert llm.prompts is None
        # A clean run is a real and common outcome. Paying for a model
        # call to say "nothing found" would be waste, and asking a model
        # to summarise an empty list invites it to invent something.

    async def test_an_llm_failure_still_produces_a_usable_brief(self, monkeypatch):
        monkeypatch.setattr(supervisor, "llm", FakeLLM(fail=True))

        result = await supervisor.supervisor_compile({
            "signals": signals(0.9, 0.8),
            "agents_activated": ["missing_results_agent", "timeline_agent"],
            "task": "t",
        })

        assert result["run_complete"] is True
        assert "NCT00000000" in result["final_brief"]
        assert "finding 1" in result["final_brief"]
        assert "without LLM assistance" in result["final_brief"]
        # The fallback is built from the signal data directly. Six agents
        # may have just spent thirty seconds and real money finding these
        # violations; an OpenAI rate limit at the final step must not
        # throw that away. The brief says plainly that it is degraded.

    async def test_compile_does_not_resend_agents_activated(self, monkeypatch):
        monkeypatch.setattr(supervisor, "llm", FakeLLM())

        result = await supervisor.supervisor_compile({
            "signals": signals(0.9),
            "agents_activated": ["missing_results_agent", "timeline_agent"],
            "task": "t",
        })

        assert "agents_activated" not in result
        # The regression test for the duplicate-agents bug: that field
        # carries an appending reducer, so returning it here added a
        # second copy of every name and /analyze reported twelve agents
        # for a six-agent run.


class TestSignalFormatting:

    def test_every_signal_field_a_reader_needs_is_included(self):
        text = supervisor._format_signals_for_llm(signals(0.85))

        assert "NCT00000000" in text
        assert "missing_results" in text
        assert "0.85" in text
        assert "finding 0" in text

    def test_an_empty_list_is_stated_not_left_blank(self):
        assert "No signals" in supervisor._format_signals_for_llm([])

    def test_missing_fields_do_not_break_formatting(self):
        text = supervisor._format_signals_for_llm([{"summary": "bare"}])

        assert "bare" in text
        assert "unknown" in text
        # Signals come from LLM output, so a field can be absent. The
        # formatter must degrade to "unknown" rather than raising and
        # taking the whole brief down with it.
