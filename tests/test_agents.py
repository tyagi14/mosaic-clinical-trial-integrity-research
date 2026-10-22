##############################################################################
# tests/test_agents.py
#
# WHAT THIS FILE TESTS:
#   agents/signal_parser.py   model output  → structured signals
#   agents/tool_executor.py   tool dispatch and its failure modes
#   graph/state.py            the reducer that lets agents run in parallel
#   graph/graph_builder.py    the shape of the graph
#   graph/hitl.py             the routing threshold and its input handling
#
# WHY THIS IS THE MOST IMPORTANT TEST FILE:
#   This is the layer where a bug is invisible. If the parser drops a
#   signal, or the reducer overwrites one agent's findings with another's,
#   or the gate waves a low-confidence accusation through, nothing errors.
#   The API returns 200 and a confident brief that is simply missing
#   findings — or publishing ones no human approved.
##############################################################################


import pytest

from agents.signal_parser import parse_signals
from agents.tool_executor import execute_tool, MAX_RESULT_CHARS
from graph.state import _merge_lists
from graph.hitl import HITLGate, CONFIDENCE_THRESHOLD, NCT_ID_PATTERN


class TestSignalParsing:
    """
    parse_signals — recovering structured findings from a model's prose.

    Agents answer in natural language with JSON blocks embedded in it, so
    this function is the only thing standing between "the agent found a
    violation" and "the API reports zero signals".
    """

    def test_extracts_a_plain_signal_block(self):
        text = """I checked the filing.
        {"nct_id": "NCT04788680", "signal_type": "missing_results",
         "summary": "No results 5 years after completion.",
         "evidence": ["completed 2019-06"], "confidence": 0.9}
        """

        signals = parse_signals(text, "missing_results_agent", "missing_results")

        assert len(signals) == 1
        assert signals[0]["nct_id"] == "NCT04788680"
        assert signals[0]["agent"] == "missing_results_agent"
        assert signals[0]["confidence"] == 0.9

    def test_extracts_a_signal_with_a_nested_object(self):
        text = """Findings below.
        ```json
        {"nct_id": "NCT00011531", "signal_type": "missing_results",
         "summary": "Overdue by 23 years.",
         "evidence": {"completion": "2001-12", "months_overdue": 286},
         "confidence": 0.95}
        ```
        """

        signals = parse_signals(text, "missing_results_agent", "missing_results")

        assert len(signals) == 1
        assert signals[0]["nct_id"] == "NCT00011531"
        assert len(signals[0]["evidence"]) == 2
        # THIS IS THE REGRESSION TEST FOR THE ORIGINAL PARSER BUG.
        #
        # The first implementation matched with r'\{[^{}]*"signal_type"[^{}]*\}'
        # — a pattern that cannot cross a brace, so any signal carrying a
        # nested object simply did not match. The agent logged
        # "signals_found=0" having actually found and verified a real
        # violation. Markdown fences are in this case too, because the
        # model usually wraps its JSON in them.

    def test_extracts_every_signal_from_an_array(self):
        text = """[
          {"nct_id": "NCT1", "signal_type": "missing_results", "summary": "a", "confidence": 0.8},
          {"nct_id": "NCT2", "signal_type": "missing_results", "summary": "b", "confidence": 0.7}
        ]"""

        signals = parse_signals(text, "agent", "missing_results")

        assert [s["nct_id"] for s in signals] == ["NCT1", "NCT2"]

    def test_braces_inside_strings_do_not_break_the_scan(self):
        text = '{"nct_id": "NCT1", "signal_type": "x", "summary": "sponsor said {nothing}", "confidence": 0.7}'

        signals = parse_signals(text, "agent", "x")

        assert len(signals) == 1
        assert "{nothing}" in signals[0]["summary"]

    def test_no_signals_marker_short_circuits(self):
        assert parse_signals("I checked all three. NO_SIGNALS_FOUND", "a", "t") == []

    def test_prose_without_json_yields_nothing(self):
        text = "The sponsor looks non-compliant but I could not confirm it."

        assert parse_signals(text, "a", "t") == []
        # Returning nothing is correct. Inventing a signal from prose
        # would mean publishing an accusation the agent never actually
        # committed to in a structured answer.

    def test_empty_and_none_input_are_safe(self):
        assert parse_signals("", "a", "t") == []
        assert parse_signals(None, "a", "t") == []

    def test_nested_objects_are_not_mistaken_for_signals(self):
        text = '{"nct_id": "NCT1", "signal_type": "x", "summary": "s", "evidence": {"a": 1}, "confidence": 0.9}'

        signals = parse_signals(text, "a", "x")

        assert len(signals) == 1
        # One signal, not two. The evidence sub-object must be consumed
        # as part of its parent, not yielded as a finding of its own.

    def test_duplicate_signals_are_collapsed(self):
        block = '{"nct_id": "NCT1", "signal_type": "x", "summary": "same", "confidence": 0.8}'

        signals = parse_signals(f"{block}\nand to summarise:\n{block}", "a", "x")

        assert len(signals) == 1
        # Models often restate a finding in a closing summary. Counting
        # it twice would double-report the violation to the analyst.

    def test_missing_signal_type_falls_back_to_the_agents_type(self):
        text = '{"nct_id": "NCT1", "summary": "s", "confidence": 0.8}'

        signals = parse_signals(text, "timeline_agent", "timeline_delay")

        assert signals[0]["signal_type"] == "timeline_delay"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0.85,   0.85),   # already correct
            ("0.85", 0.85),   # written as a string
            (85,     0.85),   # written as a percentage
            (None,   0.5),    # absent → below the review threshold
            ("high", 0.5),    # unparseable → below the review threshold
            (1.5,    1.0),    # clamped
            (-0.2,   0.0),    # clamped
        ],
    )
    def test_confidence_is_coerced_and_clamped(self, raw, expected):
        import json as _json

        fields = {"nct_id": "NCT1", "signal_type": "x", "summary": "s"}
        if raw is not None:
            fields["confidence"] = raw
        text = _json.dumps(fields)
        # Built with json.dumps so a string value is emitted with the
        # double quotes JSON requires. repr() would write 'high', which
        # is not valid JSON, and the test would be asserting against a
        # parse failure rather than against the coercion rule.

        signals = parse_signals(text, "a", "x")

        assert signals[0]["confidence"] == pytest.approx(expected)
        # Every value here comes from an LLM, so none of it is trusted.
        # The two fallbacks land on 0.5 deliberately: below the 0.6
        # auto-approve threshold, so a confidence we had to guess at
        # always reaches a human instead of being published.

    def test_evidence_is_always_a_list(self):
        as_string = '{"nct_id": "N1", "signal_type": "x", "summary": "s", "evidence": "one fact"}'
        as_dict   = '{"nct_id": "N2", "signal_type": "x", "summary": "s", "evidence": {"k": "v"}}'
        absent    = '{"nct_id": "N3", "signal_type": "x", "summary": "s"}'

        for text in (as_string, as_dict, absent):
            assert isinstance(parse_signals(text, "a", "x")[0]["evidence"], list)
        # signals.evidence is a JSONB column written with json.dumps().
        # A list keeps the stored shape consistent so the API and the
        # review UI never have to branch on type.


class TestToolExecutor:
    """execute_tool — dispatching the tool the model asked for."""

    def _tool(self):
        from langchain_core.tools import tool

        @tool
        def sample_tool(value: str) -> str:
            """Echoes the value back."""
            return f"echo:{value}"

        return sample_tool

    async def test_runs_the_requested_tool(self):
        result = await execute_tool(
            {"name": "sample_tool", "args": {"value": "hello"}}, [self._tool()]
        )

        assert result == "echo:hello"

    async def test_unknown_tool_returns_the_real_tool_names(self):
        result = await execute_tool({"name": "imaginary", "args": {}}, [self._tool()])

        assert "not available" in result
        assert "sample_tool" in result
        # Naming the available tools lets the model correct itself on the
        # next turn. Without it, a hallucinated tool name gets retried
        # until the iteration limit, wasting the whole agent run.

    async def test_a_raising_tool_becomes_an_error_string(self):
        from langchain_core.tools import tool

        @tool
        def broken_tool(value: str) -> str:
            """Always fails."""
            raise RuntimeError("database unreachable")

        result = await execute_tool(
            {"name": "broken_tool", "args": {"value": "x"}}, [broken_tool]
        )

        assert "Error executing tool" in result
        assert "database unreachable" in result
        # Never raises. An exception here would kill the agent node and
        # discard every finding it had already gathered, so a failed tool
        # is reported back into the conversation as text instead.

    async def test_oversized_output_is_truncated_with_an_explanation(self):
        from langchain_core.tools import tool

        @tool
        def chatty_tool(value: str) -> str:
            """Returns far too much text."""
            return "x" * (MAX_RESULT_CHARS + 5_000)

        result = await execute_tool(
            {"name": "chatty_tool", "args": {"value": "x"}}, [chatty_tool]
        )

        assert len(result) < MAX_RESULT_CHARS + 500
        assert "truncated" in result
        # Ten turns of unbounded tool output would overflow the model's
        # context window and fail the node. Truncating one result is the
        # cheaper loss, and the message tells the model it happened.


class TestStateReducer:
    """
    _merge_lists — the reducer that makes parallel agents possible.

    All six specialists write to state["signals"] simultaneously. This
    function is the only reason five of them do not vanish.
    """

    def test_appends_instead_of_replacing(self):
        assert _merge_lists([1, 2], [3]) == [1, 2, 3]

    def test_handles_empty_and_none(self):
        assert _merge_lists([], [1]) == [1]
        assert _merge_lists(None, [1]) == [1]
        assert _merge_lists([1], None) == [1]
        assert _merge_lists(None, None) == []

    def test_does_not_mutate_its_inputs(self):
        left, right = [1], [2]

        _merge_lists(left, right)

        assert left == [1] and right == [2]
        # LangGraph may call a reducer more than once with the same
        # inputs. Mutating them would compound the result on every call.


class TestGraphStructure:
    """
    The compiled graph — fan out to six, fan back in to one.

    Asserted structurally because the alternative is only noticing a
    missing edge when an agent silently never runs.
    """

    def test_all_six_specialists_are_registered(self):
        from graph.graph_builder import SPECIALIST_NODES, build_graph

        nodes = set(build_graph().get_graph().nodes)

        assert len(SPECIALIST_NODES) == 6
        for name in SPECIALIST_NODES:
            assert name in nodes

    def test_supervisor_fans_out_and_compile_fans_in(self):
        from graph.graph_builder import SPECIALIST_NODES, build_graph

        edges = {(e.source, e.target) for e in build_graph().get_graph().edges}

        for name in SPECIALIST_NODES:
            assert ("supervisor", name) in edges
            assert (name, "supervisor_compile") in edges
        # Both directions matter. A missing fan-out edge means that agent
        # never runs; a missing fan-in edge means the brief is compiled
        # before that agent has finished, so its findings arrive too late
        # to be included.

    def test_compile_does_not_rewrite_agents_activated(self):
        import inspect
        from agents import supervisor

        source = inspect.getsource(supervisor.supervisor_compile)

        assert '"agents_activated":' not in source
        # agents_activated carries the appending reducer. supervisor_compile
        # reads it, and returning it again appended a second copy of every
        # name — /analyze reported twelve agents for a six-agent run.


class TestHITLRouting:
    """
    The confidence threshold — what gets published and what a human sees.

    This is the system's accountability boundary, so the decision is
    tested directly rather than through the database.
    """

    def test_threshold_matches_the_supervisors_split(self):
        import inspect
        from agents import supervisor

        source = inspect.getsource(supervisor.supervisor_compile)

        assert str(CONFIDENCE_THRESHOLD) in source
        # The brief separates "findings" from "requires review" at the
        # same number the gate routes on. If the two drifted apart, the
        # brief would describe a signal as confirmed while the gate had
        # quietly parked it in the review queue.

    @pytest.mark.parametrize(
        "raw, expected",
        [(0.9, 0.9), ("0.75", 0.75), (None, 0.0), ("very high", 0.0),
         (2.0, 1.0), (-1.0, 0.0)],
    )
    def test_confidence_coercion_fails_safe(self, raw, expected):
        assert HITLGate._coerce_confidence(raw) == pytest.approx(expected)

    def test_unreadable_confidence_routes_to_a_human(self):
        assert HITLGate._coerce_confidence("nonsense") < CONFIDENCE_THRESHOLD
        assert HITLGate._coerce_confidence(None) < CONFIDENCE_THRESHOLD
        # The direction of the fallback is the point. Guessing high would
        # auto-publish an accusation on the strength of a malformed
        # field; guessing low only costs a human a few seconds.

    @pytest.mark.parametrize(
        "value, is_real",
        [("NCT04788680", True), ("NCT00011531", True),
         ("PATTERN", False), ("MULTIPLE", False), ("NCT123", False),
         ("nct04788680", False), ("", False)],
    )
    def test_only_real_nct_ids_are_treated_as_trials(self, value, is_real):
        assert bool(NCT_ID_PATTERN.match(value)) is is_real
        # signals.nct_id is a foreign key to studies. The pattern finder
        # has no single trial to name and writes labels like "PATTERN";
        # those must be stored as NULL rather than creating a fake study
        # row to satisfy the constraint.
