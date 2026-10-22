##############################################################################
# tests/test_blind_mode.py
#
# WHAT THIS FILE TESTS:
#   evaluation/blind_mode.py — the ablation that stops the agents reading
#   the evaluation's own answer key out of their tool results.
#
# WHY THIS EXISTS AT ALL:
#   The scored evaluation labels a trial a violation with this predicate:
#
#       status = 'COMPLETED' AND results_posted = false
#       AND completion_date older than the 12-month deadline
#
#   Two things then hand that same predicate straight to the agent:
#
#     1. processing/chunker.py writes "RESULTS POSTED: NO" into the text
#        that gets embedded, so a semantic search returns the label as
#        part of the retrieved chunk.
#     2. tools/clinical_tools.py::check_results_posted applies the rule
#        itself and returns is_violation / months_overdue / years_overdue.
#
#   So a perfect score measures whether a boolean survived the trip from
#   the parser to the scorer, not whether anything reasoned. Blind mode
#   removes both shortcuts and leaves the raw fields, so the agent has to
#   verify the trial and do its own arithmetic.
#
#   What blind mode deliberately does NOT do is hide results_posted from
#   the verification tool. The registry's own records are the only
#   evidence that results exist, so withholding that field entirely would
#   make the task unanswerable rather than harder. See the module docstring.
##############################################################################


import json

import pytest

from agents.tool_executor import execute_tool
from evaluation.blind_mode import (
    WITHHELD_MARKER,
    WITHHELD_VERDICT_KEYS,
    blind_mode,
    is_blind,
    redact_payload,
    redact_text,
    redact_tool_output,
)


# A real check_results_posted payload, shape copied from the live tool.
VERDICT_PAYLOAD = {
    "nct_id": "NCT00011531",
    "status": "COMPLETED",
    "results_posted": False,
    "completion_date": "2001-12",
    "sponsor": "US Department of Veterans Affairs",
    "months_overdue": 286,
    "years_overdue": 23.8,
    "is_violation": True,
}

# A real search_studies_by_meaning payload, trimmed to the parts that matter.
SEARCH_PAYLOAD = {
    "results": [
        {
            "nct_id": "NCT00011531",
            "similarity": 0.41,
            "chunk_text": (
                "TITLE: A Study Of Something\n"
                "SPONSOR: US Department of Veterans Affairs\n"
                "COMPLETION DATE: 2001-12\n"
                "RESULTS POSTED: NO\n"
                "ENROLLMENT: 240"
            ),
        }
    ],
    "count": 1,
    "query": "completed trials with no results",
}


class TestBlindFlag:
    """is_blind / blind_mode — the switch itself."""

    def test_off_by_default(self):
        assert is_blind() is False
        # Normal runs must be untouched. The ablation is opt-in, so a
        # forgotten reset can never silently degrade a scored run.

    def test_on_inside_the_context(self):
        with blind_mode():
            assert is_blind() is True

    def test_restored_after_the_context(self):
        with blind_mode():
            pass

        assert is_blind() is False

    def test_restored_even_when_the_body_raises(self):
        with pytest.raises(RuntimeError):
            with blind_mode():
                raise RuntimeError("boom")

        assert is_blind() is False
        # An agent crashing mid-run must not leave blind mode latched on
        # for everything that follows in the same process.

    def test_can_be_explicitly_disabled(self):
        with blind_mode(enabled=False):
            assert is_blind() is False

    def test_nests(self):
        with blind_mode():
            with blind_mode(enabled=False):
                assert is_blind() is False
            assert is_blind() is True


class TestRedactText:
    """redact_text — the embedded label line inside free text."""

    def test_replaces_a_no_label(self):
        assert WITHHELD_MARKER in redact_text("RESULTS POSTED: NO")

    def test_replaces_a_yes_label(self):
        out = redact_text("RESULTS POSTED: YES")

        assert WITHHELD_MARKER in out
        assert "YES" not in out
        # Both classes get redacted. Hiding only "NO" would leak the
        # label by its absence — a compliant trial would be the only
        # kind still carrying a readable line.

    def test_is_case_insensitive(self):
        assert WITHHELD_MARKER in redact_text("results posted: no")

    def test_replaces_every_occurrence(self):
        out = redact_text("RESULTS POSTED: NO\n---\nRESULTS POSTED: YES")

        assert out.count(WITHHELD_MARKER) == 2

    def test_leaves_surrounding_text_intact(self):
        out = redact_text("COMPLETION DATE: 2001-12\nRESULTS POSTED: NO\nENROLLMENT: 240")

        assert "COMPLETION DATE: 2001-12" in out
        assert "ENROLLMENT: 240" in out
        # The agent still needs the completion date to apply the deadline
        # rule. Blind mode removes the answer, not the evidence.

    def test_does_not_mutate_its_input(self):
        original = "RESULTS POSTED: NO"
        redact_text(original)

        assert original == "RESULTS POSTED: NO"

    def test_rejects_non_text(self):
        with pytest.raises(TypeError):
            redact_text({"not": "a string"})


class TestRedactPayload:
    """redact_payload — dropping the pre-computed verdict from a decoded body."""

    def test_drops_every_verdict_key(self):
        out = redact_payload(VERDICT_PAYLOAD)

        for key in WITHHELD_VERDICT_KEYS:
            assert key not in out

    def test_keeps_the_raw_fields(self):
        out = redact_payload(VERDICT_PAYLOAD)

        assert out["status"] == "COMPLETED"
        assert out["results_posted"] is False
        assert out["completion_date"] == "2001-12"
        # results_posted stays on purpose. It is the registry's own
        # record and the only evidence that results are absent; the
        # agent must still apply the 12-month rule to it itself.

    def test_flags_that_something_was_withheld(self):
        out = redact_payload(VERDICT_PAYLOAD)

        assert out["verdict_withheld"] is True
        # The model is told the shortcut was removed rather than being
        # left to wonder why a documented field vanished.

    def test_adds_no_flag_when_nothing_was_dropped(self):
        out = redact_payload({"nct_id": "NCT1", "status": "RECRUITING"})

        assert "verdict_withheld" not in out

    def test_does_not_mutate_its_input(self):
        before = dict(VERDICT_PAYLOAD)
        redact_payload(VERDICT_PAYLOAD)

        assert VERDICT_PAYLOAD == before

    def test_reaches_nested_dicts(self):
        out = redact_payload({"study": dict(VERDICT_PAYLOAD)})

        assert "is_violation" not in out["study"]
        # fetch_study_details nests its body under a "study" key, so a
        # top-level-only pass would miss it entirely.

    def test_reaches_dicts_inside_lists(self):
        out = redact_payload({"items": [dict(VERDICT_PAYLOAD)]})

        assert "months_overdue" not in out["items"][0]

    def test_redacts_label_lines_in_nested_strings(self):
        out = redact_payload(SEARCH_PAYLOAD)
        chunk = out["results"][0]["chunk_text"]

        assert WITHHELD_MARKER in chunk
        assert "RESULTS POSTED: NO" not in chunk

    def test_passes_scalars_through(self):
        assert redact_payload(7) == 7
        assert redact_payload(None) is None
        assert redact_payload(True) is True


class TestRedactToolOutput:
    """redact_tool_output — what the agent actually receives."""

    def test_strips_the_verdict_from_tool_json(self):
        out = json.loads(redact_tool_output(json.dumps(VERDICT_PAYLOAD)))

        assert "is_violation" not in out
        assert out["results_posted"] is False

    def test_output_stays_valid_json(self):
        json.loads(redact_tool_output(json.dumps(VERDICT_PAYLOAD, indent=2)))
        # The agent parses these strings. Redaction that produced
        # unparseable output would break the reasoning loop instead of
        # constraining it.

    def test_strips_the_label_from_retrieved_chunks(self):
        out = redact_tool_output(json.dumps(SEARCH_PAYLOAD, indent=2))

        assert "RESULTS POSTED: NO" not in out
        assert WITHHELD_MARKER in out

    def test_redacts_plain_text_too(self):
        assert WITHHELD_MARKER in redact_tool_output("RESULTS POSTED: NO")
        # Not every tool returns JSON, and a tool error message is plain
        # text. The label must not survive by arriving in the wrong format.

    def test_survives_malformed_json(self):
        out = redact_tool_output('{"broken": ')

        assert isinstance(out, str)
        # Never raise. This runs inside the agent loop, where an
        # exception costs every finding the node has gathered so far.

    def test_json_list_bodies_are_handled(self):
        out = json.loads(redact_tool_output(json.dumps([VERDICT_PAYLOAD])))

        assert "is_violation" not in out[0]

    def test_bare_json_scalars_pass_through(self):
        assert redact_tool_output("5") == "5"
        # json.loads("5") succeeds and yields an int. Re-serialising it
        # would rewrite a tool's formatting for no benefit, so the text
        # path handles it.

    def test_falls_back_when_a_body_will_not_re_serialise(self, monkeypatch):
        import evaluation.blind_mode as module

        payload = json.dumps({"chunk": "RESULTS POSTED: NO"})
        # Built BEFORE the patch: blind_mode imports the stdlib json
        # module itself, so patching its dumps would break this line too.

        def exploding_dumps(*args, **kwargs):
            raise TypeError("not serialisable")

        monkeypatch.setattr(module.json, "dumps", exploding_dumps)

        out = redact_tool_output(payload)

        assert WITHHELD_MARKER in out
        # The label still has to go even on the fallback path. A silent
        # pass-through here would leak the answer key into a blind run
        # and quietly invalidate the score.

    def test_rejects_non_text(self):
        with pytest.raises(TypeError):
            redact_tool_output(None)


class TestToolExecutorIntegration:
    """execute_tool — redaction applied at the one chokepoint all agents share."""

    def _verdict_tool(self):
        from langchain_core.tools import tool

        @tool
        def check_results_posted(nct_id: str) -> str:
            """Returns a canned verdict payload."""
            return json.dumps(dict(VERDICT_PAYLOAD, nct_id=nct_id), indent=2)

        return check_results_posted

    async def test_untouched_when_not_blind(self):
        result = await execute_tool(
            {"name": "check_results_posted", "args": {"nct_id": "NCT1"}},
            [self._verdict_tool()],
        )

        assert json.loads(result)["is_violation"] is True
        # The default path must be byte-identical to before this feature
        # existed, so the existing baseline score stays comparable.

    async def test_redacted_when_blind(self):
        with blind_mode():
            result = await execute_tool(
                {"name": "check_results_posted", "args": {"nct_id": "NCT1"}},
                [self._verdict_tool()],
            )

        body = json.loads(result)
        assert "is_violation" not in body
        assert body["results_posted"] is False

    async def test_error_messages_still_come_back(self):
        with blind_mode():
            result = await execute_tool(
                {"name": "imaginary", "args": {}}, [self._verdict_tool()]
            )

        assert "not available" in result
        # Redaction must not swallow the self-correction hint that lets
        # the model recover from a hallucinated tool name.
