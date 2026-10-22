##############################################################################
# evaluation/blind_mode.py
#
# PURPOSE:
#   Removes the evaluation's own answer key from what the agents can read,
#   so a score measures reasoning instead of plumbing.
#
# THE PROBLEM THIS SOLVES:
#   eval_runner labels a trial a violation with this predicate:
#
#       status = 'COMPLETED'
#       AND results_posted = false
#       AND completion_date older than the 12-month FDAAA 801 deadline
#
#   Two paths hand that same predicate to the agent before it reasons:
#
#     1. processing/chunker.py writes "RESULTS POSTED: NO" into the text
#        that gets embedded. Any semantic search returns the label as
#        part of the retrieved chunk — the agent can read the answer out
#        of a search result without verifying anything.
#
#     2. tools/clinical_tools.py::check_results_posted applies the rule
#        itself and returns is_violation, months_overdue and
#        years_overdue. The tool does the classification; the model only
#        has to repeat it.
#
#   With both in place, a perfect score says a boolean survived the trip
#   from the parser through the embedding through the tool to the scorer.
#   It says nothing about whether the agent can apply a rule.
#
# WHAT BLIND MODE WITHHOLDS:
#   - the pre-computed verdict fields (is_violation, months_overdue,
#     years_overdue) from every tool result
#   - the "RESULTS POSTED: YES/NO" line from retrieved chunk text
#
# WHAT IT DELIBERATELY KEEPS, AND WHY:
#   results_posted stays on the verification tool's output.
#
#   This is the honest limit of the ablation. The registry's own record
#   is the ONLY evidence that results exist or do not — there is no
#   second source in this corpus. Withholding that field entirely would
#   not make the task harder, it would make it unanswerable, and the
#   agents would score zero for a reason that has nothing to do with
#   their reasoning.
#
#   So blind mode asks a narrower but answerable question: with the
#   verdict gone and the label no longer readable from search results,
#   does the agent still verify the trial and apply the 12-month rule
#   correctly itself? A score that holds up means the reasoning was
#   real. A score that collapses means the baseline was measuring the
#   tool.
#
#   Breaking the circularity completely needs labels the agents cannot
#   reach at all — an external audit set, hand adjudication, or the
#   published FDAAA compliance trackers. That is a separate piece of
#   work and this module does not pretend to substitute for it.
#
# HOW IT IS SWITCHED ON:
#   python -m evaluation.eval_runner --blind
#
#   The flag sets a ContextVar for the duration of the graph run, and
#   agents/tool_executor.py consults it on every tool call. A ContextVar
#   rather than a module global because the six specialists run
#   concurrently in one event loop — the value is inherited by the tasks
#   LangGraph spawns and cannot leak into unrelated work.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No. eval_runner imports it. It holds no entry point of its own.
##############################################################################


import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from config.logging_config import setup_logging

logger = setup_logging(__name__)


##############################################################################
# WHAT GETS WITHHELD
##############################################################################

WITHHELD_VERDICT_KEYS: tuple[str, ...] = (
    "is_violation",
    "months_overdue",
    "years_overdue",
)
# The fields where a tool has already done the agent's job.
#
# is_violation is the verdict itself. months_overdue and years_overdue
# are the arithmetic behind it — leaving those in would let the model
# read "286 months overdue" and restate the conclusion without ever
# comparing a completion date to a deadline.

WITHHELD_FLAG_KEY = "verdict_withheld"
# Added to any payload that lost a field, so the model is told the
# shortcut was removed rather than left to wonder why a field documented
# in the tool's own docstring is missing.

WITHHELD_MARKER = "RESULTS POSTED: [WITHHELD — verify with check_results_posted]"
# Replaces the embedded label line. Phrased as an instruction because
# the agent has a tool that answers it: the point is to force the
# verification call, not to hide that the field exists.

RESULTS_LABEL_PATTERN = re.compile(
    r"RESULTS\s+POSTED:\s*(?:YES|NO)",
    re.IGNORECASE,
)
# Matches what processing/chunker.py writes:
#     sections.append(f"RESULTS POSTED: {'YES' if ... else 'NO'}")
#
# Both YES and NO are redacted. Redacting only NO would leak the label
# by its absence — a compliant trial would be the only kind still
# carrying a readable line, which is a louder signal than the original.


##############################################################################
# THE SWITCH
##############################################################################

_blind: ContextVar[bool] = ContextVar("mosaic_blind_mode", default=False)
# Defaults to False so an ordinary analysis run, and the existing scored
# baseline, behave exactly as they did before this module existed.


def is_blind() -> bool:
    """
    Reports whether the current context is running blind.

    Returns:
        True if tool output should have the verdict withheld.
    """
    return _blind.get()


@contextmanager
def blind_mode(enabled: bool = True) -> Iterator[None]:
    """
    Runs a block with the verdict withheld from every tool result.

    Args:
        enabled: False runs normally, which keeps call sites free of
                 `if args.blind:` branching.

    Yields:
        None. The flag applies for the duration of the block.
    """
    token = _blind.set(bool(enabled))
    # Keeping the token and resetting it, rather than setting False on
    # the way out, is what makes nesting behave: an inner block restores
    # whatever the outer one had, not the default.

    try:
        yield
    finally:
        _blind.reset(token)
        # finally, so an agent raising mid-run cannot latch blind mode
        # on for everything that follows in the same process.


##############################################################################
# REDACTION — PURE FUNCTIONS, NEW OBJECTS ONLY
##############################################################################

def redact_text(text: str) -> str:
    """
    Replaces every embedded results-posted label in a block of text.

    Args:
        text: Any text an agent might read.

    Returns:
        A new string with each label line replaced by WITHHELD_MARKER.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"redact_text expects str, got {type(text).__name__}")
        # Fail fast and loudly. A silent pass-through here would leave
        # the label in place and quietly invalidate a scored run, which
        # is far worse than a crash in the eval harness.

    return RESULTS_LABEL_PATTERN.sub(WITHHELD_MARKER, text)


def redact_payload(payload: Any) -> Any:
    """
    Strips verdict fields and label lines from a decoded tool body.

    Walks dicts and lists to any depth. Never mutates the input —
    every container is rebuilt.

    Args:
        payload: A decoded JSON body, or any value found inside one.

    Returns:
        A new value of the same shape with the verdict removed.
    """
    if isinstance(payload, dict):
        return _redact_mapping(payload)

    if isinstance(payload, (list, tuple)):
        return [redact_payload(item) for item in payload]
        # Always a list on the way out. These bodies are about to be
        # re-serialised as JSON, where a tuple has no distinct form.

    if isinstance(payload, str):
        return redact_text(payload)
        # Reaches chunk_text inside search results, which is where the
        # embedded label actually lives.

    return payload
    # Numbers, booleans and None pass through untouched.


def _redact_mapping(payload: dict) -> dict:
    """
    Rebuilds one dict without its verdict keys.

    Args:
        payload: The dict to copy.

    Returns:
        A new dict, flagged if anything was dropped.
    """
    kept = {
        key: redact_payload(value)
        for key, value in payload.items()
        if key not in WITHHELD_VERDICT_KEYS
    }

    if any(key in payload for key in WITHHELD_VERDICT_KEYS):
        kept[WITHHELD_FLAG_KEY] = True
        # Only when something was actually removed, so a payload that
        # never carried a verdict is returned unchanged in shape.

    return kept


def redact_tool_output(text: str) -> str:
    """
    Redacts one tool result, whatever format it arrived in.

    JSON bodies are decoded, walked and re-serialised so the verdict
    fields are genuinely gone rather than string-replaced. Anything else
    falls back to a text pass, because a tool error message is plain
    text and the label must not survive by arriving in the wrong format.

    Args:
        text: The tool's output as returned to the agent.

    Returns:
        A new string safe to show a blind run. Never raises on content.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"redact_tool_output expects str, got {type(text).__name__}"
        )

    try:
        body = json.loads(text)
    except (ValueError, TypeError):
        return redact_text(text)
        # Not JSON. Expected for tool error strings and plain messages.

    if not isinstance(body, (dict, list)):
        return redact_text(text)
        # A bare JSON scalar, e.g. "5". Re-serialising would rewrite the
        # tool's formatting for no benefit.

    try:
        return json.dumps(
            redact_payload(body), indent=2, default=str, ensure_ascii=False
        )
        # ensure_ascii=False keeps the marker's em dash as a character
        # rather than escaping it to \u2014, which the model would read
        # literally in the middle of an instruction.
    except (TypeError, ValueError) as e:
        logger.error(f"Blind redaction fell back to text | error={e}")
        return redact_text(text)
        # A body that decoded but will not re-serialise is unexpected.
        # It is logged rather than swallowed, and the text pass still
        # removes the label, so the run stays valid.


def redact_if_blind(text: str) -> str:
    """
    Applies redaction only when the current context is blind.

    The single call site is agents/tool_executor.py, which every
    specialist routes its tool calls through.

    Args:
        text: The tool's output.

    Returns:
        The redacted output when blind, otherwise the input unchanged.
    """
    if not is_blind():
        return text
        # The ordinary path returns the exact object it was given, so a
        # normal run is byte-identical to one from before this feature.

    return redact_tool_output(text)
