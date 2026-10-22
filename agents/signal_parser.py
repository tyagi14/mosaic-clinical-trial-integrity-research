##############################################################################
# agents/signal_parser.py
#
# PURPOSE:
#   Turns an agent's free-text answer into structured SignalOutput dicts.
#
# WHY THIS IS ITS OWN MODULE:
#   All six specialists finish the same way: GPT-4o writes a paragraph of
#   reasoning with one or more JSON signal blocks embedded in it, and the
#   node has to pull those blocks out. That logic was copy-pasted into all
#   six agent files, byte for byte. One shared, well-tested parser is
#   better than six identical ones that can drift apart.
#
# WHY THE OLD REGEX WAS NOT GOOD ENOUGH:
#   The original parser used:
#
#       re.compile(r'\{[^{}]*"signal_type"[^{}]*\}')
#
#   [^{}]* means "any character that is not a brace", so the pattern can
#   only ever match a FLAT JSON object. The moment GPT-4o writes a signal
#   containing a nested object — which it does, for example:
#
#       {"nct_id": "NCT00011531", "signal_type": "missing_results",
#        "evidence": {"completion": "2001-12", "months_overdue": 286},
#        "confidence": 0.9}
#
#   the regex fails to match and the finding is silently thrown away. The
#   agent reports "signals_found=0" even though it did the work and found
#   the violation. That is the worst kind of bug: the pipeline looks
#   healthy and the intelligence quietly goes missing.
#
#   This module scans for BALANCED braces instead, so nesting, markdown
#   code fences and JSON arrays of signals all parse correctly.
#
# HOW TO USE:
#   from agents.signal_parser import parse_signals
#   signals = parse_signals(response.content, AGENT_NAME, SIGNAL_TYPE)
##############################################################################


import json
from typing import Any, Iterator

from graph.state import SignalOutput
from config.logging_config import setup_logging

logger = setup_logging(__name__)


NO_SIGNALS_MARKER = "NO_SIGNALS_FOUND"
# The exact phrase every agent's system prompt tells it to use when it
# found nothing. Checked first so a clean run costs no parsing at all.

DEFAULT_CONFIDENCE = 0.5
# Used when a signal arrives with no confidence field at all. Deliberately
# below the 0.6 auto-approve threshold in graph/hitl.py: a signal whose
# confidence we had to guess must be seen by a human.

PERCENT_SCALE_MIN = 10.0
PERCENT_SCALE_MAX = 100.0
# The window in which a confidence value is read as a percentage rather
# than a 0-1 score. Starts at 10 so that a value like 1.5 — a model
# overshooting the 0-1 scale — is clamped to 1.0 instead of being
# misread as 1.5%.

REQUIRED_ANY = ("signal_type", "nct_id")
# A parsed object must carry at least one of these to count as a signal.
# This is what stops us mistaking a nested "evidence" object, or some
# unrelated JSON the model happened to print, for a finding.


def parse_signals(
    response_text: str | None,
    agent_name: str,
    default_signal_type: str,
) -> list[SignalOutput]:
    """
    Extracts every signal an agent reported in its final answer.

    Args:
        response_text:       The agent's final message content.
        agent_name:          Stamped onto each signal so the API, the
                             brief and the learning loop all know which
                             agent to credit — and which agent to teach
                             if a human rejects it.
        default_signal_type: This agent's signal type, used when the
                             model omitted the field.

    Returns:
        A list of SignalOutput dicts, in the order they appeared.
        Empty list when the agent found nothing, said NO_SIGNALS_FOUND,
        or wrote nothing parseable.
    """
    if not response_text or NO_SIGNALS_MARKER in response_text:
        return []

    signals: list[SignalOutput] = []
    seen: set[tuple[str, str]] = set()
    # Guards against the same finding being counted twice when a model
    # both lists a signal inline and repeats it in a summary block.

    for candidate in _iter_json_objects(response_text):
        data = _load_object(candidate)

        if data is None or not any(key in data for key in REQUIRED_ANY):
            continue

        signal = _build_signal(data, agent_name, default_signal_type)

        fingerprint = (signal["nct_id"], signal["summary"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        signals.append(signal)

    if not signals:
        logger.warning(
            f"No parseable signal found in agent output | "
            f"agent={agent_name} | chars={len(response_text)}"
        )
        # Logged rather than ignored: an agent that reasoned at length
        # and produced nothing parseable is worth seeing in the logs,
        # because it usually means the model answered in prose instead
        # of the JSON format its prompt asked for.

    return signals


def _build_signal(
    data: dict[str, Any],
    agent_name: str,
    default_signal_type: str,
) -> SignalOutput:
    """
    Normalises one parsed JSON object into a SignalOutput.

    Everything here comes from an LLM, so every field is treated as
    untrusted: types are coerced, missing keys get defaults, and nothing
    is allowed through in a shape the database or the API would reject.

    Args:
        data:                One parsed JSON object from the response.
        agent_name:          The agent that produced it.
        default_signal_type: Fallback when signal_type is missing.

    Returns:
        A fully populated SignalOutput dict.
    """
    return {
        "agent":       agent_name,
        "signal_type": str(data.get("signal_type") or default_signal_type),
        "nct_id":      str(data.get("nct_id") or "").strip(),
        "summary":     str(data.get("summary") or "").strip(),
        "evidence":    _normalise_evidence(data.get("evidence")),
        "confidence":  _normalise_confidence(data.get("confidence")),
    }


def _normalise_evidence(raw: Any) -> list:
    """
    Forces the evidence field into a list, whatever the model wrote.

    Models variously return a list of strings (as asked), a single
    string, or a dict of labelled facts. All three are useful to a human
    reader, so none is discarded — they are just reshaped into the list
    the signals.evidence JSONB column expects.

    Args:
        raw: The evidence field exactly as the model wrote it.

    Returns:
        A list, possibly empty. Never None.
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        return [f"{key}: {value}" for key, value in raw.items()]

    return [str(raw)]


def _normalise_confidence(raw: Any) -> float:
    """
    Coerces the confidence field to a float in the range 0.0 - 1.0.

    Models sometimes write "0.85" as a string, or 85 meaning percent.

    Values from 10 to 100 are read as percentages, because clamping 85
    down to 1.0 would push a signal straight past human review on what
    was really a formatting slip. Anything else above 1.0 is clamped
    instead of divided: 1.5 is a model overshooting a 0-1 scale, and
    reading it as 1.5% (0.015) would be a far worse misreading than
    treating it as maximum confidence.

    Args:
        raw: The confidence field as written by the model.

    Returns:
        A float between 0.0 and 1.0.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE

    if PERCENT_SCALE_MIN <= value <= PERCENT_SCALE_MAX:
        value = value / PERCENT_SCALE_MAX
        # 85 → 0.85

    return max(0.0, min(1.0, value))


def _load_object(candidate: str) -> dict[str, Any] | None:
    """
    Parses one candidate string as a JSON object.

    Args:
        candidate: A balanced "{...}" slice of the response.

    Returns:
        The parsed dict, or None if it was not valid JSON or not a dict.
    """
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _iter_json_objects(text: str) -> Iterator[str]:
    """
    Yields every top-level balanced {...} block found in the text.

    WHY A HAND-WRITTEN SCANNER AND NOT A REGEX:
    Regular expressions cannot count. Matching balanced brackets to
    arbitrary depth is outside what they can express, which is exactly
    why the original one-line pattern dropped any signal containing a
    nested object.

    HOW IT WORKS:
    Walk the text once. On an unquoted "{", remember where the object
    started and count depth up and down until it returns to zero — that
    slice is one complete object. Then continue scanning AFTER it, so
    objects nested inside it are never yielded separately (an "evidence"
    sub-object is part of its signal, not a signal of its own).

    Quote handling matters: a brace inside a JSON string, as in
    {"summary": "the sponsor said {nothing}"}, must not change the depth
    count, and a quote escaped with a backslash must not end the string.

    Markdown code fences need no special handling — ```json wrappers sit
    outside the braces and are simply skipped as ordinary text.

    Args:
        text: The agent's full response.

    Yields:
        Each balanced object as a string, in order of appearance.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
            # Inside a string nothing else can change the structure,
            # so no brace or depth logic runs here.

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
                # A stray closing brace in prose. Ignore it rather than
                # letting depth go negative and desynchronise the scan.
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:index + 1]
                start = -1
