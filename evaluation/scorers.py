##############################################################################
# evaluation/scorers.py
#
# PURPOSE:
#   Measures how good MOSAIC's signals actually are, against a labelled
#   set of trials whose true status is known.
#
# WHY THIS MATTERS MORE HERE THAN IN MOST SYSTEMS:
#   MOSAIC publicly accuses named sponsors of breaking federal law. The
#   two ways it can be wrong are not equally bad:
#
#     A FALSE POSITIVE is an accusation against a compliant sponsor.
#     That is a reputational and legal problem, and it is the expensive
#     mistake.
#
#     A FALSE NEGATIVE is a violation nobody noticed. Bad, but it is the
#     status quo — it is what happens today without the system.
#
#   So PRECISION (of what we flagged, how much was real) is the headline
#   number, and recall is read alongside it. A system with 100% recall
#   and 50% precision is unusable; one with 70% recall and 98% precision
#   is valuable. This file keeps that trade-off visible instead of
#   collapsing it into a single score.
#
# WHAT A GROUND TRUTH ENTRY LOOKS LIKE:
#   {"nct_id": "NCT00011531", "is_violation": True,
#    "signal_type": "missing_results"}
#
# HOW TO USE:
#   from evaluation.scorers import score_signals
#   result = score_signals(signals=signals, ground_truth=labels)
#   print(result["precision"], result["recall"], result["f1"])
##############################################################################


from typing import Any

from config.logging_config import setup_logging

logger = setup_logging(__name__)


CONFIDENCE_BANDS = (
    ("high",   0.8, 1.01),
    ("medium", 0.6, 0.8),
    ("low",    0.0, 0.6),
)
# Bands used by score_calibration(). The top of the high band is 1.01 so
# a confidence of exactly 1.0 lands inside it.
#
# The medium/low boundary is 0.6 on purpose: it is the same threshold
# graph/hitl.py routes on, so the "low" band is exactly the population a
# human reviewer sees.


def score_signals(
    signals: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Scores a set of generated signals against labelled trials.

    Only trials that appear in ground_truth are scored. A signal about a
    trial nobody labelled cannot be called right or wrong, so counting
    it either way would be inventing a result — those are reported
    separately as "unlabelled".

    Args:
        signals:      Signals produced by a run, each with at least
                      nct_id and confidence.
        ground_truth: Labelled trials, each with nct_id and is_violation.

    Returns:
        A dict with precision, recall, f1, the four counts, and the
        unlabelled count.
    """
    truth = {
        entry["nct_id"]: bool(entry.get("is_violation"))
        for entry in ground_truth
        if entry.get("nct_id")
    }

    if not truth:
        logger.warning("score_signals called with empty ground truth")
        return _empty_score()

    flagged = {
        signal["nct_id"]
        for signal in signals
        if signal.get("nct_id")
    }
    # A set, not a list: two agents independently flagging the same trial
    # is one accusation about one trial, not two. Counting it twice would
    # inflate whichever number it lands in.

    unlabelled = flagged - set(truth)

    true_positives  = sum(1 for nct in truth if truth[nct] and nct in flagged)
    false_positives = sum(1 for nct in truth if not truth[nct] and nct in flagged)
    false_negatives = sum(1 for nct in truth if truth[nct] and nct not in flagged)
    true_negatives  = sum(1 for nct in truth if not truth[nct] and nct not in flagged)

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall    = _safe_divide(true_positives, true_positives + false_negatives)
    f1        = _safe_divide(2 * precision * recall, precision + recall)

    logger.info(
        f"Signals scored | "
        f"precision={precision:.2f} | recall={recall:.2f} | f1={f1:.2f} | "
        f"tp={true_positives} fp={false_positives} fn={false_negatives}"
    )

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives":  true_negatives,
        "labelled_trials": len(truth),
        "unlabelled_flagged": sorted(unlabelled),
    }


def score_calibration(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Reports how confidence is distributed across a run's signals.

    WHAT THIS CATCHES:
    An agent that stamps 0.9 on everything. Its signals may even be
    mostly correct, but the confidence carries no information, and the
    whole human-review design depends on confidence meaning something —
    it is the only thing deciding what a person has to look at.

    A healthy run has a spread. A run where every signal is 0.9 means
    the threshold is doing nothing and nothing ever reaches a reviewer.

    Args:
        signals: Signals from one run.

    Returns:
        Counts per confidence band, the mean, and how many signals would
        be sent to human review.
    """
    if not signals:
        return {
            "total": 0, "mean_confidence": 0.0,
            "bands": {name: 0 for name, _, _ in CONFIDENCE_BANDS},
            "would_need_review": 0,
        }

    confidences = [float(s.get("confidence", 0.0)) for s in signals]

    bands = {
        name: sum(1 for c in confidences if low <= c < high)
        for name, low, high in CONFIDENCE_BANDS
    }

    return {
        "total":             len(signals),
        "mean_confidence":   round(sum(confidences) / len(confidences), 4),
        "bands":             bands,
        "would_need_review": bands["low"],
    }


def score_agent_contribution(signals: list[dict[str, Any]]) -> dict[str, int]:
    """
    Counts signals per agent for one run.

    Used to spot an agent that has silently stopped contributing. All six
    run on every task, so one of them returning zero signals across many
    runs is a sign its tools are failing or its prompt has drifted — and
    because an agent that finds nothing is a legitimate outcome, that
    failure is invisible without counting across runs.

    Args:
        signals: Signals from one or more runs.

    Returns:
        agent name → number of signals, highest first.
    """
    counts: dict[str, int] = {}

    for signal in signals:
        agent = signal.get("agent", "unknown")
        counts[agent] = counts.get(agent, 0) + 1

    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Divides, returning 0.0 instead of raising when the denominator is 0.

    All three of precision, recall and f1 have a legitimately empty
    denominator: a run that flagged nothing has no precision to speak of.
    0.0 is the honest answer there, and it keeps a scoring run from
    crashing on an empty result set.
    """
    return numerator / denominator if denominator else 0.0


def _empty_score() -> dict[str, Any]:
    """The zeroed result used when there is nothing to score against."""
    return {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "true_positives": 0, "false_positives": 0,
        "false_negatives": 0, "true_negatives": 0,
        "labelled_trials": 0, "unlabelled_flagged": [],
    }
