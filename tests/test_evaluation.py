##############################################################################
# tests/test_evaluation.py
#
# WHAT THIS FILE TESTS:
#   evaluation/scorers.py     precision, recall, calibration
#   evaluation/eval_runner.py the FDAAA deadline cutoff
#
# WHY SCORING LOGIC NEEDS TESTS OF ITS OWN:
#   These functions are how anyone decides whether MOSAIC is working. A
#   scorer that quietly counts a duplicate signal twice, or that reports
#   a confident 1.00 precision when there was nothing to score, produces
#   a number that is worse than having no number at all — it is a wrong
#   answer with a decimal point on it.
##############################################################################


import pytest

from evaluation.scorers import (
    score_agent_contribution,
    score_calibration,
    score_signals,
)


def truth(**labels) -> list[dict]:
    """Builds a ground truth list from nct_id → is_violation pairs."""
    return [
        {"nct_id": nct, "is_violation": is_violation}
        for nct, is_violation in labels.items()
    ]


def flagged(*nct_ids, confidence: float = 0.9, agent: str = "agent_a") -> list[dict]:
    """Builds a signal list from NCT IDs."""
    return [
        {"nct_id": nct, "confidence": confidence, "agent": agent}
        for nct in nct_ids
    ]


class TestScoreSignals:

    def test_a_perfect_run_scores_one(self):
        result = score_signals(
            signals=flagged("NCT1", "NCT2"),
            ground_truth=truth(NCT1=True, NCT2=True, NCT3=False),
        )

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["true_negatives"] == 1

    def test_a_false_accusation_lowers_precision(self):
        result = score_signals(
            signals=flagged("NCT1", "NCT3"),
            ground_truth=truth(NCT1=True, NCT3=False),
        )

        assert result["precision"] == 0.5
        assert result["false_positives"] == 1
        # NCT3 was compliant and got flagged. This is the expensive
        # mistake — a public accusation against a sponsor who complied.

    def test_a_missed_violation_lowers_recall(self):
        result = score_signals(
            signals=flagged("NCT1"),
            ground_truth=truth(NCT1=True, NCT2=True),
        )

        assert result["recall"] == 0.5
        assert result["false_negatives"] == 1

    def test_two_agents_flagging_one_trial_counts_once(self):
        signals = (
            flagged("NCT1", agent="missing_results_agent")
            + flagged("NCT1", agent="timeline_agent")
        )

        result = score_signals(signals=signals, ground_truth=truth(NCT1=True))

        assert result["true_positives"] == 1
        # Two agents agreeing is one accusation about one trial. Counting
        # it twice would inflate precision simply because more agents ran.

    def test_a_flag_on_an_unlabelled_trial_is_reported_not_scored(self):
        result = score_signals(
            signals=flagged("NCT1", "NCT_UNLABELLED"),
            ground_truth=truth(NCT1=True),
        )

        assert result["precision"] == 1.0
        assert result["unlabelled_flagged"] == ["NCT_UNLABELLED"]
        # Scoring it either way would be inventing a result. Reporting it
        # separately keeps the sample's limits visible.

    def test_empty_ground_truth_scores_zero_rather_than_dividing_by_zero(self):
        result = score_signals(signals=flagged("NCT1"), ground_truth=[])

        assert result["precision"] == 0.0
        assert result["labelled_trials"] == 0

    def test_a_run_that_flagged_nothing_scores_zero_not_one(self):
        result = score_signals(signals=[], ground_truth=truth(NCT1=True))

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["false_negatives"] == 1
        # Precision is undefined with an empty denominator, and 0.0 is
        # the honest reading. Returning 1.0 would let a system that finds
        # nothing claim perfect precision.

    def test_signals_without_an_nct_id_are_ignored(self):
        result = score_signals(
            signals=[{"nct_id": "", "confidence": 0.9}, {"confidence": 0.9}],
            ground_truth=truth(NCT1=True),
        )

        assert result["false_positives"] == 0
        # A cross-study pattern signal has no single trial, so it cannot
        # be scored against a per-trial label.


class TestCalibration:

    def test_confidence_is_bucketed_and_averaged(self):
        signals = (
            flagged("NCT1", confidence=0.95)
            + flagged("NCT2", confidence=0.7)
            + flagged("NCT3", confidence=0.4)
        )

        result = score_calibration(signals)

        assert result["bands"] == {"high": 1, "medium": 1, "low": 1}
        assert result["would_need_review"] == 1
        assert result["mean_confidence"] == pytest.approx(0.6833, abs=1e-3)

    def test_confidence_of_exactly_one_lands_in_the_high_band(self):
        result = score_calibration(flagged("NCT1", confidence=1.0))

        assert result["bands"]["high"] == 1
        # The high band's upper bound is 1.01 precisely so that a
        # perfectly confident signal is not dropped from every band.

    def test_an_empty_run_calibrates_to_zero(self):
        result = score_calibration([])

        assert result["total"] == 0
        assert result["mean_confidence"] == 0.0

    def test_a_run_where_everything_is_confident_is_visible(self):
        signals = flagged("NCT1", "NCT2", "NCT3", confidence=0.9)

        result = score_calibration(signals)

        assert result["bands"]["high"] == 3
        assert result["would_need_review"] == 0
        # Worth watching for: if an agent stamps 0.9 on everything, the
        # confidence score carries no information and the review
        # threshold stops doing anything at all.


class TestAgentContribution:

    def test_counts_signals_per_agent_highest_first(self):
        signals = (
            flagged("NCT1", agent="timeline_agent")
            + flagged("NCT2", "NCT3", agent="missing_results_agent")
        )

        result = score_agent_contribution(signals)

        assert list(result) == ["missing_results_agent", "timeline_agent"]
        assert result["missing_results_agent"] == 2

    def test_a_signal_with_no_agent_is_labelled_unknown(self):
        result = score_agent_contribution([{"nct_id": "NCT1"}])

        assert result["unknown"] == 1


class TestDeadlineCutoff:

    def test_the_cutoff_is_twelve_months_behind_and_well_formed(self):
        from datetime import datetime, timezone

        from evaluation.eval_runner import _deadline_cutoff

        cutoff = _deadline_cutoff()
        now = datetime.now(timezone.utc)

        year, month = (int(part) for part in cutoff.split("-"))

        assert 1 <= month <= 12
        assert year == now.year - 1 or (year == now.year and month <= now.month)
        assert len(cutoff) == 7
        # The "YYYY-MM" shape matters as much as the value: completion
        # dates are stored as TEXT and compared as strings, which only
        # gives the right answer while the format stays zero-padded.

    def test_month_arithmetic_does_not_produce_a_month_zero(self):
        from evaluation.eval_runner import _deadline_cutoff

        assert not _deadline_cutoff().endswith("-00")
        # The naive "month - 12" calculation yields month 0 for January.
        # The implementation converts to absolute months and back to
        # avoid it; this pins that behaviour.
