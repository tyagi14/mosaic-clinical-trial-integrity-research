##############################################################################
# tests/test_eval_runner.py
#
# WHAT THIS FILE TESTS:
#   evaluation/eval_runner.py — how the labelled sample is built, and how
#   a run is turned into a report.
#
# WHY THE LABELLING IS THE PART THAT MATTERS:
#   Every accuracy number MOSAIC reports rests on these labels. If the
#   "violation" query also caught trials still inside their 12-month
#   window, then correct agent behaviour would be scored as a miss, and
#   the system would look worse than it is. If the "compliant" sample
#   were empty, false positives would become invisible — precision would
#   read 1.00 no matter how many compliant sponsors got accused.
##############################################################################


import pytest

from evaluation import eval_runner
from tests.conftest import FakeConnection


class FakeGraph:
    def __init__(self, signals=None, errors=None):
        self.signals = signals if signals is not None else []
        self.errors = errors or []
        self.state = None

    async def ainvoke(self, state):
        self.state = state
        return {
            "run_id": state["run_id"],
            "signals": self.signals,
            "agents_activated": ["missing_results_agent", "timeline_agent"],
            "error_log": self.errors,
            "final_brief": "a brief",
        }


@pytest.fixture
def fake_db(monkeypatch):
    """Replaces asyncpg.connect with a fake connection."""
    connection = FakeConnection()

    async def fake_connect(**kwargs):
        return connection

    monkeypatch.setattr(eval_runner.asyncpg, "connect", fake_connect)

    async def fake_close():
        return None

    connection.close = fake_close
    return connection


def overdue_rows(*nct_ids):
    return [
        {"nct_id": nct, "sponsor": "A Sponsor", "completion_date": "2001-12"}
        for nct in nct_ids
    ]


def posted_rows(*nct_ids):
    return [
        {"nct_id": nct, "sponsor": "B Sponsor", "completion_date": "2024-01"}
        for nct in nct_ids
    ]


class TestGroundTruth:

    async def test_both_classes_are_sampled_and_labelled(self, fake_db):
        fake_db._fetch = [overdue_rows("NCT1", "NCT2"), posted_rows("NCT3")]

        labelled = await eval_runner.build_ground_truth(violations=2, compliant=1)

        assert [entry["is_violation"] for entry in labelled] == [True, True, False]
        # Compliant trials are in the sample on purpose. They are the only
        # way a false positive can be counted at all.

    async def test_the_violation_query_excludes_trials_inside_the_deadline(
        self, fake_db
    ):
        fake_db._fetch = [[], []]

        await eval_runner.build_ground_truth()

        violation_sql, violation_args = fake_db.statements[0]
        assert "results_posted = false" in violation_sql
        assert "status = 'COMPLETED'" in violation_sql
        assert "completion_date < $1" in violation_sql
        assert violation_args[0] == eval_runner._deadline_cutoff()
        # The cutoff is what keeps a trial that completed last month out
        # of the "violation" class. Without it, the agents would be
        # marked wrong for correctly NOT flagging it.

    async def test_the_compliant_query_requires_posted_results(self, fake_db):
        fake_db._fetch = [[], []]

        await eval_runner.build_ground_truth()

        compliant_sql, _ = fake_db.statements[1]
        assert "results_posted = true" in compliant_sql

    async def test_blank_completion_dates_are_excluded(self, fake_db):
        fake_db._fetch = [[], []]

        await eval_runner.build_ground_truth()

        assert "completion_date <> ''" in fake_db.statements[0][0]
        # completion_date is TEXT and is sometimes empty in the registry.
        # An empty string sorts BELOW every real date, so without this
        # filter those trials would all be labelled overdue.

    async def test_the_sample_sizes_are_passed_to_the_queries(self, fake_db):
        fake_db._fetch = [[], []]

        await eval_runner.build_ground_truth(violations=7, compliant=4)

        assert fake_db.statements[0][1][1] == 7
        assert fake_db.statements[1][1][0] == 4


class TestRunEvaluation:

    async def test_a_dry_run_returns_the_sample_without_running_agents(
        self, fake_db
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT2")]

        report = await eval_runner.run_evaluation(dry_run=True)

        assert report["dry_run"] is True
        assert len(report["ground_truth"]) == 2
        assert "accuracy" not in report
        # Costs nothing and makes no LLM calls, so the labelling can be
        # checked before any tokens are spent on a real run.

    async def test_an_empty_sample_is_reported_rather_than_scored(self, fake_db):
        fake_db._fetch = [[], []]

        report = await eval_runner.run_evaluation()

        assert report["error"] == "empty ground truth"
        # Scoring against nothing would produce a confident 0.00 that
        # looks like a failing system rather than an empty database.

    async def test_a_full_run_scores_the_signals_it_found(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1", "NCT2"), posted_rows("NCT3")]

        graph = FakeGraph(signals=[
            {"nct_id": "NCT1", "agent": "missing_results_agent", "confidence": 0.9},
            {"nct_id": "NCT2", "agent": "timeline_agent", "confidence": 0.85},
        ])

        import graph.graph_builder as graph_builder
        monkeypatch.setattr(graph_builder, "mosaic_graph", graph)

        report = await eval_runner.run_evaluation(violations=2, compliant=1)

        assert report["trials_evaluated"] == 3
        assert report["signals_found"] == 2
        assert report["accuracy"]["precision"] == 1.0
        assert report["accuracy"]["recall"] == 1.0
        assert report["accuracy"]["false_positives"] == 0
        assert report["by_agent"]["missing_results_agent"] == 1

    async def test_a_false_positive_shows_up_in_the_report(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]

        graph = FakeGraph(signals=[
            {"nct_id": "NCT3", "agent": "missing_results_agent", "confidence": 0.9},
        ])

        import graph.graph_builder as graph_builder
        monkeypatch.setattr(graph_builder, "mosaic_graph", graph)

        report = await eval_runner.run_evaluation(violations=1, compliant=1)

        assert report["accuracy"]["false_positives"] == 1
        assert report["accuracy"]["precision"] == 0.0
        # NCT3 posted its results and was flagged anyway. This is the
        # failure the evaluation exists to surface.

    async def test_the_whole_sample_is_handed_to_the_graph(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]

        graph = FakeGraph()
        import graph.graph_builder as graph_builder
        monkeypatch.setattr(graph_builder, "mosaic_graph", graph)

        await eval_runner.run_evaluation(violations=1, compliant=1)

        assert graph.state["nct_ids"] == ["NCT1", "NCT3"]
        assert graph.state["max_studies"] == 2
        assert "FDAAA 801" in graph.state["task"]

    async def test_errors_during_the_run_are_carried_into_the_report(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]

        graph = FakeGraph(errors=["timeline_agent: API unreachable"])
        import graph.graph_builder as graph_builder
        monkeypatch.setattr(graph_builder, "mosaic_graph", graph)

        report = await eval_runner.run_evaluation(violations=1, compliant=1)

        assert report["errors"] == ["timeline_agent: API unreachable"]
        # An agent that failed changes how the scores should be read: a
        # missed violation may be a crashed agent, not a bad judgment.


class TestReportPrinting:
    """
    _print_report — exercised so a formatting bug cannot crash the CLI
    after a run has already spent real money on LLM calls.
    """

    def test_printing_a_full_report_does_not_raise(self, capsys):
        eval_runner._print_report({
            "run_id": "r1", "trials_evaluated": 3, "signals_found": 2,
            "duration_seconds": 28.4,
            "agents_activated": ["a", "b"],
            "errors": ["timeline_agent: failed"],
            "accuracy": {
                "precision": 1.0, "recall": 0.5, "f1": 0.67,
                "true_positives": 1, "false_positives": 1,
                "false_negatives": 1, "true_negatives": 0,
            },
            "calibration": {
                "mean_confidence": 0.88, "bands": {"high": 2, "medium": 0, "low": 0},
                "would_need_review": 0,
            },
            "by_agent": {"missing_results_agent": 2},
        })

        output = capsys.readouterr().out
        assert "precision" in output
        assert "false positives present" in output
        # The false-positive warning is printed because precision alone
        # is easy to skim past, and a false positive is the one result
        # that needs a human to look at it.

    def test_printing_a_dry_run_lists_the_labels(self, capsys):
        eval_runner._print_report({
            "dry_run": True,
            "ground_truth": [
                {"nct_id": "NCT1", "is_violation": True,
                 "sponsor": "A Sponsor", "completion_date": "2001-12"},
                {"nct_id": "NCT2", "is_violation": False,
                 "sponsor": "B Sponsor", "completion_date": "2024-01"},
            ],
        })

        output = capsys.readouterr().out
        assert "VIOLATION" in output
        assert "compliant" in output

    def test_printing_an_error_report_does_not_raise(self, capsys):
        eval_runner._print_report({"error": "empty ground truth"})

        assert "could not run" in capsys.readouterr().out


class TestBlindRuns:
    """
    --blind — the ablation that withholds the pre-computed verdict from
    the agents for the duration of the graph run.

    WHY THIS IS TESTED SEPARATELY FROM blind_mode ITSELF:
    tests/test_blind_mode.py proves the redaction is correct. These tests
    prove it is actually SWITCHED ON while the agents are running, and
    recorded in the report afterwards. A redactor that works perfectly
    but never engages would produce a blind score identical to the
    baseline and nobody would be able to tell from the output.
    """

    class _BlindRecordingGraph(FakeGraph):
        """Records whether blind mode was active at invoke time."""

        def __init__(self, signals=None, errors=None):
            super().__init__(signals=signals, errors=errors)
            self.blind_during_run = None

        async def ainvoke(self, state):
            from evaluation.blind_mode import is_blind

            self.blind_during_run = is_blind()
            return await super().ainvoke(state)

    def _patch_graph(self, monkeypatch, graph):
        import graph.graph_builder as graph_builder

        monkeypatch.setattr(graph_builder, "mosaic_graph", graph)

    async def test_blind_is_active_while_the_graph_runs(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]
        graph = self._BlindRecordingGraph()
        self._patch_graph(monkeypatch, graph)

        await eval_runner.run_evaluation(violations=1, compliant=1, blind=True)

        assert graph.blind_during_run is True
        # The whole point. Every tool call the agents make during this
        # invoke is what gets redacted.

    async def test_normal_runs_are_not_blind(self, fake_db, monkeypatch):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]
        graph = self._BlindRecordingGraph()
        self._patch_graph(monkeypatch, graph)

        await eval_runner.run_evaluation(violations=1, compliant=1)

        assert graph.blind_during_run is False
        # The baseline must stay exactly as it was, or the two scores
        # are not comparable and the ablation tells you nothing.

    async def test_the_report_records_the_mode(self, fake_db, monkeypatch):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]
        self._patch_graph(monkeypatch, self._BlindRecordingGraph())

        report = await eval_runner.run_evaluation(
            violations=1, compliant=1, blind=True
        )

        assert report["blind"] is True
        # A saved JSON report has to say which mode produced it. Two
        # files of numbers with no mode field would be indistinguishable
        # a week later.

    async def test_normal_report_records_the_mode_too(
        self, fake_db, monkeypatch
    ):
        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]
        self._patch_graph(monkeypatch, self._BlindRecordingGraph())

        report = await eval_runner.run_evaluation(violations=1, compliant=1)

        assert report["blind"] is False

    async def test_blind_is_released_after_the_run(
        self, fake_db, monkeypatch
    ):
        from evaluation.blind_mode import is_blind

        fake_db._fetch = [overdue_rows("NCT1"), posted_rows("NCT3")]
        self._patch_graph(monkeypatch, self._BlindRecordingGraph())

        await eval_runner.run_evaluation(violations=1, compliant=1, blind=True)

        assert is_blind() is False
        # An eval run must not leave the process redacting tool output
        # for any analysis that follows it.

    def test_the_printed_report_states_the_mode(self, capsys):
        eval_runner._print_report({
            "run_id": "r1", "trials_evaluated": 2, "signals_found": 1,
            "duration_seconds": 10.0, "agents_activated": ["a"],
            "errors": [], "blind": True,
            "accuracy": {
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "true_positives": 1, "false_positives": 0,
                "false_negatives": 0, "true_negatives": 1,
            },
            "calibration": {
                "mean_confidence": 0.9,
                "bands": {"high": 1, "medium": 0, "low": 0},
                "would_need_review": 0,
            },
            "by_agent": {"timeline_agent": 1},
        })

        output = capsys.readouterr().out
        assert "BLIND" in output
        # Printed loudly, because a blind score and a baseline score
        # answer different questions and the numbers look identical.

    def test_the_printed_baseline_report_says_so(self, capsys):
        eval_runner._print_report({
            "run_id": "r1", "trials_evaluated": 2, "signals_found": 1,
            "duration_seconds": 10.0, "agents_activated": ["a"],
            "errors": [], "blind": False,
            "accuracy": {
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "true_positives": 1, "false_positives": 0,
                "false_negatives": 0, "true_negatives": 1,
            },
            "calibration": {
                "mean_confidence": 0.9,
                "bands": {"high": 1, "medium": 0, "low": 0},
                "would_need_review": 0,
            },
            "by_agent": {"timeline_agent": 1},
        })

        assert "baseline" in capsys.readouterr().out.lower()

    def test_the_cli_accepts_the_flag(self):
        import argparse

        parser = eval_runner._build_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.parse_args(["--blind"]).blind is True
        assert parser.parse_args([]).blind is False


class TestJsonOutputIsParseable:
    """
    --json has to emit JSON and nothing else.

    WHY THIS NEEDS A TEST:
    config/logging_config.py sends every log line to stdout on purpose,
    because Cloud Run captures stdout and tags it INFO in Cloud Logging.
    That is right for the deployed service and wrong for a CLI whose
    output you pipe into a file: the report came out preceded by a
    hundred INFO lines, and json.load() failed on the first one. Comparing
    a baseline run against a blind run means diffing two JSON files, so
    this is the mode that matters for the ablation.
    """

    def test_log_records_move_off_stdout(self):
        import logging
        import sys

        import config.logging_config as logging_config

        original_stream = logging_config._log_stream
        logger = logging_config.setup_logging("blind_mode_json_probe")
        handler = logger.handlers[0]

        try:
            logging_config.route_logs_to_stderr()

            assert handler.stream is sys.stderr
            # An ALREADY-created handler gets switched. Every module in
            # the project builds its logger at import time, so routing
            # only future handlers would leave most of them on stdout.

            later = logging_config.setup_logging("blind_mode_json_probe_2")
            assert later.handlers[0].stream is sys.stderr
            # And handlers created afterwards. The graph and its six
            # agents are imported lazily, so their loggers do not exist
            # yet when the CLI makes this call.

            assert logger.isEnabledFor(logging.WARNING) is True
            # Nothing is silenced. A run that warned still says so — a
            # clean, parseable, wrong report would be worse than a
            # broken one.
        finally:
            logging_config._log_stream = original_stream
            handler.setStream(original_stream)
            for name in ("blind_mode_json_probe", "blind_mode_json_probe_2"):
                logging.getLogger(name).handlers.clear()
