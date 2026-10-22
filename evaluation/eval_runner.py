##############################################################################
# evaluation/eval_runner.py
#
# PURPOSE:
#   Runs MOSAIC over a labelled sample of trials and scores what it found.
#
# HOW TO RUN:
#   # free — build and print the labelled sample, call no agents:
#   python -m evaluation.eval_runner --dry-run
#
#   # the real thing — runs all six agents, costs OpenAI tokens:
#   python -m evaluation.eval_runner --violations 3 --compliant 3
#
# WHERE THE GROUND TRUTH COMES FROM:
#   The studies table, labelled by the FDAAA 801 rule itself rather than
#   by hand:
#
#     COMPLETED, no results posted, primary completion > 12 months ago
#         → is_violation = True
#     COMPLETED, results posted
#         → is_violation = False
#
#   This is a deliberate choice with a real limitation, stated plainly:
#   the labels come from the same registry fields the agents can read, so
#   this measures whether the agents correctly apply the rule to the data
#   — not whether the registry itself is accurate. A sponsor who filed
#   results late and backdated them, or whose record is simply wrong,
#   is mislabelled here too.
#
#   Both classes are sampled on purpose. Scoring only known violations
#   would measure recall while hiding false positives, which are the
#   expensive mistake — see the note at the top of scorers.py.
##############################################################################


import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone

import asyncpg

from evaluation.blind_mode import blind_mode
# Withholds the pre-computed violation verdict from every tool result
# for the duration of the graph run. See evaluation/blind_mode.py for
# what is withheld, what is kept, and why the distinction matters.

from evaluation.scorers import (
    score_agent_contribution,
    score_calibration,
    score_signals,
)
from config.settings import settings
from config.logging_config import route_logs_to_stderr, setup_logging

logger = setup_logging(__name__)


POSTING_DEADLINE_MONTHS = 12
# FDAAA 801: results are due within 12 months of primary completion.

DEFAULT_SAMPLE_PER_CLASS = 3
# Kept small by default because every evaluation run costs real OpenAI
# tokens — six agents, each up to ten reasoning turns, per run.


async def build_ground_truth(
    violations: int = DEFAULT_SAMPLE_PER_CLASS,
    compliant: int = DEFAULT_SAMPLE_PER_CLASS,
) -> list[dict]:
    """
    Builds a labelled sample of trials from the studies table.

    Args:
        violations: How many overdue, unposted COMPLETED trials to take.
        compliant:  How many COMPLETED trials with results to take.

    Returns:
        A list of {"nct_id", "is_violation", "sponsor", "completion_date"}.
    """
    cutoff_year_month = _deadline_cutoff()

    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        overdue = await conn.fetch(
            """
            SELECT nct_id, sponsor, completion_date
            FROM studies
            WHERE status = 'COMPLETED'
              AND results_posted = false
              AND completion_date <> ''
              AND completion_date < $1
            ORDER BY completion_date ASC
            LIMIT $2
            """,
            cutoff_year_month,
            violations,
        )
        # completion_date is stored as TEXT in "YYYY-MM" form, which
        # compares correctly as a string precisely because that format is
        # zero-padded and big-endian. "2019-06" < "2025-10" is true both
        # as text and as a date.

        posted = await conn.fetch(
            """
            SELECT nct_id, sponsor, completion_date
            FROM studies
            WHERE status = 'COMPLETED'
              AND results_posted = true
              AND completion_date <> ''
            ORDER BY completion_date DESC
            LIMIT $1
            """,
            compliant,
        )
    finally:
        await conn.close()

    labelled = [
        {
            "nct_id":          row["nct_id"],
            "is_violation":    True,
            "sponsor":         row["sponsor"],
            "completion_date": row["completion_date"],
        }
        for row in overdue
    ] + [
        {
            "nct_id":          row["nct_id"],
            "is_violation":    False,
            "sponsor":         row["sponsor"],
            "completion_date": row["completion_date"],
        }
        for row in posted
    ]

    logger.info(
        f"Ground truth built | "
        f"violations={len(overdue)} | compliant={len(posted)} | "
        f"cutoff={cutoff_year_month}"
    )

    return labelled


def _deadline_cutoff() -> str:
    """
    Returns the "YYYY-MM" before which a completed trial is overdue.

    A trial that completed more recently than this is still inside its
    12-month window and is not a violation yet, so it belongs in neither
    class and is excluded from the sample.
    """
    now = datetime.now(timezone.utc)

    months = now.year * 12 + (now.month - 1) - POSTING_DEADLINE_MONTHS

    return f"{months // 12:04d}-{months % 12 + 1:02d}"


async def run_evaluation(
    violations: int = DEFAULT_SAMPLE_PER_CLASS,
    compliant: int = DEFAULT_SAMPLE_PER_CLASS,
    dry_run: bool = False,
    blind: bool = False,
) -> dict:
    """
    Builds the labelled sample, runs MOSAIC over it, and scores the result.

    Args:
        violations: Overdue trials to include.
        compliant:  Compliant trials to include.
        dry_run:    Build and return the sample without running any
                    agents. Costs nothing and makes no LLM calls — use
                    it to check the labelling before spending tokens.
        blind:      Withhold the pre-computed violation verdict and the
                    embedded results-posted label from the agents, so
                    they have to verify each trial and apply the
                    12-month rule themselves. Run it alongside the
                    baseline, not instead of it — the pair is what tells
                    you how much of the baseline score was reasoning.

    Returns:
        A report dict carrying the mode that produced it. In dry_run mode
        it contains the ground truth only.
    """
    ground_truth = await build_ground_truth(violations, compliant)

    if not ground_truth:
        logger.error(
            "No labelled trials found — has ingestion/processing been run?"
        )
        return {"error": "empty ground truth"}

    if dry_run:
        return {"dry_run": True, "ground_truth": ground_truth}

    from graph.graph_builder import mosaic_graph
    # Imported here rather than at module top so that --dry-run does not
    # pay for compiling the graph, which constructs six ChatOpenAI
    # clients and therefore requires a valid API key.

    nct_ids = [entry["nct_id"] for entry in ground_truth]

    started = datetime.now(timezone.utc)

    with blind_mode(enabled=blind):
        # blind=False is a no-op, which keeps one code path for both
        # modes instead of branching around the invoke.
        result = await mosaic_graph.ainvoke({
            "task": (
                "For each of these completed trials, determine whether the "
                "sponsor violated FDAAA 801 by failing to post results "
                "within 12 months of primary completion. Report a signal "
                "only for trials you can confirm are in violation."
            ),
            "nct_ids":     nct_ids,
            "max_studies": len(nct_ids),
            "messages":    [],
            "signals":     [],
            "agents_activated": [],
            "error_log":   [],
            "final_brief": "",
            "run_complete": False,
            "run_id":      str(uuid.uuid4()),
        })

    duration = (datetime.now(timezone.utc) - started).total_seconds()
    signals  = result.get("signals", [])

    return {
        "run_id":           result.get("run_id"),
        "blind":            blind,
        # Recorded so a saved report says which mode produced it. Two
        # files of numbers with no mode field are indistinguishable.
        "duration_seconds": round(duration, 2),
        "trials_evaluated": len(nct_ids),
        "signals_found":    len(signals),
        "agents_activated": sorted(set(result.get("agents_activated", []))),
        "errors":           result.get("error_log", []),
        "accuracy":         score_signals(signals, ground_truth),
        "calibration":      score_calibration(signals),
        "by_agent":         score_agent_contribution(signals),
        "ground_truth":     ground_truth,
    }


def _print_report(report: dict) -> None:
    """Prints a human-readable summary of an evaluation report."""
    if report.get("error"):
        print(f"\nEvaluation could not run: {report['error']}\n")
        return

    if report.get("dry_run"):
        print("\nGROUND TRUTH (dry run — no agents were called)")
        print("=" * 68)
        for entry in report["ground_truth"]:
            label = "VIOLATION" if entry["is_violation"] else "compliant"
            print(
                f"  {entry['nct_id']:<14} {label:<10} "
                f"completed {entry['completion_date']:<9} "
                f"{(entry['sponsor'] or '')[:30]}"
            )
        print()
        return

    accuracy = report["accuracy"]

    mode = (
        "BLIND — verdict withheld from the agents"
        if report.get("blind")
        else "baseline — agents could read the verdict from their tools"
    )
    # Printed next to the scores because the two modes answer different
    # questions and their numbers look identical on the page.

    print("\nMOSAIC EVALUATION REPORT")
    print("=" * 68)
    print(f"  mode              {mode}")
    print(f"  run_id            {report['run_id']}")
    print(f"  trials evaluated  {report['trials_evaluated']}")
    print(f"  signals found     {report['signals_found']}")
    print(f"  duration          {report['duration_seconds']}s")
    print(f"  agents activated  {len(report['agents_activated'])}/6")

    print("\n  ACCURACY")
    print(f"    precision       {accuracy['precision']:.2f}   "
          f"(of what we flagged, how much was real)")
    print(f"    recall          {accuracy['recall']:.2f}   "
          f"(of the real violations, how many we caught)")
    print(f"    f1              {accuracy['f1']:.2f}")
    print(f"    tp/fp/fn/tn     {accuracy['true_positives']}/"
          f"{accuracy['false_positives']}/"
          f"{accuracy['false_negatives']}/"
          f"{accuracy['true_negatives']}")

    if accuracy["false_positives"]:
        print("    ⚠ false positives present — these are accusations "
              "against compliant sponsors")

    calibration = report["calibration"]
    print("\n  CONFIDENCE")
    print(f"    mean            {calibration['mean_confidence']:.2f}")
    print(f"    bands           {calibration['bands']}")
    print(f"    need review     {calibration['would_need_review']}")

    print("\n  SIGNALS BY AGENT")
    for agent, count in report["by_agent"].items():
        print(f"    {agent:<24} {count}")

    if report["errors"]:
        print("\n  ERRORS DURING RUN")
        for error in report["errors"]:
            print(f"    - {error}")

    print()


def _build_parser() -> argparse.ArgumentParser:
    """
    Builds the command line parser.

    Separate from main() so the flags can be tested without running an
    evaluation, which costs real OpenAI tokens.

    Returns:
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Run MOSAIC over a labelled trial sample and score it.",
    )
    parser.add_argument(
        "--violations", type=int, default=DEFAULT_SAMPLE_PER_CLASS,
        help="Overdue, unposted completed trials to include.",
    )
    parser.add_argument(
        "--compliant", type=int, default=DEFAULT_SAMPLE_PER_CLASS,
        help="Completed trials WITH results to include. These are how "
             "false positives get measured — do not set this to 0.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the labelled sample and exit. Makes no LLM calls.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the raw report as JSON instead of a summary.",
    )
    parser.add_argument(
        "--blind", action="store_true",
        help="Withhold the pre-computed violation verdict and the "
             "embedded results-posted label from the agents, so they "
             "must verify each trial and apply the 12-month rule "
             "themselves. Run it ALONGSIDE the baseline: the pair is "
             "what shows how much of the baseline score was reasoning "
             "rather than the tool handing over the answer.",
    )

    return parser


def main() -> None:
    """Command line entry point."""
    args = _build_parser().parse_args()

    if args.json:
        route_logs_to_stderr()
        # Nothing but the report may reach stdout in this mode. Log
        # records are not silenced — they move to stderr, so a run that
        # warned still tells you, and `> report.json` stays parseable.

    report = asyncio.run(run_evaluation(
        violations=args.violations,
        compliant=args.compliant,
        dry_run=args.dry_run,
        blind=args.blind,
    ))

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
