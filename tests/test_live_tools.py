##############################################################################
# tests/test_live_tools.py
#
# WHAT THIS FILE TESTS:
#   tools/clinical_tools.py — the four ClinicalTrials.gov tools
#   tools/pubmed_tools.py   — the three PubMed tools
#
# HOW, WITHOUT THE NETWORK:
#   Each module imported its API client class by name, so a test can
#   replace that name with a fake client. The real tool body then runs:
#   it opens the (fake) client, parses the response with the REAL
#   DocumentParser, and serialises the result to JSON.
#
# WHY THESE TOOLS DESERVE CAREFUL TESTS:
#   They are the only part of MOSAIC that sees live data. Everything else
#   reasons over the database snapshot, which can be months stale. When
#   an agent is about to accuse a sponsor, these tools are what confirm
#   the trial is still non-compliant TODAY — and check_results_posted in
#   particular computes the overdue figure that ends up in the signal.
##############################################################################


import json

import pytest


# ── FAKE CLIENTS ──────────────────────────────────────────────────────

class FakeClinicalTrialsClient:
    """
    Stands in for ClinicalTrialsClient, including its async context
    manager — the tools all use "async with ClinicalTrialsClient()".
    """

    study = None
    studies = []
    raise_on_fetch = None

    def __init__(self):
        self.fetched = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def fetch_study(self, nct_id):
        if type(self).raise_on_fetch:
            raise type(self).raise_on_fetch
        self.fetched.append(nct_id)
        return type(self).study

    async def search_studies(self, **kwargs):
        return type(self).studies


class FakePubMedClient:
    papers = []

    def __init__(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def fetch_papers_for_trial(self, nct_id, max_results=50):
        return type(self).papers

    async def _search_paper_ids(self, nct_id, max_results):
        return ["31234567"] if type(self).papers else []

    async def _fetch_paper_details(self, paper_ids):
        return type(self).papers


def raw_study(**overrides):
    """A raw ClinicalTrials.gov v2 record, overdue and unposted."""
    base = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00011531",
                "officialTitle": "A Veterans Affairs Trial",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "1998-01"},
                "primaryCompletionDateStruct": {"date": "2001-12"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "US Department of Veterans Affairs"},
            },
            "conditionsModule": {"conditions": ["Hypertension"]},
            "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 120}},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Blood pressure at 12 months"}],
            },
            "armsInterventionsModule": {"interventions": [{"name": "Drug A"}]},
        },
    }
    base["protocolSection"].update(overrides.pop("protocolSection", {}))
    base.update(overrides)
    return base


def raw_paper(**overrides):
    base = {
        "pmid": "31234567",
        "title": "Outcomes of a Veterans Affairs trial",
        "abstract": "Three patients were hospitalised with serious adverse events.",
        "journal": "NEJM",
        "pub_date": "2005",
        "authors": ["Smith J"],
        "nct_ids_referenced": ["NCT00011531"],
    }
    return {**base, **overrides}


@pytest.fixture
def clinical(monkeypatch):
    import tools.clinical_tools as module

    FakeClinicalTrialsClient.study = raw_study()
    FakeClinicalTrialsClient.studies = []
    FakeClinicalTrialsClient.raise_on_fetch = None
    monkeypatch.setattr(module, "ClinicalTrialsClient", FakeClinicalTrialsClient)
    return module


@pytest.fixture
def pubmed(monkeypatch):
    import tools.pubmed_tools as module

    FakePubMedClient.papers = [raw_paper()]
    monkeypatch.setattr(module, "PubMedClient", FakePubMedClient)
    return module


# ── CLINICAL TRIALS TOOLS ─────────────────────────────────────────────

class TestFetchStudyDetails:

    def test_returns_the_parsed_study_as_json(self, clinical):
        output = clinical.fetch_study_details.invoke({"nct_id": "NCT00011531"})

        parsed = json.loads(output)
        assert parsed["found"] is True
        assert parsed["nct_id"] == "NCT00011531"
        assert parsed["study"]["sponsor"] == "US Department of Veterans Affairs"
        assert parsed["study"]["status"] == "COMPLETED"
        assert parsed["study"]["results_posted"] is False
        # The full parsed study is nested under "study", alongside a
        # "found" flag — so the agent can tell "no such trial" apart
        # from "a trial with empty fields".

    def test_a_missing_study_is_reported_not_raised(self, clinical):
        FakeClinicalTrialsClient.study = None

        parsed = json.loads(
            clinical.fetch_study_details.invoke({"nct_id": "NCT99999999"})
        )

        assert parsed.get("found") is False or "error" in parsed
        # An NCT ID that does not exist is a normal outcome — models do
        # occasionally invent plausible-looking identifiers.

    def test_a_network_failure_becomes_an_error_payload(self, clinical):
        FakeClinicalTrialsClient.raise_on_fetch = ConnectionError("API down")

        parsed = json.loads(
            clinical.fetch_study_details.invoke({"nct_id": "NCT00011531"})
        )

        assert "error" in parsed
        assert "API down" in parsed["error"]
        # ClinicalTrials.gov being down must cost one wasted reasoning
        # turn, not the whole agent node.


class TestCheckResultsPosted:
    """
    The compliance check. This tool produces the overdue figure that goes
    into the signal and, from there, into the brief.
    """

    def test_an_overdue_unposted_trial_is_flagged_as_a_violation(self, clinical):
        parsed = json.loads(
            clinical.check_results_posted.invoke({"nct_id": "NCT00011531"})
        )

        assert parsed["found"] is True
        assert parsed["results_posted"] is False
        assert parsed["status"] == "COMPLETED"
        assert parsed["months_overdue"] > 12
        assert parsed["is_violation"] is True
        # Completed 2001-12 with no results. The 12-month FDAAA deadline
        # passed long ago, so this is the shape of a real finding.

    def test_a_trial_with_results_is_not_a_violation(self, clinical):
        FakeClinicalTrialsClient.study = raw_study(
            resultsSection={"baselineCharacteristicsModule": {}},
        )

        parsed = json.loads(
            clinical.check_results_posted.invoke({"nct_id": "NCT00011531"})
        )

        assert parsed["results_posted"] is True
        assert parsed.get("is_violation") is not True
        # No overdue figures and no violation flag are emitted at all
        # once results are posted: there is nothing to be overdue about,
        # and an is_violation=False key would still read as a verdict
        # the tool was asked to reach.
        # The false-positive guard. A sponsor who complied must never be
        # reported as a violation, whatever the dates say.

    def test_a_missing_study_does_not_claim_a_violation(self, clinical):
        FakeClinicalTrialsClient.study = None

        parsed = json.loads(
            clinical.check_results_posted.invoke({"nct_id": "NCT99999999"})
        )

        assert parsed["found"] is False
        assert parsed.get("is_violation") is not True
        # Absence of evidence is not evidence of a violation.


class TestSearchStudiesByCondition:

    def test_returns_matching_studies(self, clinical):
        FakeClinicalTrialsClient.studies = [raw_study()]

        parsed = json.loads(clinical.search_studies_by_condition.invoke(
            {"condition": "hypertension", "max_results": 5}
        ))

        assert parsed["count"] == 1
        assert parsed["studies"][0]["nct_id"] == "NCT00011531"

    def test_no_matches_is_an_explanation_not_an_error(self, clinical):
        FakeClinicalTrialsClient.studies = []

        parsed = json.loads(clinical.search_studies_by_condition.invoke(
            {"condition": "a condition nobody studies"}
        ))

        assert parsed.get("count", 0) == 0
        assert "error" not in parsed


class TestGetStudyAmendments:
    """
    Amendments are the difference between a disclosed delay and a silent
    one — the entire basis of the timeline agent's signal.
    """

    def test_returns_amendment_information_for_a_study(self, clinical):
        parsed = json.loads(
            clinical.get_study_amendments.invoke({"nct_id": "NCT00011531"})
        )

        assert parsed["nct_id"] == "NCT00011531"
        assert "amendments" in parsed or "amendment_count" in parsed

    def test_a_missing_study_is_handled(self, clinical):
        FakeClinicalTrialsClient.study = None

        parsed = json.loads(
            clinical.get_study_amendments.invoke({"nct_id": "NCT99999999"})
        )

        assert parsed.get("found") is False or "error" in parsed


# ── PUBMED TOOLS ──────────────────────────────────────────────────────

class TestFetchPapersForTrial:

    def test_returns_papers_that_reference_the_trial(self, pubmed):
        parsed = json.loads(
            pubmed.fetch_papers_for_trial.invoke({"nct_id": "NCT00011531"})
        )

        assert parsed["count"] == 1
        assert parsed["papers"][0]["pmid"] == "31234567"

    def test_a_trial_with_no_papers_is_a_normal_answer(self, pubmed):
        FakePubMedClient.papers = []

        parsed = json.loads(
            pubmed.fetch_papers_for_trial.invoke({"nct_id": "NCT00011531"})
        )

        assert parsed.get("count", 0) == 0
        assert "error" not in parsed
        # Most trials are never directly cited. "No published research"
        # is information, not a failure.


class TestSearchPubmedByQuery:

    def test_returns_papers_for_a_free_text_query(self, pubmed):
        parsed = json.loads(pubmed.search_pubmed_by_query.invoke(
            {"query": "semaglutide adverse events"}
        ))

        assert parsed["count"] == 1

    def test_no_results_returns_an_empty_payload(self, pubmed):
        FakePubMedClient.papers = []

        parsed = json.loads(pubmed.search_pubmed_by_query.invoke(
            {"query": "a topic nobody has published on"}
        ))

        assert parsed.get("count", 0) == 0


class TestCompareFilingVsPapers:
    """
    The safety-gap comparison: what the sponsor filed against what was
    actually published. The highest-stakes tool in the system — its
    output becomes a public claim about a drug's safety reporting.
    """

    def test_compares_the_filing_against_the_published_papers(self, pubmed):
        parsed = json.loads(pubmed.compare_filing_vs_papers.invoke({
            "nct_id": "NCT00011531",
            "filing_summary": "Filing reports no serious adverse events.",
        }))

        assert parsed["nct_id"] == "NCT00011531"
        assert parsed["filing_summary"].startswith("Filing reports")
        assert "hospitalised" in json.dumps(parsed)
        # Both sides come back in one payload — what the sponsor filed
        # and what the paper actually said. The tool deliberately does
        # NOT decide whether that is a discrepancy; the agent does, under
        # its reasoning rules, so the judgment stays auditable.

    def test_no_papers_means_no_comparison_is_claimed(self, pubmed):
        FakePubMedClient.papers = []

        parsed = json.loads(pubmed.compare_filing_vs_papers.invoke({
            "nct_id": "NCT00011531",
            "filing_summary": "Filing reports no serious adverse events.",
        }))

        assert parsed.get("paper_count", 0) == 0 or not parsed.get("papers")
        # With nothing published there is nothing to contradict the
        # filing. The tool must not imply a discrepancy it cannot see.
