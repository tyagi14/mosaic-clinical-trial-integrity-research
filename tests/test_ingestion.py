##############################################################################
# tests/test_ingestion.py
#
# WHAT THIS FILE TESTS:
#   ingestion/document_parser.py    raw API JSON  → ParsedStudy / ParsedPaper
#   ingestion/clinical_trials_client.py  our arguments → API query parameters
#
# WHY THESE TWO:
#   They are the system's boundary with the outside world. Everything
#   downstream — chunking, embedding, every agent's reasoning — trusts
#   whatever the parser produced. A field the parser reads from the wrong
#   place does not raise; it quietly makes every later conclusion wrong.
#
#   The HTTP calls themselves are not tested here. Asserting that
#   requests.get works would test the library, not our code, and would
#   need the network.
##############################################################################


from ingestion.document_parser import DocumentParser, ParsedStudy


class TestParseStudy:
    """DocumentParser.parse_study — raw v2 JSON to a clean ParsedStudy."""

    def test_extracts_every_field_from_deep_nesting(self, raw_study):
        study = DocumentParser().parse_study(raw=raw_study)

        assert isinstance(study, ParsedStudy)
        assert study.nct_id == "NCT04788680"
        assert study.sponsor == "Novo Nordisk A/S"
        assert study.phase == "PHASE3"
        assert study.status == "COMPLETED"
        assert study.enrollment == 1200
        assert study.conditions == ["Type 2 Diabetes", "Obesity"]
        assert study.primary_outcome == "Change in HbA1c at week 52"
        assert len(study.secondary_outcomes) == 2

    def test_prefers_official_title_over_brief_title(self, raw_study):
        study = DocumentParser().parse_study(raw=raw_study)

        assert study.title == "A Study of Semaglutide in Type 2 Diabetes"
        # The official title is the legally filed one. The brief title is
        # a marketing-style summary, so it is only a fallback.

    def test_falls_back_to_brief_title_when_official_is_absent(self, raw_study):
        raw = {
            **raw_study,
            "protocolSection": {
                **raw_study["protocolSection"],
                "identificationModule": {
                    "nctId": "NCT04788680",
                    "briefTitle": "A Short Title",
                },
            },
        }

        study = DocumentParser().parse_study(raw=raw)

        assert study.title == "A Short Title"

    def test_missing_results_section_means_results_not_posted(self, raw_study):
        study = DocumentParser().parse_study(raw=raw_study)

        assert study.results_posted is False
        # THE SINGLE MOST IMPORTANT ASSERTION IN THIS FILE.
        #
        # results_posted drives the whole missing-results signal, which is
        # the system's core accusation. If this field were ever read the
        # wrong way round, MOSAIC would accuse compliant sponsors of
        # breaking federal law and ignore the ones actually doing it.

    def test_present_results_section_means_results_posted(
        self, raw_study_with_results
    ):
        study = DocumentParser().parse_study(raw=raw_study_with_results)

        assert study.results_posted is True

    def test_blank_intervention_names_are_dropped(self, raw_study):
        study = DocumentParser().parse_study(raw=raw_study)

        assert study.interventions == ["Semaglutide", "Placebo"]
        assert "" not in study.interventions

    def test_study_without_nct_id_is_rejected(self):
        raw = {"protocolSection": {"identificationModule": {}}}

        assert DocumentParser().parse_study(raw=raw) is None
        # No NCT ID means the record cannot be referenced, linked to a
        # signal, or de-duplicated. It is useless, so it is dropped here
        # rather than halfway down the pipeline.

    def test_empty_and_malformed_input_returns_none_instead_of_raising(self):
        parser = DocumentParser()

        assert parser.parse_study(raw={}) is None
        assert parser.parse_study(raw={"protocolSection": None}) is None
        # One bad record in a 1000-study ingestion run must not abort the
        # other 999, so the parser reports failure by returning None.

    def test_completion_date_prefers_primary_completion(self, raw_study):
        study = DocumentParser().parse_study(raw=raw_study)

        assert study.completion_date == "2019-06"
        # The FDAAA 801 12-month clock runs from PRIMARY completion
        # (2019-06), not overall completion (2019-08). Reading the wrong
        # one would shift every overdue calculation by two months.

    def test_falls_back_to_overall_completion_date(self, raw_study):
        status = {
            **raw_study["protocolSection"]["statusModule"],
        }
        status.pop("primaryCompletionDateStruct")
        raw = {
            **raw_study,
            "protocolSection": {
                **raw_study["protocolSection"],
                "statusModule": status,
            },
        }

        study = DocumentParser().parse_study(raw=raw)

        assert study.completion_date == "2019-08"


class TestParseStudies:
    """DocumentParser.parse_studies — the batch path."""

    def test_good_records_survive_a_bad_one(self, raw_study):
        bad = {"protocolSection": {"identificationModule": {}}}

        studies = DocumentParser().parse_studies(raw_studies=[raw_study, bad, raw_study])

        assert len(studies) == 2
        # The malformed record is skipped; the two valid ones come
        # through. This is the behaviour a long ingestion run depends on.

    def test_empty_input_returns_empty_list(self):
        assert DocumentParser().parse_studies(raw_studies=[]) == []


class TestSearchParameterBuilding:
    """
    ClinicalTrialsClient._build_search_params — our arguments mapped onto
    the API's parameter names.

    Worth testing because the mapping is not guessable: a condition
    becomes "query.cond", and a list of statuses becomes a single
    pipe-joined string. A wrong name here does not error — the API simply
    ignores the filter and returns unfiltered results, which looks like
    success.
    """

    def _client(self):
        from ingestion.clinical_trials_client import ClinicalTrialsClient
        return ClinicalTrialsClient()
        # Constructed without entering the async context manager, so no
        # HTTP session is opened. _build_search_params is pure.

    def test_maps_arguments_to_api_parameter_names(self):
        params = self._client()._build_search_params(
            condition="diabetes",
            intervention="semaglutide",
            sponsor="Novo Nordisk",
            status=None,
            page_token=None,
        )

        assert params["query.cond"] == "diabetes"
        assert params["query.intr"] == "semaglutide"
        assert params["query.spons"] == "Novo Nordisk"
        assert params["format"] == "json"

    def test_multiple_statuses_are_pipe_joined(self):
        params = self._client()._build_search_params(
            condition=None, intervention=None, sponsor=None,
            status=["COMPLETED", "TERMINATED"], page_token=None,
        )

        assert params["filter.overallStatus"] == "COMPLETED|TERMINATED"
        # Pipe, not comma. A comma is read as part of one status value
        # and silently matches nothing.

    def test_absent_filters_are_omitted_entirely(self):
        params = self._client()._build_search_params(
            condition=None, intervention=None, sponsor=None,
            status=None, page_token=None,
        )

        assert "query.cond" not in params
        assert "filter.overallStatus" not in params
        # Sending query.cond="" is not the same as omitting it — an empty
        # filter can match zero studies instead of all of them.

    def test_page_token_is_passed_through_for_pagination(self):
        params = self._client()._build_search_params(
            condition=None, intervention=None, sponsor=None,
            status=None, page_token="abc123",
        )

        assert params["pageToken"] == "abc123"
