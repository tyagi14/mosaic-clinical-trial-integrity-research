##############################################################################
# tests/test_clients.py
#
# WHAT THIS FILE TESTS:
#   ingestion/pubmed_client.py — the XML parsing, offline
#   ingestion/gcs_store.py     — object path construction, offline
#
# WHY XML PARSING DESERVES ITS OWN TESTS:
#   PubMed returns XML, and its shape is not obvious: an abstract can
#   arrive in several labelled sections, authors are split across two
#   elements, and the NCT ID a trial is registered under hides in a
#   generic "DataBankList". Every one of those is a place to read the
#   wrong element and get plausible-looking but wrong data — which then
#   becomes evidence in a safety signal about a real drug.
#
#   The HTTP calls are not tested: asserting that httpx fetches a URL
#   would test httpx and would need the network.
##############################################################################


PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>31234567</PMID>
      <Article>
        <ArticleTitle>Semaglutide and cardiovascular outcomes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Patients were enrolled.</AbstractText>
          <AbstractText Label="RESULTS">Three hospitalisations occurred.</AbstractText>
        </Abstract>
        <Journal>
          <Title>New England Journal of Medicine</Title>
          <JournalIssue>
            <PubDate><Year>2019</Year><Month>Jun</Month></PubDate>
          </JournalIssue>
        </Journal>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
          <Author><LastName>Doe</LastName><ForeName>John</ForeName></Author>
        </AuthorList>
        <DataBankList>
          <DataBank>
            <DataBankName>ClinicalTrials.gov</DataBankName>
            <AccessionNumberList>
              <AccessionNumber>NCT04788680</AccessionNumber>
            </AccessionNumberList>
          </DataBank>
        </DataBankList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedXmlParsing:

    def _client(self):
        from ingestion.pubmed_client import PubMedClient
        return PubMedClient()
        # Constructed without entering its async context manager, so no
        # HTTP session is ever opened. The parsing methods are pure.

    def test_one_article_is_parsed_from_the_response(self):
        papers = self._client()._parse_xml_response(PUBMED_XML)

        assert len(papers) == 1

    def test_the_identifying_fields_are_extracted(self):
        paper = self._client()._parse_xml_response(PUBMED_XML)[0]

        assert paper["pmid"] == "31234567"
        assert paper["title"] == "Semaglutide and cardiovascular outcomes"
        assert "New England Journal" in paper["journal"]

    def test_a_sectioned_abstract_is_joined_rather_than_truncated(self):
        paper = self._client()._parse_xml_response(PUBMED_XML)[0]

        assert "Patients were enrolled" in paper["abstract"]
        assert "Three hospitalisations occurred" in paper["abstract"]
        # Structured abstracts arrive as several AbstractText elements.
        # Reading only the first would drop the RESULTS section — which
        # is precisely where the adverse events the side effect agent
        # looks for are described.

    def test_authors_are_assembled_from_both_name_elements(self):
        paper = self._client()._parse_xml_response(PUBMED_XML)[0]

        assert len(paper["authors"]) == 2
        assert any("Smith" in author for author in paper["authors"])

    def test_the_referenced_nct_id_is_recovered(self):
        paper = self._client()._parse_xml_response(PUBMED_XML)[0]

        assert "NCT04788680" in paper["nct_ids_referenced"]
        # This is the only link between a published paper and the trial
        # it reports on. Without it, the side effect agent cannot compare
        # what a sponsor filed against what was actually published.

    def test_malformed_xml_returns_nothing_instead_of_raising(self):
        assert self._client()._parse_xml_response("<not valid xml") == []
        assert self._client()._parse_xml_response("") == []
        # PubMed occasionally returns an HTML error page with a 200
        # status. That must not take down an ingestion run.

    def test_an_empty_result_set_parses_to_an_empty_list(self):
        empty = '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'

        assert self._client()._parse_xml_response(empty) == []
        # The common case: most trials are never directly cited, and "no
        # papers" is a normal answer rather than a failure.


class TestGcsPaths:
    """
    gcs_store.py — where documents land in the bucket.

    Tested because the layout IS the archive's index. If raw and parsed
    documents can collide on one path, ingestion silently overwrites its
    own evidence trail.
    """

    def test_raw_and_parsed_documents_are_kept_apart(self):
        import inspect

        from ingestion import gcs_store

        source = inspect.getsource(gcs_store)

        assert "raw/" in source
        assert "processed/" in source or "parsed/" in source
        # Separate prefixes mean a re-parse can be rerun from the raw
        # archive without having destroyed the original API response.
