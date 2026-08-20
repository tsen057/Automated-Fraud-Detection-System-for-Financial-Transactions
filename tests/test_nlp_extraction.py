from src.nlp_extraction import analyze_report, analyze_reports, extract_entities


SAMPLE_REPORT = (
    "On 2024-03-15 a transfer of $12,500.00 was flagged as a suspicious "
    "unusual pattern involving a shell company in a high-risk jurisdiction. "
    "Reference invoice #INV-2291."
)

CLEAN_REPORT = (
    "Routine quarterly review of account activity. No unusual pattern "
    "observed. All transactions verified against invoice #INV-1001."
)


class TestExtractEntities:
    def test_extracts_amount(self):
        entities = extract_entities(SAMPLE_REPORT)
        assert "$12,500.00" in entities["amounts"]

    def test_extracts_date(self):
        entities = extract_entities(SAMPLE_REPORT)
        assert "2024-03-15" in entities["dates"]

    def test_extracts_reference(self):
        entities = extract_entities(SAMPLE_REPORT)
        assert any("2291" in ref for ref in entities["references"])

    def test_handles_text_with_no_entities(self):
        entities = extract_entities("Nothing to see here.")
        assert entities["amounts"] == []
        assert entities["dates"] == []


class TestAnalyzeReport:
    def test_flags_risk_terms_in_suspicious_report(self):
        insight = analyze_report(SAMPLE_REPORT)
        assert insight.risk_score > 0
        assert any("suspicious" in term for term in insight.matched_risk_terms)

    def test_low_risk_score_for_clean_report(self):
        insight = analyze_report(CLEAN_REPORT)
        suspicious_insight = analyze_report(SAMPLE_REPORT)
        assert insight.risk_score < suspicious_insight.risk_score


class TestAnalyzeReports:
    def test_batch_matches_single_report_entities(self):
        results = analyze_reports([SAMPLE_REPORT, CLEAN_REPORT])
        assert len(results) == 2
        assert results[0].amounts == extract_entities(SAMPLE_REPORT)["amounts"]

    def test_ranks_suspicious_report_higher(self):
        results = analyze_reports([CLEAN_REPORT, SAMPLE_REPORT])
        assert results[1].risk_score > results[0].risk_score

