"""NLP-based extraction of structured risk insights from unstructured
compliance report text.

Combines:
  - Rule-based entity extraction (amounts, dates, account/invoice
    references) via regex — fast and precise for well-formatted fields.
  - TF-IDF based risk-term scoring — surfaces which known risk phrases
    appear in a report and how salient they are relative to boilerplate
    text, so an analyst can prioritize review instead of reading every
    report in full.

This avoids a heavyweight NLP model dependency (e.g. spaCy/transformers),
keeping the pipeline fast to train, test, and run in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import RISK_KEYWORDS

AMOUNT_RE = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
ACCOUNT_RE = re.compile(r"\b(?:acct|account|invoice)[\s#:]*([A-Z0-9\-]{4,})\b", re.I)


@dataclass
class ComplianceInsight:
    """Structured result extracted from one compliance report."""

    amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    matched_risk_terms: list[str] = field(default_factory=list)
    risk_score: float = 0.0


def extract_entities(text: str) -> dict[str, list[str]]:
    """Rule-based extraction of amounts, dates, and account/invoice references."""
    return {
        "amounts": AMOUNT_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "references": ACCOUNT_RE.findall(text),
    }


def score_risk_terms(
    documents: list[str], risk_keywords: list[str] | None = None
) -> list[dict[str, float]]:
    """Score each document's risk-keyword salience using TF-IDF.

    For each document, returns the TF-IDF weight of every risk keyword that
    appears in it (keywords absent from the document are omitted). Weights
    are comparable within a single corpus, so higher generally means the
    term is more central to that particular report rather than boilerplate.
    """
    risk_keywords = risk_keywords or RISK_KEYWORDS

    vectorizer = TfidfVectorizer(
        vocabulary=[kw.lower() for kw in risk_keywords],
        ngram_range=(1, 3),
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    results = []
    for row in tfidf_matrix.toarray():
        doc_scores = {
            terms[i]: float(row[i]) for i in range(len(terms)) if row[i] > 0
        }
        results.append(doc_scores)
    return results


def analyze_report(text: str, risk_keywords: list[str] | None = None) -> ComplianceInsight:
    """Run entity extraction + risk scoring on a single compliance report."""
    entities = extract_entities(text)
    term_scores = score_risk_terms([text], risk_keywords=risk_keywords)[0]

    matched_terms = sorted(term_scores, key=term_scores.get, reverse=True)
    overall_score = sum(term_scores.values())

    return ComplianceInsight(
        amounts=entities["amounts"],
        dates=entities["dates"],
        references=entities["references"],
        matched_risk_terms=matched_terms,
        risk_score=round(overall_score, 4),
    )


def analyze_reports(
    texts: list[str], risk_keywords: list[str] | None = None
) -> list[ComplianceInsight]:
    """Batch version of analyze_report — scores risk terms across the whole
    corpus at once so TF-IDF weighting reflects relative salience across
    reports, not just within a single document.
    """
    risk_keywords = risk_keywords or RISK_KEYWORDS
    term_scores_list = score_risk_terms(texts, risk_keywords=risk_keywords)

    insights = []
    for text, term_scores in zip(texts, term_scores_list):
        entities = extract_entities(text)
        matched_terms = sorted(term_scores, key=term_scores.get, reverse=True)
        insights.append(
            ComplianceInsight(
                amounts=entities["amounts"],
                dates=entities["dates"],
                references=entities["references"],
                matched_risk_terms=matched_terms,
                risk_score=round(sum(term_scores.values()), 4),
            )
        )
    return insights

