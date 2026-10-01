from __future__ import annotations

import pytest

from app.models import Chunk, RetrievalHit
from app.validators import (
    Violation,
    check_asof_line,
    check_citation,
    check_numbers,
    check_performance,
    check_pii,
    check_sentence_count,
    check_word_count,
    is_citation_failure,
    is_suppressing,
    normalise_url,
    split_answer_lines,
    validate_answer,
)
from config import AppConfig

URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
OTHER_URL = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
UNCONFIGURED_URL = "https://example.com/not-a-source"

CHUNK_TEXT = (
    "The expense ratio of the fund is 1.5% plus additional charges. The minimum "
    "lump sum investment is Rs. 1,00,000 and exit load is nil for the direct plan."
)


def make_hit(
    text: str = CHUNK_TEXT,
    *,
    url: str = URL,
    content_as_of: str = "2026-01-15",
) -> RetrievalHit:
    chunk = Chunk(
        source_id="s1",
        source_url=url,
        publisher="Groww (mirroring HDFC MF official data)",
        source_domain="groww.in",
        page_title="HDFC Large Cap Fund - Direct Growth",
        doc_type="scheme_page",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        scheme_category="large_cap",
        plan="direct",
        option="growth",
        topic="expense_ratio",
        section_heading="Fees",
        content_as_of=content_as_of,
        chunk_index=0,
        text=text,
        ingested_at="2026-01-16T00:00:00Z",
        token_count=len(text.split()),
    )
    return RetrievalHit(chunk=chunk, score=0.026, dense_rank=1, lexical_rank=1)


def codes(violations: list[Violation]) -> set[str]:
    return {violation.code for violation in violations}


def good_answer(body: str = "The expense ratio of the fund is 1.5%.", url: str = URL, asof: str = "2026-01-15") -> str:
    return f"{body}\nSource: {url}\nLast updated from sources: {asof}"


# --- line splitting ---------------------------------------------------------------


def test_split_answer_lines() -> None:
    body, source, asof = split_answer_lines(good_answer())
    assert body == "The expense ratio of the fund is 1.5%."
    assert source == URL
    assert asof == "2026-01-15"


def test_source_inside_a_sentence_is_body_text() -> None:
    body, source, _ = split_answer_lines("The page lists Source: values for each field.\nSource: " + URL)
    assert source == URL
    assert "Source: values" in body


# --- citation ---------------------------------------------------------------------


def test_citation_in_allowlist_and_retrieved_is_clean() -> None:
    assert check_citation(good_answer(), {URL}) == []


def test_citation_from_a_non_retrieved_configured_url() -> None:
    violations = check_citation(good_answer(url=OTHER_URL), {URL}, {URL, OTHER_URL})
    assert codes(violations) == {"citation_not_retrieved"}


def test_citation_not_in_the_allowlist_at_all() -> None:
    violations = check_citation(good_answer(url=UNCONFIGURED_URL), {URL}, {URL, OTHER_URL})
    assert codes(violations) == {"citation_not_in_allowlist"}


def test_two_source_lines() -> None:
    text = f"{good_answer()}\nSource: {OTHER_URL}"
    assert codes(check_citation(text, {URL, OTHER_URL})) == {"citation_multiple"}


def test_missing_source_line() -> None:
    text = "The expense ratio is 1.5%.\nLast updated from sources: 2026-01-15"
    assert codes(check_citation(text, {URL})) == {"citation_missing"}


def test_url_normalisation() -> None:
    assert normalise_url(f"{URL}.") == URL
    assert normalise_url(f"{URL},") == URL
    assert normalise_url(f"[groww]({URL})") == URL
    assert normalise_url("https://www.groww.in/mutual-funds/x/") == "https://groww.in/mutual-funds/x"


# --- numbers ----------------------------------------------------------------------


@pytest.mark.parametrize("body", ["The expense ratio is 1.5%.", "Minimum is Rs. 1,00,000.", "The NAV is 12."])
def test_numbers_present_in_the_retrieved_text(body: str) -> None:
    assert check_numbers(body, CHUNK_TEXT, "What is the expense ratio?") == []


def test_number_absent_from_the_retrieved_text() -> None:
    violations = check_numbers("The expense ratio is 7.9%.", CHUNK_TEXT, "What is the expense ratio?")
    assert codes(violations) == {"unsourced_number"}
    assert violations[0].detail == "7.9%"


def test_thousands_separators_compare_numerically() -> None:
    assert check_numbers("It is 100000.", "The minimum is 1,00,000.", "") == []


def test_question_numbers_are_allowed() -> None:
    assert check_numbers("It is 4.5.", CHUNK_TEXT, "What about 4.5 percent?") == []


# --- performance ------------------------------------------------------------------


def test_expense_ratio_answer_with_a_percentage_is_not_flagged() -> None:
    assert check_performance("The expense ratio is 1.5% and the exit load is nil.") == []


def test_return_claim_is_flagged() -> None:
    assert codes(check_performance("The fund returned 32% last year.")) == {"performance"}
    assert codes(check_performance("The 3 year CAGR is healthy.")) == {"performance"}
    assert codes(check_performance("The fund's 1 year return was 9.9%.")) == {"performance"}
    assert codes(check_performance("The fund outperformed its peers with 12% gains.")) == {"performance"}


def test_benchmark_index_name_is_not_a_performance_claim() -> None:
    assert check_performance(
        "The benchmark for the HDFC Small Cap Fund is the BSE 250 SmallCap Total Return Index."
    ) == []
    assert check_performance("It tracks the Nifty 50 Total Return Index.") == []


# --- PII --------------------------------------------------------------------------


def test_pan_is_flagged(real_config: AppConfig) -> None:
    violations = check_pii("Your PAN is ABCDE1234F.", real_config.refusal.pii_patterns)
    assert codes(violations) == {"pii"}


def test_email_and_phone_are_flagged(real_config: AppConfig) -> None:
    assert codes(check_pii("Write to me@example.com.", real_config.refusal.pii_patterns)) == {"pii"}
    assert codes(check_pii("Call 9876543210 now.", real_config.refusal.pii_patterns)) == {"pii"}


def test_clean_answer_has_no_pii_violation(real_config: AppConfig) -> None:
    assert check_pii(good_answer(), real_config.refusal.pii_patterns) == []


# --- sentence and word counts -----------------------------------------------------


def test_sentence_count() -> None:
    assert check_sentence_count("One. Two. Three.", 3) == []
    assert codes(check_sentence_count("One. Two. Three. Four.", 3)) == {"too_many_sentences"}


def test_decimals_and_abbreviations_do_not_inflate_the_count() -> None:
    assert check_sentence_count("The expense ratio is 1.5% e.g. plus charges.", 3) == []


def test_word_count() -> None:
    assert check_word_count(" ".join(["word"] * 60), 60) == []
    assert codes(check_word_count(" ".join(["word"] * 61), 60)) == {"too_long"}


# --- as-of ------------------------------------------------------------------------


def test_asof_matching_the_cited_chunk() -> None:
    assert check_asof_line("2026-01-15", [make_hit()]) == []


def test_asof_mismatch() -> None:
    assert codes(check_asof_line("2024-05-05", [make_hit()])) == {"asof_mismatch"}


def test_asof_unknown_is_allowed_when_a_retrieved_chunk_has_no_date() -> None:
    assert check_asof_line("unknown", [make_hit(content_as_of="")]) == []
    assert check_asof_line("unknown", [make_hit(), make_hit(content_as_of="")]) == []


def test_asof_unknown_is_rejected_when_every_chunk_has_a_date() -> None:
    assert codes(check_asof_line("unknown", [make_hit()])) == {"asof_mismatch"}
    assert codes(check_asof_line("unknown", [make_hit(), make_hit()])) == {"asof_mismatch"}


# --- end-to-end -------------------------------------------------------------------


def test_validate_answer_is_clean(real_config: AppConfig) -> None:
    hits = [make_hit()]
    assert validate_answer(good_answer(), hits, "What is the expense ratio?", real_config) == []


def test_context_insufficiency_is_reported_and_not_treated_as_an_answer(real_config: AppConfig) -> None:
    hits = [make_hit()]
    text = good_answer(body="I don't have that in my sources.")
    assert "context_insufficient" in codes(validate_answer(text, hits, "What is the objective?", real_config))
    assert "context_insufficient" not in codes(validate_answer(good_answer(), hits, "q", real_config))


def test_validate_answer_reports_several_violations(real_config: AppConfig) -> None:
    hits = [make_hit()]
    text = "A. B. C. D. The PAN is ABCDE1234F.\nSource: " + UNCONFIGURED_URL + "\nLast updated from sources: 1999-01-01"
    found = codes(validate_answer(text, hits, "What is the expense ratio?", real_config))
    assert {"citation_not_in_allowlist", "too_many_sentences", "pii", "asof_mismatch"} <= found


def test_validate_answer_flags_an_unsourced_number(real_config: AppConfig) -> None:
    hits = [make_hit()]
    text = good_answer(body="The expense ratio is 9.9%.")
    assert "unsourced_number" in codes(validate_answer(text, hits, "What is the expense ratio?", real_config))


# --- suppression policy (§9.6.8) --------------------------------------------------


def test_too_long_alone_does_not_suppress() -> None:
    assert not is_suppressing([Violation("too_long", "61 words", "61")])


def test_too_long_with_too_many_sentences_suppresses() -> None:
    assert is_suppressing([Violation("too_long", "61 words", "61"), Violation("too_many_sentences", "4", "4")])


def test_unsourced_number_suppresses_immediately() -> None:
    assert is_suppressing([Violation("unsourced_number", "9.9%", "9.9%")])


def test_citation_failures_are_classified_for_a_single_retry() -> None:
    assert is_citation_failure([Violation("citation_not_retrieved", "u", "u")])
    assert not is_citation_failure([Violation("too_long", "61 words", "61")])
