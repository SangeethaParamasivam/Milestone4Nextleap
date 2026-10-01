from __future__ import annotations

import json
import logging

import pytest

from app.guard import (
    REFUSAL_MESSAGES,
    REFUSAL_STATUSES,
    SAFE_ADVICE_TEXT,
    SAFE_OUT_OF_CORPUS_TEXT,
    SAFE_PII_TEXT,
    SAFE_PERFORMANCE_TEXT,
    classify,
    normalise_question,
    refusal_answer,
    scan_pii,
)
from app.models import Decision
from config import REPO_ROOT, AppConfig, load_config
from tests.conftest import CONFIG_TEXT

GOLDEN_FILE = REPO_ROOT / "data" / "golden_questions.jsonl"

FROZEN_MESSAGES: dict[str, str] = {
    "ADVICE": (
        "I answer only published facts about these schemes and can't recommend, compare or rate "
        "funds or suggest a portfolio."
    ),
    "PERFORMANCE": (
        "I don't state or compute returns. Performance figures for each scheme are in its official "
        "factsheet, linked below."
    ),
    "PII": (
        "Please don't share personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). "
        "This message has not been stored. Delete it from your chat."
    ),
    "OUT_OF_CORPUS": (
        "I don't have that in my sources. I answer only about: HDFC Large Cap, HDFC Equity (Flexi Cap), "
        "HDFC ELSS Tax Saver, HDFC Small Cap and HDFC Balanced Advantage funds, and general official "
        "SEBI/AMFI information."
    ),
}

PRD_REFUSAL_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Should I buy HDFC Large Cap Fund now?", "ADVICE"),
    ("Which of these five funds is best for me?", "ADVICE"),
    ("How much did HDFC Flexi Cap return last year compared to a large cap?", "PERFORMANCE"),
    ("Is HDFC Small Cap Fund good for my retirement?", "ADVICE"),
    ("My PAN is ABCDE1234F, can you check my returns?", "PII"),
    ("What is my account number's IFSC?", "PII"),
    ("Give me an OTP for my Groww login.", "PII"),
    ("Is now a good time to exit HDFC ELSS?", "ADVICE"),
)

PII_QUESTIONS: tuple[str, ...] = (
    "My PAN is ABCDE1234F.",
    "My aadhaar is 1234 5678 9012.",
    "My aadhaar is 123456789012.",
    "Contact me on 9876543210.",
    "My phone number is +91 9876543210.",
    "Email me at priya.sharma@example.com.",
    "my IFSC is HDFC0001234",
    "OTP 482913",
    "account number 0012345678",
    "What is my CVV?",
    "Send me my demat account details.",
    "The verification code 553311 was sent to me.",
)

REASON_CODE_CASES: tuple[tuple[str, str, str], ...] = (
    ("Should I buy HDFC Large Cap Fund now?", "ADVICE", "advice:should i"),
    ("How much did HDFC Flexi Cap return last year?", "PERFORMANCE", "performance:return"),
    ("What is the expense ratio of Parag Parflex Fund?", "OUT_OF_CORPUS", "out_of_corpus:parag parflex"),
    ("What is the expense ratio of HDFC Large Cap Fund?", "FACTUAL", "factual"),
)

BOUNDARY_CONFIG = CONFIG_TEXT.replace(
    'advice: ["should i", "recommend"]', 'advice: ["ter"]'
).replace('performance: ["return", "cagr"]', 'performance: ["nav"]')


def _golden_questions() -> tuple[tuple[str, str, bool], ...]:
    rows = [
        json.loads(line)
        for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple((row["id"], row["question"], bool(row["expect_refusal"])) for row in rows)


GOLDEN_QUESTIONS = _golden_questions()


@pytest.mark.parametrize(
    "question,expected_intent",
    PRD_REFUSAL_QUESTIONS,
    ids=[question for question, _ in PRD_REFUSAL_QUESTIONS],
)
def test_every_prd_refusal_question_is_refused(
    real_config: AppConfig, question: str, expected_intent: str
) -> None:
    decision = classify(question, real_config)
    assert decision.intent != "FACTUAL"
    assert decision.intent == expected_intent


@pytest.mark.parametrize(
    "golden_id,question,expect_refusal",
    GOLDEN_QUESTIONS,
    ids=[golden_id for golden_id, _, _ in GOLDEN_QUESTIONS],
)
def test_golden_set_is_answered_or_refused_as_declared(
    real_config: AppConfig, golden_id: str, question: str, expect_refusal: bool
) -> None:
    decision = classify(question, real_config)
    if expect_refusal:
        assert decision.intent == "OUT_OF_CORPUS"
    else:
        assert decision.intent == "FACTUAL", f"{golden_id} was wrongly refused"


@pytest.mark.parametrize("question", PII_QUESTIONS)
def test_pii_variants_are_refused(real_config: AppConfig, question: str) -> None:
    decision = classify(question, real_config)
    assert decision.intent == "PII"
    assert decision.reason_code.startswith("pii:")


def test_pii_keyword_patterns_are_case_insensitive(real_config: AppConfig) -> None:
    assert classify("my pan is abcde1234f", real_config).intent == "PII"
    assert classify("my cvv is 123", real_config).intent == "PII"


def test_pii_beats_advice(real_config: AppConfig) -> None:
    decision = classify("Should I buy HDFC Large Cap Fund? My PAN is ABCDE1234F", real_config)
    assert decision.intent == "PII"
    assert decision.reason_code == "pii:0"


def test_advice_beats_performance(real_config: AppConfig) -> None:
    decision = classify("Which is best and how much did it return?", real_config)
    assert decision.intent == "ADVICE"


def test_foreign_scheme_without_an_in_scope_alias_is_out_of_corpus(real_config: AppConfig) -> None:
    decision = classify("What is the expense ratio of Parag Parflex Fund?", real_config)
    assert decision.intent == "OUT_OF_CORPUS"


def test_in_scheme_question_is_not_flagged(real_config: AppConfig) -> None:
    decision = classify("What is the expense ratio of HDFC Large Cap Fund?", real_config)
    assert decision.intent == "FACTUAL"
    assert decision.educational_url is None


def test_foreign_scheme_named_alongside_an_in_scope_scheme_is_not_flagged(
    real_config: AppConfig,
) -> None:
    decision = classify(
        "What is the expense ratio of HDFC Large Cap Fund, and also of Parag Parflex Fund?",
        real_config,
    )
    assert decision.intent == "FACTUAL"


@pytest.mark.parametrize(
    "question,intent,reason_code",
    REASON_CODE_CASES,
    ids=[question for question, _, _ in REASON_CODE_CASES],
)
def test_reason_codes_name_the_matched_keyword(
    real_config: AppConfig, question: str, intent: str, reason_code: str
) -> None:
    decision = classify(question, real_config)
    assert decision.intent == intent
    assert decision.reason_code == reason_code


def test_pii_reason_code_carries_no_matched_text(real_config: AppConfig) -> None:
    decision = classify("My PAN is ABCDE1234F, can you check my returns?", real_config)
    assert decision.reason_code == "pii:0"
    assert "ABCDE1234F" not in decision.reason_code


def test_short_phrase_does_not_match_inside_a_word(write_config) -> None:
    cfg = load_config(write_config(BOUNDARY_CONFIG))
    assert classify("What is the expense ratio after the merger?", cfg).intent == "FACTUAL"
    assert classify("What is the navigator used for?", cfg).intent == "FACTUAL"
    assert classify("What is the ter of HDFC Large Cap Fund?", cfg).intent == "ADVICE"
    assert classify("What is the nav of HDFC Large Cap Fund?", cfg).intent == "PERFORMANCE"


def test_refusal_messages_are_the_frozen_text() -> None:
    assert REFUSAL_MESSAGES == FROZEN_MESSAGES
    assert SAFE_ADVICE_TEXT == FROZEN_MESSAGES["ADVICE"]
    assert SAFE_PERFORMANCE_TEXT == FROZEN_MESSAGES["PERFORMANCE"]
    assert SAFE_PII_TEXT == FROZEN_MESSAGES["PII"]
    assert SAFE_OUT_OF_CORPUS_TEXT == FROZEN_MESSAGES["OUT_OF_CORPUS"]


@pytest.mark.parametrize(
    "question,intent",
    [*PRD_REFUSAL_QUESTIONS, ("What is the expense ratio of Parag Parflex Fund?", "OUT_OF_CORPUS")],
    ids=[*[q for q, _ in PRD_REFUSAL_QUESTIONS], "What is the expense ratio of Parag Parflex Fund?"],
)
def test_refusal_answer_payload(real_config: AppConfig, question: str, intent: str) -> None:
    decision = classify(question, real_config)
    answer = refusal_answer(decision)
    assert answer.status == REFUSAL_STATUSES[intent]
    assert answer.status in {"refused_advice", "refused_performance", "refused_pii", "out_of_corpus"}
    assert REFUSAL_MESSAGES[intent] in answer.text
    assert answer.doc_type == ""
    assert answer.publisher == ""
    assert answer.content_as_of == "unknown"
    assert answer.used_chunk_ids == ()
    assert answer.was_cached is False
    if intent == "OUT_OF_CORPUS":
        assert decision.educational_url is None
        assert answer.source_url == ""
        assert answer.text == REFUSAL_MESSAGES["OUT_OF_CORPUS"]
    else:
        assert decision.educational_url == real_config.refusal.educational_links[intent.lower()]
        assert answer.source_url == decision.educational_url
        assert answer.text == REFUSAL_MESSAGES[intent]


def test_factual_decision_is_not_a_refusal() -> None:
    with pytest.raises(ValueError):
        refusal_answer(Decision(intent="FACTUAL", educational_url=None, reason_code="factual"))


def test_no_question_text_is_logged(real_config: AppConfig, caplog: pytest.LogCaptureFixture) -> None:
    question = "My PAN is ABCDE1234F, what are the returns?"
    with caplog.at_level(logging.INFO):
        decision = classify(question, real_config)
    assert decision.intent == "PII"
    tokens = {token for token in normalise_question(question).split() if len(token) > 3}
    assert tokens
    assert caplog.records
    for record in caplog.records:
        message = record.getMessage()
        assert "intent=PII" in message
        assert "reason_code=pii:0" in message
        assert "ABCDE1234F" not in message
        assert not any(token in message.lower() for token in tokens)


def test_scan_pii_uses_the_supplied_patterns() -> None:
    patterns = (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", r"\b\d{10}\b")
    assert scan_pii("my pan is ABCDE1234F", patterns) is True
    assert scan_pii("my number is 9876543210", patterns) is True
    assert scan_pii("What is the expense ratio of HDFC Large Cap Fund?", patterns) is False


def test_normalise_question_lowercases_and_strips_punctuation() -> None:
    assert normalise_question("HDFC Large Cap — Direct?  Growth") == "hdfc large cap direct growth"
    assert normalise_question("  What   is  the 80C benefit? ") == "what is the 80c benefit"


def test_guard_needs_no_network(real_config: AppConfig, no_network: None) -> None:
    assert classify("What is the expense ratio of HDFC Large Cap Fund?", real_config).intent == "FACTUAL"


def test_golden_set_still_declares_ten_questions_with_one_refusal() -> None:
    assert len(GOLDEN_QUESTIONS) == 10
    assert sum(1 for _, _, expect_refusal in GOLDEN_QUESTIONS if expect_refusal) == 1
