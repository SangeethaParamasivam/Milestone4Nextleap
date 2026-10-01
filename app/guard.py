"""Deterministic pre-LLM guard (architecture §6.1, implementation §10).

Evaluation order is fixed at PII > ADVICE > PERFORMANCE > OUT_OF_CORPUS > FACTUAL. PII is
scanned on the raw question so the case-sensitive identifier regexes keep working, the
question body is dropped before anything is logged, and every log record carries only
`intent` and `reason_code` (FR-7.5, R8). The LLM never adjudicates a refusal: the
generator consumes `decision.intent` and never re-classifies (A1).

Keyword lists and PII regexes are read from `config.yaml` (`refusal.advice`,
`refusal.performance`, `refusal.pii_patterns`, `out_of_corpus.known_foreign_schemes`) so
behaviour stays inspectable without reading code (FR-7.2, FR-10.3). No PII pattern is
hard-coded here. The only regular expressions built in code are:

* the word-boundary anchored phrase matcher for advice/performance keywords, foreign
  scheme names and in-scope scheme aliases, so `ter` cannot match `after` and `nav`
  cannot match `navigator`;
* the punctuation-to-space normaliser behind `normalise_question`.

Configured PII regexes are compiled with `re.IGNORECASE` so lowercase spellings of the
configured identifier keywords are caught as well.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from app.models import Answer, Decision
from config import AppConfig

_LOG = logging.getLogger("app.guard")

SAFE_ADVICE_TEXT = (
    "I answer only published facts about these schemes and can't recommend, compare or rate "
    "funds or suggest a portfolio."
)
SAFE_PERFORMANCE_TEXT = (
    "I don't state or compute returns. Performance figures for each scheme are in its official "
    "factsheet, linked below."
)
SAFE_PII_TEXT = (
    "Please don't share personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). "
    "This message has not been stored. Delete it from your chat."
)
SAFE_OUT_OF_CORPUS_TEXT = (
    "I don't have that in my sources. I answer only about: HDFC Large Cap, HDFC Equity (Flexi Cap), "
    "HDFC ELSS Tax Saver, HDFC Small Cap and HDFC Balanced Advantage funds, and general official "
    "SEBI/AMFI information."
)

REFUSAL_MESSAGES: dict[str, str] = {
    "ADVICE": SAFE_ADVICE_TEXT,
    "PERFORMANCE": SAFE_PERFORMANCE_TEXT,
    "PII": SAFE_PII_TEXT,
    "OUT_OF_CORPUS": SAFE_OUT_OF_CORPUS_TEXT,
}

REFUSAL_STATUSES: dict[str, str] = {
    "ADVICE": "refused_advice",
    "PERFORMANCE": "refused_performance",
    "PII": "refused_pii",
    "OUT_OF_CORPUS": "out_of_corpus",
}

_PUNCTUATION = re.compile(r"[^\w\s]")
_PLAN_OPTION_TOKENS = frozenset({"direct", "regular", "growth", "dividend", "plan", "option"})
_PHRASE_PATTERNS: dict[str, re.Pattern[str]] = {}
_PII_PATTERNS: dict[str, re.Pattern[str]] = {}


def normalise_question(question: str) -> str:
    """Lowercase, replace punctuation with a space and collapse runs of whitespace."""
    return " ".join(_PUNCTUATION.sub(" ", question).lower().split())


def scan_pii(question: str, patterns: Sequence[str]) -> bool:
    """True when any configured PII regex matches; the matched text is never returned."""
    return _pii_pattern_index(question, patterns) is not None


def classify(question: str, cfg: AppConfig) -> Decision:
    """Classify the question in order PII > ADVICE > PERFORMANCE > OUT_OF_CORPUS > FACTUAL."""
    pii_index = _pii_pattern_index(question, cfg.refusal.pii_patterns)
    if pii_index is not None:
        return _decide("PII", cfg, f"pii:{pii_index}")
    haystack = normalise_question(question)
    advice = _first_match(haystack, cfg.refusal.advice)
    if advice is not None:
        return _decide("ADVICE", cfg, f"advice:{advice}")
    performance = _first_match(haystack, cfg.refusal.performance)
    if performance is not None:
        return _decide("PERFORMANCE", cfg, f"performance:{performance}")
    foreign = _first_match(haystack, sorted(cfg.out_of_corpus))
    if foreign is not None and not _has_in_scope_alias(haystack, cfg):
        return _decide("OUT_OF_CORPUS", cfg, f"out_of_corpus:{foreign}")
    return _decide("FACTUAL", cfg, "factual")


def refusal_answer(decision: Decision) -> Answer:
    """Render the frozen refusal payload, appending the educational link when there is one."""
    if decision.intent not in REFUSAL_MESSAGES:
        raise ValueError(f"not a refusal intent: {decision.intent}")
    message = REFUSAL_MESSAGES[decision.intent]
    link = decision.educational_url or ""
    return Answer(
        text=message,
        source_url=link,
        doc_type="",
        publisher="",
        content_as_of="unknown",
        used_chunk_ids=(),
        status=REFUSAL_STATUSES[decision.intent],
        was_cached=False,
    )


def _decide(intent: str, cfg: AppConfig, reason_code: str) -> Decision:
    educational_url = cfg.refusal.educational_links.get(intent.lower()) or None
    _LOG.info("intent=%s reason_code=%s", intent, reason_code)
    return Decision(intent=intent, educational_url=educational_url, reason_code=reason_code)


def _pii_pattern_index(question: str, patterns: Sequence[str]) -> int | None:
    for index, pattern in enumerate(patterns):
        if _pii_pattern(pattern).search(question) is not None:
            return index
    return None


def _pii_pattern(pattern: str) -> re.Pattern[str]:
    compiled = _PII_PATTERNS.get(pattern)
    if compiled is None:
        compiled = re.compile(pattern, re.IGNORECASE)
        _PII_PATTERNS[pattern] = compiled
    return compiled


def _first_match(haystack: str, phrases: Iterable[str]) -> str | None:
    for phrase in phrases:
        if _phrase_pattern(phrase).search(haystack) is not None:
            return phrase
    return None


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Anchor the phrase on word boundaries so short keywords cannot match inside a word."""
    compiled = _PHRASE_PATTERNS.get(phrase)
    if compiled is not None:
        return compiled
    body = r"\s+".join(re.escape(token) for token in normalise_question(phrase).split())
    if body and body[0].isalnum():
        body = rf"\b{body}"
    if body and body[-1].isalnum():
        body = rf"{body}\b"
    compiled = re.compile(body)
    _PHRASE_PATTERNS[phrase] = compiled
    return compiled


def _has_in_scope_alias(haystack: str, cfg: AppConfig) -> bool:
    return any(_phrase_pattern(alias).search(haystack) is not None for alias in _in_scope_aliases(cfg))


def _in_scope_aliases(cfg: AppConfig) -> tuple[str, ...]:
    """Allowlisted scheme aliases: the full configured name and the name without its plan/option words."""
    aliases: list[str] = []
    for spec in cfg.sources:
        if not spec.scheme_name:
            continue
        full = normalise_question(spec.scheme_name)
        if not full:
            continue
        aliases.append(full)
        stripped = " ".join(token for token in full.split() if token not in _PLAN_OPTION_TOKENS)
        if stripped and stripped != full:
            aliases.append(stripped)
    return tuple(aliases)
