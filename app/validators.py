"""Pure answer validators (implementation §9.4-§9.5). No LLM, no network, no I/O.

Every function returns `list[Violation]` so `validate_answer` can concatenate the results of
checks 2-8 in order. These are the post-LLM gates: they run on model output before it may reach
a user, and they must be testable with no API key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.models import RetrievalHit
from config import AppConfig

SOURCE_LINE = re.compile(r"^\s*Source:\s*(\S+)")
ASOF_LINE = re.compile(r"^\s*Last updated from sources:\s*(.+)$")
_MARKDOWN_LINK = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREVIATION = re.compile(r"\b(?:e\.g\.|i\.e\.|Rs\.|vs\.|p\.a\.|Mr\.|Mrs\.|Dr\.|No\.)", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PERFORMANCE_WORD = re.compile(
    r"\b\d+(?:\.\d+)?\s*%\s*(?:return|cagr|growth|gain)"
    r"|\b(?:cagr|xirr|ytd|p\.?a\.?|irr)\b"
    r"|1\s*[- ]?year\s+return"
    r"|\breturned\s+\d",
    re.IGNORECASE,
)
_PERFORMANCE_KEYWORDS = re.compile(
    r"\b(?:returns?|performance|outperformed|underperformed|profit|gain)\b", re.IGNORECASE
)
_BENCHMARK_NAME = re.compile(
    r"\b(?:total\s+return|return)\s+(?:index|tri\b)|index\s+return", re.IGNORECASE
)
_SPELLED = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_PERCENT = re.compile(r"%")
INSUFFICIENT_CONTEXT = "I don't have that in my sources."


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str
    value: str = ""


def split_answer_lines(text: str) -> tuple[str, str, str]:
    """Split model output into `(body, source_line, asof_line)`.

    The `^\\s*` anchors are required: a `Source:` occurring inside a sentence is body text, not
    the citation line (§9.5.1).
    """
    body_lines: list[str] = []
    source_line = ""
    asof_line = ""
    for line in text.splitlines():
        source_match = SOURCE_LINE.match(line)
        if source_match is not None:
            source_line = source_match.group(1).strip()
            continue
        asof_match = ASOF_LINE.match(line)
        if asof_match is not None:
            asof_line = asof_match.group(1).strip()
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip(), source_line, asof_line


def normalise_url(url: str) -> str:
    """Strip markdown and trailing punctuation, lowercase the host, drop `www.` and the trailing `/`."""
    candidate = url.strip()
    link = _MARKDOWN_LINK.match(candidate)
    if link is not None:
        candidate = link.group(2)
    candidate = candidate.strip().rstrip(".,;)")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    scheme, _, rest = candidate.partition("://")
    host, slash, path = rest.partition("/")
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    path = path.rstrip("/")
    return f"{scheme.lower()}://{host}" + (f"/{path}" if slash and path else "")


def check_citation(
    text: str,
    allowed: set[str],
    configured: set[str] | None = None,
) -> list[Violation]:
    """Codes: `citation_missing`, `citation_not_retrieved`, `citation_not_in_allowlist`, `citation_multiple`.

    `allowed` is the retrieved chunk URLs intersected with the configured source URLs. A cited
    URL that is in `configured` but was not retrieved is `citation_not_retrieved`; anything else
    is `citation_not_in_allowlist`. The optional `configured` argument exists only to tell those
    two cases apart, and defaults to treating any unrecognised URL as not in the allowlist.
    """
    citations = [line for line in text.splitlines() if SOURCE_LINE.match(line)]
    if not citations:
        return [Violation("citation_missing", "no line starting with 'Source:'", "")]
    if len(citations) > 1:
        return [
            Violation("citation_multiple", f"{len(citations)} Source: lines", citations[0].strip())
        ]
    cited = normalise_url(SOURCE_LINE.match(citations[0]).group(1))
    if cited in {normalise_url(url) for url in allowed}:
        return []
    if configured is not None and cited in {normalise_url(url) for url in configured}:
        return [Violation("citation_not_retrieved", cited, cited)]
    return [Violation("citation_not_in_allowlist", cited, cited)]


def check_sentence_count(body: str, max_sentences: int) -> list[Violation]:
    """Count body sentences after masking decimals and common abbreviations."""
    masked = _ABBREVIATION.sub(lambda match: match.group(0).replace(".", ""), body)
    masked = _DECIMAL.sub("0", masked)
    segments = [segment for segment in _SENTENCE_SPLIT.split(masked) if segment.strip()]
    if len(segments) > max_sentences:
        return [Violation("too_many_sentences", f"{len(segments)} sentences", str(len(segments)))]
    return []


def check_word_count(body: str, max_words: int) -> list[Violation]:
    """Whitespace tokens in the body. Reported; only suppresses together with `too_many_sentences`."""
    count = len(body.split())
    if count > max_words:
        return [Violation("too_long", f"{count} words", str(count))]
    return []


def check_numbers(body: str, retrieved_text: str, question: str) -> list[Violation]:
    """Every numeric literal in the body must appear in the retrieved text or the question."""
    allowed = _allowed_numbers(retrieved_text, question)
    violations: list[Violation] = []
    for literal in _NUMBER.findall(body):
        if _numeric_value(literal) in allowed:
            continue
        violations.append(Violation("unsourced_number", literal, literal))
    return violations


def _allowed_numbers(retrieved_text: str, question: str) -> set[float]:
    values = {_numeric_value(literal) for literal in _NUMBER.findall(f"{retrieved_text} {question}")}
    values.update(float(number) for number in _SPELLED.values())
    values.add(80.0)
    values.update(float(digit) for digit in range(10))
    return values


def _numeric_value(literal: str) -> float:
    """`1,00,000` and `100000` are equal, and a trailing `%` is decoration."""
    bare = literal.replace(",", "")
    if bare.endswith("%"):
        bare = bare[:-1]
    return float(bare)


def check_pii(text: str, patterns: Sequence[str]) -> list[Violation]:
    """The configured `refusal.pii_patterns`, case-insensitively."""
    violations: list[Violation] = []
    for index, pattern in enumerate(patterns):
        try:
            match = re.search(pattern, text, re.IGNORECASE)
        except re.error:
            continue
        if match is not None:
            violations.append(Violation("pii", f"pii_patterns[{index}]", match.group(0)))
    return violations


def check_performance(text: str) -> list[Violation]:
    """Only return-context percentages violate; an expense ratio, exit load or benchmark name may hold `%`."""
    structured = _PERFORMANCE_WORD.search(text)
    if structured is not None:
        return [Violation("performance", structured.group(0), structured.group(0))]
    if _PERCENT.search(text):
        keyword = _PERFORMANCE_KEYWORDS.search(text)
        if keyword is not None and not _rate_context(text) and not _BENCHMARK_NAME.search(text):
            return [Violation("performance", keyword.group(0), keyword.group(0))]
    return []


def _rate_context(text: str) -> bool:
    """True when a percentage is attached to a rate noun such as expense ratio or exit load."""
    return re.search(
        r"(?:expense\s+ratio|exit\s+load|ter|ter\b|charge[sd]?|yield|nav|holdings|fund\s+size|aum|sip|minimum)",
        text,
        re.IGNORECASE,
    ) is not None


def check_asof_line(asof_line: str, hits: Sequence[RetrievalHit]) -> list[Violation]:
    """The printed as-of value must match a retrieved chunk's `content_as_of`, or be `unknown`.

    §9.5.8 allows `unknown` "when all cited chunks have no date". One page is chunked into
    several blocks and only some of them carry a date, so "all" cannot be tested per chunk: the
    same `source_url` legitimately appears with a date on one hit and without one on the next.
    A cited page is therefore treated as undated when *any* retrieved chunk for it is undated,
    which is what keeps a correct `unknown` from being suppressed.
    """
    stated = asof_line.strip()
    if not stated:
        return [Violation("asof_missing", "no 'Last updated from sources:' line", "")]
    dates = {hit.chunk.content_as_of.strip() for hit in hits if hit.chunk.content_as_of.strip()}
    if stated in dates:
        return []
    undated = sum(1 for hit in hits if not hit.chunk.content_as_of.strip())
    if stated == "unknown" and (undated or not dates):
        return []
    return [Violation("asof_mismatch", stated, stated)]


def validate_answer(
    text: str,
    hits: Sequence[RetrievalHit],
    question: str,
    cfg: AppConfig,
) -> list[Violation]:
    """Run checks 2-8 of §9.5 in order and concatenate the violations."""
    body, _, asof_line = split_answer_lines(text)
    configured = {source.url for source in cfg.sources}
    allowed = citation_allowlist_urls(hits, cfg)

    violations: list[Violation] = check_citation(text, allowed, configured)
    violations.extend(check_sentence_count(body, cfg.generation.max_sentences))
    violations.extend(check_word_count(body, cfg.generation.max_words))
    violations.extend(check_numbers(body, _retrieved_text(hits), question))
    violations.extend(check_pii(text, cfg.refusal.pii_patterns))
    violations.extend(check_performance(body))
    violations.extend(check_asof_line(asof_line, hits))
    if body.strip() == INSUFFICIENT_CONTEXT:
        violations.append(Violation("context_insufficient", "model reported the context lacks the answer", ""))
    return violations


def _retrieved_text(hits: Sequence[RetrievalHit]) -> str:
    return " ".join(hit.chunk.text for hit in hits)


def citation_allowlist_urls(hits: Sequence[RetrievalHit], cfg: AppConfig) -> set[str]:
    """Retrieved chunk URLs intersected with the configured source URLs (§9.5.2)."""
    configured = {source.url for source in cfg.sources}
    return {hit.chunk.source_url for hit in hits if hit.chunk.source_url in configured}


CITATION_CODES = frozenset(
    {"citation_missing", "citation_not_retrieved", "citation_not_in_allowlist", "citation_multiple"}
)
SUPPRESSING_CODES = frozenset(
    {"unsourced_number", "pii", "performance", "too_many_sentences", "asof_mismatch", "asof_missing"}
)


def is_insufficient_context(violations: Sequence[Violation]) -> bool:
    """The model stated that the context does not answer the question, so nothing was answered."""
    return any(violation.code == "context_insufficient" for violation in violations)


def is_suppressing(violations: Sequence[Violation]) -> bool:
    """A violation that suppresses immediately, with no retry (§9.6.8).

    `too_long` alone is reported to the eval report but does not suppress; combined with
    `too_many_sentences` it does.
    """
    codes = {violation.code for violation in violations}
    if codes & SUPPRESSING_CODES:
        return True
    return "too_long" in codes and "too_many_sentences" in codes


def is_citation_failure(violations: Sequence[Violation]) -> bool:
    """Citation failures earn exactly one retry (§9.6.7)."""
    return bool({violation.code for violation in violations} & CITATION_CODES)
