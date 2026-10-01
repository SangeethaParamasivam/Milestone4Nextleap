from __future__ import annotations

import re
from typing import Mapping

from config import AppConfig, ConfigError

GENERAL_TOPIC = "general"

TOPIC_VOCABULARY: tuple[str, ...] = (
    "expense_ratio",
    "exit_load",
    "entry_load",
    "min_sip",
    "min_lump_sum",
    "lockin_80c",
    "riskometer",
    "benchmark",
    "statements_download",
    "nav",
    "plan_direct_vs_regular",
    "objective",
    GENERAL_TOPIC,
)

_PHRASE_PATTERNS: dict[str, re.Pattern[str]] = {}


def classify_topic(
    heading_path: str,
    text: str,
    cfg: AppConfig,
    *,
    scheme: str | None = None,
) -> str:
    """First `cfg.topics` key whose any phrase occurs in `heading_path + " " + text`, else `general`.

    Three navigational sections are deliberately `general` before any phrase scan, because their
    text carries tokens from many fund titles and would otherwise steal a wrong topic label:

    * an "About <fund>" section — one paragraph holding many distinct facts (launch date, fund
      manager, AUM, NAV, `rated <level> risk`, minimum SIP, minimum lump sum, exit load),
      answered by every topic filter through its `general` fallback;
    * a manager's "Also manages these schemes ..." list — scheme-name tokens such as
      "Direct Plan Growth" or "NIFTY 50" are not the section's content, but they match the
      `plan_direct_vs_regular` and `benchmark` phrases word-for-word;
    * a "Compare similar funds" table — competitor scheme rows matching those same phrases.

    These rules exist because chunk topics are single-valued even though these sections are not;
    mislabelling silently steers the wrong chunks into the top-`k` for unrelated questions.

    The owning scheme's own name is stripped from the phrase-scan haystack before any `cfg.topics`
    phrase is matched, for the same reason: every H1 title and scheme-name suffix would otherwise
    carry the scheme's plan word ("... - Direct Plan - Growth") and label ordinary chunks
    `plan_direct_vs_regular` or `benchmark`.
    """
    validate_topics(cfg.topics)
    leaf = heading_path.split(" > ")[-1].strip().lower()
    if leaf.startswith(("about ", "compare ")):
        return GENERAL_TOPIC
    if "also manages these schemes" in text.lower():
        return GENERAL_TOPIC
    haystack = f"{heading_path} {text}".lower()
    if scheme:
        scheme_tokens = {token for token in scheme.lower().split() if token.isalnum()}
        haystack = " ".join(token for token in haystack.split() if token not in scheme_tokens)
    for topic, phrases in cfg.topics.items():
        for phrase in phrases:
            if phrase_pattern(phrase).search(haystack):
                return topic
    return GENERAL_TOPIC


def validate_topics(topics: Mapping[str, tuple[str, ...]]) -> None:
    """Reject any `config.yaml` topic key outside the PRD §6.4 controlled vocabulary."""
    unknown = sorted(set(topics) - set(TOPIC_VOCABULARY))
    if unknown:
        raise ConfigError(f"topics outside the PRD 6.4 controlled vocabulary: {', '.join(unknown)}")
    if GENERAL_TOPIC in topics:
        raise ConfigError(f"topics must not redefine the reserved topic {GENERAL_TOPIC!r}")


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary anchored so `ter` cannot match `after` and `nav` cannot match `navigation`."""
    cached = _PHRASE_PATTERNS.get(phrase)
    if cached is not None:
        return cached
    body = r"\s+".join(re.escape(token) for token in phrase.split())
    if body and body[0].isalnum():
        body = rf"\b{body}"
    if body and body[-1].isalnum():
        body = rf"{body}\b"
    pattern = re.compile(body)
    _PHRASE_PATTERNS[phrase] = pattern
    return pattern
