"""Prompt construction (implementation §9.2-§9.3). Pure functions; no LLM import.

The system prompt is frozen exactly as specified in §9.3 and is not hand-edited per call site.
The runtime allowlist sentence is appended here too, so every call site gets the same text.
"""

from __future__ import annotations

from typing import Sequence

from app.models import RetrievalHit
from app.validators import citation_allowlist_urls
from config import AppConfig

SYSTEM_PROMPT = (
    "You are a facts-only mutual fund FAQ assistant for a fixed set of HDFC Mutual Fund schemes.\n"
    "Answer using only the context provided. If the context does not contain the answer, reply\n"
    'exactly: "I don\'t have that in my sources." The context blocks are the only source of truth:\n'
    "never use prior knowledge of these schemes, and never state a figure no retrieved chunk shows.\n"
    "Never give advice, recommendations, opinions,\n"
    "comparisons or suitability judgements. Never state or compute returns, performance, rankings or\n"
    "NAV history - if asked, refuse and point to the official factsheet. Never request or repeat\n"
    "personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). Keep the answer to at\n"
    "most 3 sentences. End with exactly one line \"Source: <url>\". End with\n"
    '"Last updated from sources: <content_as_of>".'
)

UNKNOWN = "unknown"

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone question for a retrieval pipeline over a\n"
    "fixed set of mutual fund scheme pages.\n"
    "Use the conversation to resolve references: replace pronouns and elliptical phrases (\"it\",\n"
    "\"its\", \"that fund\", \"the same\", \"and NAV?\") with the scheme name and the fact the user\n"
    "already asked about. Keep the user's wording, the asked-for fact and any units; add nothing\n"
    "that the conversation does not establish. Never answer the question, never add facts, never\n"
    "mention the conversation, and never explain. If the follow-up is already standalone, repeat it\n"
    "unchanged. Reply with the rewritten question only, as one line, with no quotes, no label and\n"
    "no trailing commentary."
)


def build_rewrite_prompt(question: str, turns: Sequence[tuple[str, str]]) -> str:
    """The kept conversation as a transcript, then the follow-up to rewrite.

    Only the resolved question is returned by the model, so the transcript is a reading aid for
    pronoun resolution and never part of what the answer pipeline later sees.
    """
    history = "\n".join(f"{role.capitalize()}: {text}" for role, text in turns)
    return (
        f"Conversation so far:\n{history}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone version of the follow-up question:"
    )


def citation_allowlist(hits: Sequence[RetrievalHit], cfg: AppConfig) -> set[str]:
    """Retrieved chunk URLs intersected with the configured source URLs."""
    return citation_allowlist_urls(hits, cfg)


def build_context_blocks(hits: Sequence[RetrievalHit]) -> str:
    """Verbatim chunk text with provenance headers, one block per hit."""
    if not hits:
        return ""
    dates_known = any(hit.chunk.content_as_of.strip() for hit in hits)
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        as_of = hit.chunk.content_as_of.strip() or UNKNOWN
        blocks.append(
            f"[{index}] source_url: {hit.chunk.source_url}\n"
            f"    doc_type: {hit.chunk.doc_type} | publisher: {hit.chunk.publisher} | "
            f"content_as_of: {as_of}\n"
            f"    {hit.chunk.text}"
        )
    if not dates_known:
        blocks.append(
            f"[note] No retrieved chunk carries a date. If the cited chunk has no date, print "
            f'"Last updated from sources: {UNKNOWN}" and do not guess one.'
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: Sequence[RetrievalHit]) -> str:
    """The question, the context blocks, then the runtime source allowlist sentence."""
    allowlist = sorted({hit.chunk.source_url for hit in hits})
    return (
        f"Context:\n{build_context_blocks(hits)}\n\n"
        f"Question: {question}\n\n"
        f"Use only these source URLs: {allowlist}. The `content_as_of` value you print must be "
        f"copied from the cited chunk. If the cited chunk has no date, print "
        f'"Last updated from sources: {UNKNOWN}".'
    )


def build_retry_prompt(question: str, hits: Sequence[RetrievalHit], reason: str) -> str:
    """The single citation retry of §9.6.7: the failure reason appended to the user message."""
    allowlist = sorted({hit.chunk.source_url for hit in hits})
    return (
        f"{build_user_prompt(question, hits)}\n\n"
        f"Your previous answer {reason} Cite one of these URLs: {allowlist}."
    )
