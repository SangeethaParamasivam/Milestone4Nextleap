from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

Hashable = str | int | float | bool

_WHITESPACE = re.compile(r"\s+")
_HASH_STRIP = re.compile(r"[^a-z0-9.%]+")

ANSWER_STATUSES = frozenset(
    {"ok", "out_of_corpus", "refused_advice", "refused_performance", "refused_pii", "suppressed"}
)
RETRIEVAL_STATUSES = frozenset({"ok", "out_of_corpus"})
INTENTS = frozenset({"FACTUAL", "ADVICE", "PERFORMANCE", "PII", "OUT_OF_CORPUS"})


def hash_normalise(text: str) -> str:
    """Lowercase, collapse whitespace and drop punctuation except '.' and '%' (ID derivation input)."""
    folded = unicodedata.normalize("NFKC", text).lower()
    stripped = _HASH_STRIP.sub(" ", folded)
    return _WHITESPACE.sub(" ", stripped).strip()


@dataclass(frozen=True)
class SourceSpec:
    id: str
    url: str
    domain: str
    doc_type: str
    scheme_name: str | None
    scheme_category: str | None
    plan: str | None
    option: str | None
    publisher: str


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    url: str
    status: str
    http_code: int | None
    bytes: int
    sha256: str | None
    fetched_at: str
    cache_hit: bool
    error: str | None


@dataclass(frozen=True)
class Block:
    kind: str
    heading_path: str
    text: str
    row_group: int | None


@dataclass(frozen=True)
class CleanDocument:
    source_id: str
    url: str
    page_title: str
    blocks: tuple[Block, ...]
    extracted_chars: int


@dataclass(frozen=True)
class Chunk:
    source_id: str
    source_url: str
    publisher: str
    source_domain: str
    page_title: str
    doc_type: str
    scheme_name: str | None
    scheme_category: str | None
    plan: str | None
    option: str | None
    topic: str
    section_heading: str
    content_as_of: str
    chunk_index: int
    text: str
    ingested_at: str
    token_count: int

    @property
    def normalised_text(self) -> str:
        return hash_normalise(self.text)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.normalised_text.encode("utf-8")).hexdigest()

    @property
    def chunk_id(self) -> str:
        return f"{self.source_id}::{self.topic}::{self.content_hash[:8]}"

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_metadata(self) -> dict[str, Hashable]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "publisher": self.publisher,
            "source_domain": self.source_domain,
            "page_title": self.page_title,
            "doc_type": self.doc_type,
            "scheme_name": self.scheme_name or "",
            "scheme_category": self.scheme_category or "",
            "plan": self.plan or "",
            "option": self.option or "",
            "topic": self.topic,
            "section_heading": self.section_heading,
            "content_as_of": self.content_as_of,
            "chunk_index": self.chunk_index,
            "char_len": self.char_len,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "ingested_at": self.ingested_at,
        }

    def to_record(self) -> dict[str, Hashable | str]:
        record: dict[str, Hashable | str] = dict(self.to_metadata())
        record["text"] = self.text
        return record

    @classmethod
    def from_metadata(cls, md: Mapping[str, Any], text: str) -> "Chunk":
        return cls(
            source_id=str(md.get("source_id", "")),
            source_url=str(md.get("source_url", "")),
            publisher=str(md.get("publisher", "")),
            source_domain=str(md.get("source_domain", "")),
            page_title=str(md.get("page_title", "")),
            doc_type=str(md.get("doc_type", "")),
            scheme_name=_opt_str(md.get("scheme_name")),
            scheme_category=_opt_str(md.get("scheme_category")),
            plan=_opt_str(md.get("plan")),
            option=_opt_str(md.get("option")),
            topic=str(md.get("topic", "general")),
            section_heading=str(md.get("section_heading", "")),
            content_as_of=str(md.get("content_as_of", "")),
            chunk_index=int(md.get("chunk_index", 0)),
            text=text,
            ingested_at=str(md.get("ingested_at", "")),
            token_count=int(md.get("token_count", 0)),
        )


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    dense_rank: int
    lexical_rank: int | None


@dataclass(frozen=True)
class RetrievalResult:
    hits: tuple[RetrievalHit, ...]
    status: str
    filtered_topic: str
    total_candidates: int


@dataclass(frozen=True)
class Decision:
    intent: str
    educational_url: str | None
    reason_code: str


@dataclass(frozen=True)
class Answer:
    text: str
    source_url: str
    doc_type: str
    publisher: str
    content_as_of: str
    used_chunk_ids: tuple[str, ...]
    status: str
    was_cached: bool

    def render(self) -> str:
        source = self.source_url or "none"
        as_of = self.content_as_of or "unknown"
        extra = f" ({self.doc_type} - {self.publisher})" if self.doc_type else ""
        return (
            f"{self.text}\n\nSource: {source}{extra}\n"
            f"Last updated from sources: {as_of}"
        )


@dataclass(frozen=True)
class IngestStatus:
    corpus_hash: str
    chunk_count: int
    model: str
    collection: str
    last_ingest_at: str
    source_hashes: Mapping[str, str]


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
