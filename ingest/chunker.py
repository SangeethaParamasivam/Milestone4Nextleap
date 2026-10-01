from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.models import Block, Chunk, CleanDocument, SourceSpec
from config import DEFAULT_CONFIG_PATH, AppConfig
from ingest.loader import meta_path
from ingest.topics import classify_topic

_LOG = logging.getLogger("ingest.chunker")

CHUNKS_FILE = "chunks.jsonl"
PREVIEW_FILE = "chunks_preview.txt"
HEADING_SEPARATOR = " > "
BLOCK_SEPARATOR = "\n"
BOUNDARY_MARKS = frozenset(".?!;:")
FIRST_WINDOW_CHARS = 40
CHARS_PER_TOKEN = 4
SOURCES_CSV_HEADER = (
    "source_id",
    "url",
    "publisher",
    "doc_type",
    "scheme_name",
    "scheme_category",
    "plan",
    "option",
    "allowed_domain",
    "first_fetched_at",
    "last_fetched_at",
    "sha256",
    "chunk_count",
    "retrieved_at",
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# The 4-digit year alternative must be tried FIRST. With `\d{2}|\d{4}` the regex
# matches "29 Sep 2026" as "29 Sep 20" and silently stamps 2020-09-29, because the
# shorter alternative is listed first and nothing anchors the year to a boundary.
_YEAR = r"(?:['’]?\d{4}|['’]?\d{2})"
_DATE_STAMP = re.compile(
    r"(?:as on|as of|as at|last updated(?:\s+on)?)\s+"
    rf"(\d{{1,2}}\s+[A-Za-z]{{3,9}}\.?,?\s+{_YEAR})"
    rf"|NAV:?\s+(\d{{1,2}}\s+[A-Za-z]{{3,9}}\.?,?\s+{_YEAR})",
    re.IGNORECASE,
)
_PROTECTED = re.compile(
    r"(?:as on|as of|as at)\s+\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}"
    r"|\d+(?:[.,]\d+)*\s?%"
    r"|\u20b9\s?\d+(?:[.,]\d+)*"
    r"|Rs\.?\s?\d+(?:[.,]\d+)*"
    r"|\d+(?:[.,]\d+)*\s?(?:bps|bp|cr|lakh|crore|years?|months?|days?)",
    re.IGNORECASE,
)


class ChunkingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkPlan:
    chunks: tuple[Chunk, ...]
    dropped: int


def count_tokens(text: str) -> int:
    """Model-free estimate; Phase 4's real tokenizer asserts the true count."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def sources_csv_path(config_path: str | Path | None = None) -> Path:
    """D2 source list, written beside the loaded `config.yaml` so no module hardcodes a path."""
    return (Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH).parent / "sources.csv"


def chunk_document(doc: CleanDocument, spec: SourceSpec, cfg: AppConfig) -> list[Chunk]:
    """Chunks for one cleaned page, in document order, with 0-based `chunk_index` per page."""
    return list(plan_chunks(doc, spec, cfg).chunks)


def plan_chunks(doc: CleanDocument, spec: SourceSpec, cfg: AppConfig) -> ChunkPlan:
    """Section-aware, heading-anchored, token-bounded chunks plus the count dropped under `min_chars`."""
    size, overlap = cfg.chunking.size, cfg.chunking.overlap
    if overlap >= size:
        raise ChunkingError(f"chunking.overlap={overlap} must be smaller than chunking.size={size}")
    if not spec.url or not spec.doc_type:
        raise ChunkingError(f"source {spec.id} cannot produce chunks without url and doc_type")
    ingested_at = _utc_now()
    chunks: list[Chunk] = []
    dropped = 0
    for section in _sections(doc.blocks):
        for text, section_heading in _section_pieces(section, size, overlap, cfg.chunking.max_tokens):
            if len(text) < cfg.chunking.min_chars:
                dropped += 1
                continue
            chunks.append(
                _build_chunk(
                    text=text,
                    section_heading=section_heading,
                    section_path=section[-1].heading_path,
                    index=len(chunks),
                    doc=doc,
                    spec=spec,
                    cfg=cfg,
                    ingested_at=ingested_at,
                )
            )
            if len(chunks) > cfg.chunking.max_chunks_per_page:
                raise ChunkingError(
                    f"{doc.source_id} produced {len(chunks)} chunks, above max_chunks_per_page="
                    f"{cfg.chunking.max_chunks_per_page}: a cleaning failure, not a truncation"
                )
    page_as_of = _content_as_of("\n".join(block.text for block in doc.blocks))
    chunks = _apply_page_as_of(chunks, page_as_of)
    if dropped:
        _LOG.info("[chunk] %s dropped %d blocks under min_chars=%d", doc.source_id, dropped, cfg.chunking.min_chars)
    return ChunkPlan(tuple(chunks), dropped)


def write_chunks(chunks: Sequence[Chunk], cfg: AppConfig, *, dropped: int = 0) -> None:
    """Write `chunks.jsonl` and `chunks_preview.txt` from scratch, both via a temp file swap."""
    chunks_dir = cfg.paths.chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n" for chunk in chunks)
    _write_atomic(chunks_dir / CHUNKS_FILE, lines.encode("utf-8"))
    preview = _render_preview(chunks, dropped)
    _write_atomic(chunks_dir / PREVIEW_FILE, preview.encode("utf-8"))
    _LOG.info(
        "[chunk] wrote %d chunks to %s and %s (dropped %d)", len(chunks), CHUNKS_FILE, PREVIEW_FILE, dropped
    )


def write_sources_csv(chunks: Sequence[Chunk], path: Path, *, cfg: AppConfig | None = None) -> None:
    """D2 source list: one row per source, with the loader meta timestamps and sha256 when available."""
    rows: dict[str, list[str]] = {}
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        rows.setdefault(
            chunk.source_id,
            [
                chunk.source_id,
                chunk.source_url,
                chunk.publisher,
                chunk.doc_type,
                chunk.scheme_name or "",
                chunk.scheme_category or "",
                chunk.plan or "",
                chunk.option or "",
                chunk.source_domain,
            ],
        )
        chunk_counts[chunk.source_id] = chunk_counts.get(chunk.source_id, 0) + 1
    meta = _loader_meta(chunks, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCES_CSV_HEADER)
        for source_id, row in rows.items():
            fetched_at, sha256 = meta.get(source_id, ("", ""))
            writer.writerow([*row, fetched_at, fetched_at, sha256, chunk_counts[source_id], fetched_at])
    os.replace(temp, path)


def _build_chunk(
    *,
    text: str,
    section_heading: str,
    section_path: str,
    index: int,
    doc: CleanDocument,
    spec: SourceSpec,
    cfg: AppConfig,
    ingested_at: str,
) -> Chunk:
    token_count = count_tokens(text)
    if token_count > cfg.chunking.max_tokens:
        raise ChunkingError(
            f"{doc.source_id} chunk {index} of {len(text)} chars estimates {token_count} tokens, "
            f"above chunking.max_tokens={cfg.chunking.max_tokens}"
        )
    return Chunk(
        source_id=spec.id,
        source_url=spec.url,
        publisher=spec.publisher,
        source_domain=spec.domain,
        page_title=doc.page_title,
        doc_type=spec.doc_type,
        scheme_name=spec.scheme_name,
        scheme_category=spec.scheme_category,
        plan=spec.plan,
        option=spec.option,
        topic=classify_topic(section_path, text, cfg, scheme=spec.scheme_name),
        section_heading=section_heading,
        content_as_of=_content_as_of(text),
        chunk_index=index,
        text=text,
        ingested_at=ingested_at,
        token_count=token_count,
    )


def _sections(blocks: Sequence[Block]) -> list[list[Block]]:
    sections: list[list[Block]] = []
    for block in blocks:
        if not sections or sections[-1][-1].heading_path != block.heading_path:
            sections.append([block])
        else:
            sections[-1].append(block)
    return sections


def _section_pieces(
    section: Sequence[Block], size: int, overlap: int, max_tokens: int
) -> list[tuple[str, str]]:
    """Split one section into table row-group chunks and prose chunks, keeping document order."""
    section_heading = _section_heading(section)
    pieces: list[tuple[str, str]] = []
    prose: list[Block] = []
    rows: list[Block] = []
    for block in section:
        if block.row_group is None:
            if rows:
                pieces.extend(_table_pieces(rows, section_heading, size))
                rows = []
            prose.append(block)
        else:
            if prose:
                pieces.extend(_prose_pieces(prose, section_heading, size, overlap, max_tokens))
                prose = []
            rows.append(block)
    if rows:
        pieces.extend(_table_pieces(rows, section_heading, size))
    if prose:
        pieces.extend(_prose_pieces(prose, section_heading, size, overlap, max_tokens))
    return pieces


def _section_heading(section: Sequence[Block]) -> str:
    path = section[-1].heading_path
    if path:
        return path.split(HEADING_SEPARATOR)[-1]
    first = section[0]
    return first.text if first.kind == "heading" else ""


def _prose_pieces(
    blocks: Sequence[Block], section_heading: str, size: int, overlap: int, max_tokens: int
) -> list[tuple[str, str]]:
    text = BLOCK_SEPARATOR.join(block.text for block in blocks)
    if len(text) <= size:
        return [(text, section_heading)]
    cap = max_tokens * CHARS_PER_TOKEN
    pieces: list[tuple[str, str]] = []
    for start, end in _windows(text, size, overlap, cap):
        window = text[start:end].strip()
        if window:
            pieces.append((window, section_heading))
    return pieces


def _table_pieces(rows: Sequence[Block], section_heading: str, size: int) -> list[tuple[str, str]]:
    """One chunk per row group, the cleaner's header prefix repeated on every row, never split mid-row."""
    pieces: list[tuple[str, str]] = []
    for group in _row_groups(rows):
        current: list[str] = []
        for block in group:
            candidate = BLOCK_SEPARATOR.join([*current, block.text])
            if current and len(candidate) > size:
                pieces.append((BLOCK_SEPARATOR.join(current), section_heading))
                current = []
            current.append(block.text)
        if current:
            pieces.append((BLOCK_SEPARATOR.join(current), section_heading))
    return pieces


def _row_groups(rows: Sequence[Block]) -> list[list[Block]]:
    groups: list[list[Block]] = []
    for block in rows:
        if groups and groups[-1][0].row_group == block.row_group:
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def _windows(text: str, size: int, overlap: int, cap: int) -> list[tuple[int, int]]:
    if len(text) <= size:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    start = 0
    end = _window_end(text, start, size, overlap, cap)
    while True:
        spans.append((start, end))
        if end >= len(text):
            return spans
        raw_start = max(start + 1, end - overlap)
        snapped = _snap_backward(text, raw_start)
        start = snapped if snapped is not None and snapped > start else raw_start
        end = _window_end(text, start, size, overlap, cap)
        if end <= start:
            return spans


def _window_end(text: str, start: int, size: int, overlap: int, cap: int) -> int:
    """Forward sentence snap inside the token cap, else the previous boundary, else the raw window.

    `size` is the target; a sentence boundary found between `size` and `cap` wins, because §6.3.4
    treats the token cap as the ceiling. A protected span (a number or date) is then completed or
    started, never split. §6.3.4's backward step exists to avoid cutting mid-sentence, but on a
    run-on section it would shrink the window to a single early sentence and then creep one
    character per window, which §6.3.8's max-chunks guard must catch as a cleaning failure, not
    a chunking style, so it is only taken while it still leaves a full non-overlapping window.
    """
    limit = min(len(text), start + size)
    hard_cap = min(len(text), start + cap)
    end = _snap_forward(text, limit, hard_cap)
    if end is None:
        backward = _snap_backward(text, limit)
        end = backward if backward is not None and backward - start >= size - overlap else limit
    end = _protected_end(text, end, hard_cap)
    if end > hard_cap:
        end = _protected_start(text, hard_cap, start)
    if end <= start:
        end = min(len(text), start + max(1, size - overlap))
    return end


def _snap_forward(text: str, lower: int, limit: int) -> int | None:
    for index in range(max(lower, 1), min(limit + 1, len(text))):
        if text[index - 1] in BOUNDARY_MARKS:
            return index + 1
    return None


def _snap_backward(text: str, upper: int) -> int | None:
    for index in range(min(upper, len(text)), 0, -1):
        if text[index - 1] in BOUNDARY_MARKS:
            return index
    return None


def _protected_end(text: str, index: int, limit: int) -> int:
    """Complete the protected span a boundary falls inside; start it instead when that would pass `limit`."""
    match = _protected_at(text, index)
    if match is None:
        return index
    if match.end() <= limit:
        return match.end()
    return max(match.start(), 0)


def _protected_start(text: str, index: int, floor: int) -> int:
    """Start of the protected span a boundary falls inside, so the cap never lands mid-number."""
    match = _protected_at(text, index)
    if match is None or match.start() < floor:
        return index
    return match.start()


def _protected_at(text: str, index: int) -> re.Match[str] | None:
    for match in _PROTECTED.finditer(text):
        if match.start() < index < match.end():
            return match
        if match.start() >= index:
            return None
    return None


def _apply_page_as_of(chunks: list[Chunk], page_as_of: str) -> list[Chunk]:
    """Copy the page's newest source date onto chunks that do not carry their own stamp.

    Groww scheme pages stamp the NAV line (`NAV: 29 Sep '26`) and leave fee/SIP blocks undated.
    The page date is the source's last-updated value for every fact taken from that page.
    """
    dates = [chunk.content_as_of for chunk in chunks if chunk.content_as_of]
    if page_as_of:
        dates.append(page_as_of)
    newest = max(dates) if dates else ""
    if not newest:
        return chunks
    return [replace(chunk, content_as_of=chunk.content_as_of or newest) for chunk in chunks]


def _content_as_of(text: str) -> str:
    """Most recent `as on` / `Last updated` / Groww `NAV:` stamp in the text, ISO-8601, else empty."""
    dates: list[str] = []
    for match in _DATE_STAMP.finditer(text):
        stamp = next((group for group in match.groups() if group), None)
        iso = _iso_date(stamp) if stamp else None
        if iso:
            dates.append(iso)
    return max(dates) if dates else ""


def _iso_date(stamp: str) -> str | None:
    match = re.match(
        r"(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+['’]?(\d{2}|\d{4})\b",
        stamp.strip(),
    )
    if match is None:
        return None
    month = _MONTHS.get(match.group(2)[:3].lower())
    if month is None:
        return None
    year = int(match.group(3))
    if year < 100:
        year += 2000
    return f"{year:04d}-{month:02d}-{int(match.group(1)):02d}"


def _render_preview(chunks: Sequence[Chunk], dropped: int) -> str:
    lines: list[str] = []
    for number, chunk in enumerate(chunks):
        lines.append(
            f"[{number}] {chunk.source_url} | {chunk.doc_type} | {chunk.topic} | {chunk.section_heading} | "
            f"chars={chunk.char_len} tokens={chunk.token_count}"
        )
        lines.append(chunk.text)
        lines.append("---")
    topics: dict[str, int] = {}
    sources: dict[str, int] = {}
    for chunk in chunks:
        topics[chunk.topic] = topics.get(chunk.topic, 0) + 1
        sources[chunk.source_id] = sources.get(chunk.source_id, 0) + 1
    lines.append("footer")
    lines.append(f"chunks: {len(chunks)}")
    lines.append(f"dropped_chunks: {dropped}")
    lines.append(f"chars_total: {sum(chunk.char_len for chunk in chunks)}")
    lines.append(f"tokens_total: {sum(chunk.token_count for chunk in chunks)}")
    lines.append("topics: " + ", ".join(f"{topic}={count}" for topic, count in sorted(topics.items())))
    lines.append("sources: " + ", ".join(f"{source}={count}" for source, count in sorted(sources.items())))
    return "\n".join(lines) + "\n"


def _loader_meta(chunks: Sequence[Chunk], cfg: AppConfig | None) -> dict[str, tuple[str, str]]:
    if cfg is None:
        return {}
    meta: dict[str, tuple[str, str]] = {}
    for chunk in chunks:
        if chunk.source_id in meta:
            continue
        path = meta_path(
            SourceSpec(
                id=chunk.source_id,
                url=chunk.source_url,
                domain=chunk.source_domain,
                doc_type=chunk.doc_type,
                scheme_name=chunk.scheme_name,
                scheme_category=chunk.scheme_category,
                plan=chunk.plan,
                option=chunk.option,
                publisher=chunk.publisher,
            ),
            cfg,
        )
        meta[chunk.source_id] = _meta_fields(path)
    return meta


def _meta_fields(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    record = json.loads(path.read_text(encoding="utf-8"))
    return str(record.get("fetched_at", "")), str(record.get("sha256", ""))


def _write_atomic(path: Path, payload: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
