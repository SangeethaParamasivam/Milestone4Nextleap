from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import trafilatura
from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

from app.models import Block, CleanDocument, SourceSpec
from config import DEFAULT_CHUNKING, AppConfig

_LOG = logging.getLogger("ingest.cleaner")

DEFAULT_MIN_PAGE_CHARS = int(DEFAULT_CHUNKING["min_page_chars"])
MIN_BLOCK_CHARS = 3
REJECT_BELOW_MIN_PAGE_CHARS = "below_min_page_chars"
EXTRACTION_TRAFILATURA = "trafilatura"
EXTRACTION_STRUCTURAL = "bs4_lxml_structural"
NEXT_DATA_SCRIPT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
LOCK_IN_OBJECT = re.compile(r'"lock_in"\s*:\s*\{\s*"years"\s*:\s*(\d+)\s*,\s*"months"\s*:\s*(\d+)\s*,\s*"days"\s*:\s*(\d+)\s*\}')
FUND_FACT_LABELS = ("NAV", "Min. for SIP", "Fund size (AUM)", "Expense ratio")
_FUND_FACT_SPLIT_RE = re.compile(
    "(?=" + "|".join(re.escape(label) for label in FUND_FACT_LABELS) + ")"
)
_FUND_FACT_LABEL_RE = re.compile(r"([A-Za-z][^0-9]*?)\s*(?:[:\d₹]|$)")
_FUND_FACT_RATING_RE = re.compile(r"\s+Rating\s*\d+$")
_FUND_FACT_HEADING = "Fund facts"

REMOVED_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg"}
)
NOISE_PATTERN = re.compile(
    r"cookie|consent|banner|advert|ad-|\bads?\b|breadcrumb|popup|modal|skip-link|sr-only", re.IGNORECASE
)
HEADING_LEVELS: dict[str, int] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
TABLE_TAGS = frozenset({"table"})
ROW_TAGS = ("tr", "row")
CELL_TAGS = ("td", "th", "cell")
STRUCTURAL_TAGS = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "table", "dl", "blockquote", "figcaption"}
)
LABEL_SOURCE_TAGS = ("nav", "header", "footer", "aside", "menu")
MAIN_REGION_PATTERN = re.compile(r"(?:^|[-_])main(?:$|[-_])", re.IGNORECASE)

_DASHES = {"—": "-", "–": "-", "−": "-", "‒": "-", "―": "-"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}
_SPACES = {" ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " "}
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_WHITESPACE = re.compile(r"\s+")
_TRANSLATION = str.maketrans({**_DASHES, **_QUOTES, **_SPACES, "\u2026": "..."})
_ZERO_WIDTH_TO_SPACE = {ord(char): " " for char in _ZERO_WIDTH}


@dataclass(frozen=True)
class _Extraction:
    blocks: list[Block]
    element_headings: int
    chars: int


def normalise_text(text: str) -> str:
    """Apply the FR-2.3 fold in the frozen order; numerals, units, `%`, `Rs.` and dates pass through.

    Zero-width characters become a space rather than disappearing, so a hidden separator inside
    `as on 30 Sep 2026` cannot weld two tokens together; the trailing collapse removes the space.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(_TRANSLATION)
    folded = folded.translate(_ZERO_WIDTH_TO_SPACE)
    return _WHITESPACE.sub(" ", folded).strip()


def is_rejected(doc: CleanDocument, cfg: AppConfig) -> bool:
    """Single owner of the FR-2.5 fail-loud test, shared by `clean_file` and `run_ingest.py`."""
    return doc.extracted_chars < cfg.chunking.min_page_chars


def clean_html(html: str, spec: SourceSpec, *, min_chars: int = DEFAULT_MIN_PAGE_CHARS) -> CleanDocument:
    """Extract ordered semantic blocks from one page, trafilatura first and BS/lxml as fallback."""
    soup, blocks, _ = _select_blocks(html, spec, min_chars=min_chars)
    return _build_document(soup, spec, blocks)


def clean_file(path: Path, spec: SourceSpec, cfg: AppConfig) -> CleanDocument:
    """Clean one cached raw page, write the M2 block dump, and log a reject when under the floor."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup, blocks, extraction = _select_blocks(html, spec, min_chars=cfg.chunking.min_page_chars)
    doc = _build_document(soup, spec, blocks)
    _write_block_dump(doc, cfg, extraction)
    _LOG.info(
        "[clean] %s extraction=%s blocks=%d chars=%d", doc.source_id, extraction, len(doc.blocks), doc.extracted_chars
    )
    if is_rejected(doc, cfg):
        write_reject_log(cfg, doc.source_id, doc.extracted_chars, REJECT_BELOW_MIN_PAGE_CHARS)
    return doc


def write_reject_log(cfg: AppConfig, source_id: str, extracted_chars: int, reason: str) -> None:
    """Append one FR-2.5 reject record to the append-only `data/rejected.jsonl`."""
    record = {
        "source_id": source_id,
        "extracted_chars": extracted_chars,
        "reason": reason,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = cfg.paths.rejected_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    _LOG.info("[clean] %s rejected reason=%s chars=%d", source_id, reason, extracted_chars)


def dump_blocks(doc: CleanDocument) -> str:
    """Render the M2 exit-review artefact; `clean_file` writes it under `cfg.paths.chunks_dir`."""
    lines = [
        f"source_id: {doc.source_id}",
        f"url: {doc.url}",
        f"page_title: {doc.page_title}",
        f"blocks: {len(doc.blocks)}",
        f"extracted_chars: {doc.extracted_chars}",
        "",
    ]
    for index, block in enumerate(doc.blocks):
        lines.append(f"[{index}] {block.kind} | {block.heading_path} | {len(block.text)} chars")
        if block.row_group is not None:
            lines[-1] += f" | row_group={block.row_group}"
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_document(soup: BeautifulSoup, spec: SourceSpec, blocks: Sequence[Block]) -> CleanDocument:
    kept = tuple(block for block in blocks if len(block.text) >= MIN_BLOCK_CHARS)
    return CleanDocument(
        source_id=spec.id,
        url=spec.url,
        page_title=_page_title(soup),
        blocks=kept,
        extracted_chars=sum(len(block.text) for block in kept),
    )


def _select_blocks(html: str, spec: SourceSpec, *, min_chars: int) -> tuple[BeautifulSoup, list[Block], str]:
    """Choose the trafilatura main-content blocks unless that selection is unusable for this page.

    A trafilatura selection is unusable when it is shorter than `min_chars`, or when it carries no
    `h1`-`h6` at all while the source document does: boilerplate heuristics tuned for prose articles
    strip the section structure of JS-rendered scheme pages, which is a failed main-content selection
    rather than a page whose sections do not exist.
    """
    soup = _parse(html)
    noise_labels = _noise_labels(soup)
    structural = _BlockEmitter(noise_labels).extract(_content_root(soup))
    primary = _trafilatura_blocks(html, noise_labels)
    if primary is not None and _primary_is_usable(primary, structural, min_chars):
        blocks, extraction = primary.blocks, EXTRACTION_TRAFILATURA
    else:
        blocks, extraction = structural.blocks, EXTRACTION_STRUCTURAL
    blocks = _split_fund_facts(blocks, spec)
    blocks.extend(_scheme_fact_blocks(html, spec))
    return soup, blocks, extraction


def _parse(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _content_root(soup: BeautifulSoup) -> Tag:
    """Main-content region of the page, falling back to `<body>` when the page marks none.

    These scheme pages build their site header, mega menu and footer out of `div`s, so the §5.3
    removal set sees no `nav`/`footer` tag to drop. Walking the main region instead of the whole
    body removes that chrome without guessing at class names, and it keeps every block of the
    main column even when the page is mostly navigation.
    """
    body = soup.body or soup
    best: Tag | None = None
    best_depth = -1
    for tag in body.find_all(True):
        if not _marks_main_region(tag):
            continue
        depth = len(list(tag.parents))
        if depth > best_depth:
            best, best_depth = tag, depth
    return best if best is not None else body


def _marks_main_region(tag: Tag) -> bool:
    if tag.name == "main" or str(tag.get("role", "")).lower() == "main":
        return True
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    return any(MAIN_REGION_PATTERN.search(str(token)) for token in classes)


def _page_title(soup: BeautifulSoup) -> str:
    heading = soup.find(["h1", "h2"])
    if heading is not None:
        title = normalise_text(heading.get_text(" ", strip=True))
        if title:
            return title
    if soup.title is not None and soup.title.string is not None:
        return normalise_text(str(soup.title.string))
    return ""


def _noise_labels(soup: BeautifulSoup) -> frozenset[str]:
    """Stripped texts of nav/header containers, used to drop verbatim repeats elsewhere."""
    labels: set[str] = set()
    for tag in soup.find_all(LABEL_SOURCE_TAGS):
        text = normalise_text(tag.get_text(" ", strip=True)).lower()
        if text:
            labels.add(text)
    for tag in soup.find_all(True):
        if tag.name in REMOVED_TAGS:
            continue
        blob = _attribute_blob(tag)
        if blob and NOISE_PATTERN.search(blob):
            text = normalise_text(tag.get_text(" ", strip=True)).lower()
            if text:
                labels.add(text)
    return frozenset(labels)


def _attribute_blob(tag: Tag) -> str:
    return " ".join(str(tag.get(attr, "")) for attr in ("role", "aria-label", "class", "id"))


def _has_structural(tag: Tag) -> bool:
    return any(getattr(node, "name", None) in STRUCTURAL_TAGS for node in tag.descendants)


def _trafilatura_blocks(html: str, noise_labels: frozenset[str]) -> _Extraction | None:
    try:
        document = trafilatura.bare_extraction(
            html, output_format="xml", include_tables=True, include_comments=False
        )
    except Exception as exc:
        _LOG.info("[clean] trafilatura failed: %s: %s", type(exc).__name__, exc)
        return None
    if document is None:
        return None
    body = getattr(document, "body", None)
    if body is None:
        return None
    return _BlockEmitter(noise_labels).extract(_parse(f"<div>{etree.tostring(body, encoding='unicode')}</div>"))


def _primary_is_usable(primary: _Extraction, structural: _Extraction, min_chars: int) -> bool:
    if primary.chars < min_chars:
        return False
    return primary.element_headings > 0 or structural.element_headings == 0


class _BlockEmitter:
    """Document-order block emitter that keeps the ancestor heading chain on every block."""

    def __init__(self, noise_labels: frozenset[str]) -> None:
        self._noise_labels = noise_labels
        self.blocks: list[Block] = []
        self.element_headings = 0
        self._row_group = -1
        self._stack: list[tuple[int, str]] = []

    def extract(self, root: Tag) -> _Extraction:
        self._walk(root)
        return _Extraction(self.blocks, self.element_headings, sum(len(b.text) for b in self.blocks))

    def _walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = normalise_text(str(child))
                if len(text) >= MIN_BLOCK_CHARS:
                    self.blocks.append(Block("para", self._path(), text, None))
                continue
            if not isinstance(child, Tag) or self._is_removed(child):
                continue
            name = child.name
            if name in HEADING_LEVELS:
                self._heading(HEADING_LEVELS[name], child)
            elif name == "li":
                self._text_block(child, "list_item")
            elif name == "dl":
                self.blocks.extend(self._definitions(child))
            elif name in TABLE_TAGS:
                self.blocks.extend(self._table(child))
            elif name == "p":
                self._text_block(child, "para")
            elif _has_structural(child):
                self._walk(child)
            else:
                self._text_block(child, "para")

    def _is_removed(self, tag: Tag) -> bool:
        if tag.name in REMOVED_TAGS:
            return True
        if NOISE_PATTERN.search(_attribute_blob(tag)):
            return True
        if self._noise_labels:
            text = normalise_text(tag.get_text(" ", strip=True)).lower()
            if text and text in self._noise_labels:
                return True
        return False

    def _path(self) -> str:
        return " > ".join(text for _, text in self._stack)

    def _text_block(self, tag: Tag, kind: str) -> None:
        text = normalise_text(tag.get_text(" ", strip=True))
        if text:
            self.blocks.append(Block(kind, self._path(), text, None))

    def _heading(self, level: int, tag: Tag) -> None:
        text = normalise_text(tag.get_text(" ", strip=True))
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        if text:
            self.element_headings += 1
            self.blocks.append(Block("heading", self._path(), text, None))
            self._stack.append((level, text))

    def _definitions(self, tag: Tag) -> list[Block]:
        blocks: list[Block] = []
        term = ""
        for item in tag.find_all(("dt", "dd")):
            text = normalise_text(item.get_text(" ", strip=True))
            if not text:
                continue
            if item.name == "dt":
                term = text
                continue
            blocks.append(Block("definition", self._path(), f"{term}: {text}" if term else text, None))
            term = ""
        return blocks

    def _table(self, table: Tag) -> list[Block]:
        blocks: list[Block] = []
        header: list[str] = []
        for row in table.find_all(ROW_TAGS):
            cells = row.find_all(CELL_TAGS, recursive=False) or row.find_all(CELL_TAGS)
            texts = [normalise_text(cell.get_text(" ", strip=True)) for cell in cells]
            if not any(texts):
                continue
            if not header or self._is_header_row(row, cells):
                header = texts
                blocks.append(Block("heading", self._path(), " | ".join(texts), None))
                self._row_group += 1
                continue
            blocks.append(Block("table_row", self._path(), " | ".join(header + texts), self._row_group))
        return blocks

    @staticmethod
    def _is_header_row(row: Tag, cells: Sequence[Tag]) -> bool:
        return bool(cells) and all(
            cell.name == "th" or (cell.name == "cell" and str(cell.get("role", "")) == "head")
            for cell in cells
        )


def _write_block_dump(doc: CleanDocument, cfg: AppConfig, extraction: str) -> None:
    path = cfg.paths.chunks_dir / f"blocks_{doc.source_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"extraction_path: {extraction}\n\n{dump_blocks(doc)}".encode("utf-8")
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _split_fund_facts(blocks: Sequence[Block], spec: SourceSpec) -> list[Block]:
    """Split the Groww fund snapshot line into one small fact block per labelled field.

    The sticky page snapshot is extracted as a single root paragraph that reads
    "NAV: ... Min. for SIP ... Fund size (AUM) ... Expense ratio ... Rating ...". As one chunk
    it buries every fact: measured over the real corpus, the fund-size figure sat out of the
    top-5 for a fund-size question and its chunk was only reachable at rank ~11, so the
    generator answered from training memory instead of the chunks. Splitting each labelled
    field into its own block, suffixed with the scheme name, gives every field a clean lexical
    and semantic surface and its own topic label later. Graph-axis fragments and the `Rating`
    chip (server renders the riskometer as "Rating <n>" only) are dropped with the split.
    """
    out: list[Block] = []
    for block in blocks:
        if block.kind != "para" or block.heading_path.strip():
            out.append(block)
            continue
        segments = [seg for seg in _FUND_FACT_SPLIT_RE.split(normalise_text(block.text)) if seg.strip()]
        facts = [seg.strip(" .") for seg in segments if seg.strip().startswith(FUND_FACT_LABELS)]
        if len(facts) < 2:
            out.append(block)
            continue
        suffix = spec.scheme_name or spec.id
        for fact in facts:
            fact = _FUND_FACT_RATING_RE.sub("", fact)
            label = _fund_fact_label(fact)
            text = normalise_text(f"{fact} ({suffix})")
            out.append(Block("para", f"{_FUND_FACT_HEADING} > {label}", text, None))
    return out


def _fund_fact_label(fact: str) -> str:
    """The human label before the value: `Min. for SIP ₹100` -> `Min. for SIP`."""
    match = _FUND_FACT_LABEL_RE.match(fact)
    return (match.group(1).strip() if match else fact[:24]).rstrip(".:")


def _scheme_fact_blocks(html: str, spec: SourceSpec) -> list[Block]:
    """Groww scheme pages carry the scheme facts server-side in `__NEXT_DATA__`.

    The visible lock-in period is only rendered through that JSON (the on-page chip sits under a
    `header` that the cleaner removes as chrome), so extract just the verifiable lock-in fact as a
    `Fund details` block. The `years>0`-style guard keeps non-ELSS pages (whose `lock_in` object is
    `years: null`) from producing a block at all. This is a bounded, reviewable fact pull — not a
    general JSON dumper — and the only growth surface added here.
    """
    script = NEXT_DATA_SCRIPT.search(html)
    if script is None:
        return []
    lock = LOCK_IN_OBJECT.search(script.group(1))
    if lock is None:
        return []
    years, months, days = (int(value) for value in lock.groups())
    parts: list[str] = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if not parts:
        return []
    suffix = f" ({spec.scheme_name or spec.id})"
    return [Block("para", "Fund details", f"Lock-in period: {', '.join(parts)}.{suffix}", None)]
