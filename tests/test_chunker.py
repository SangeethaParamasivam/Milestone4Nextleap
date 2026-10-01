from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from app.models import Block, Chunk, CleanDocument, SourceSpec
from config import AppConfig, ConfigError
from ingest.chunker import (
    CHUNKS_FILE,
    ChunkingError,
    chunk_document,
    classify_topic,
    count_tokens,
    plan_chunks,
    write_chunks,
    write_sources_csv,
)
from ingest.topics import TOPIC_VOCABULARY

SENTENCE = (
    "The expense ratio of the scheme is 1.03% as on 30 Sep 2026 and the NAV was ₹1,165.78. "
)
LONG_SECTION = SENTENCE * 12


def _sections_doc(blocks: list[tuple[str, str]], *, page_title: str = "HDFC Large Cap Fund - Direct Growth") -> CleanDocument:
    chosen = tuple(
        Block(kind="para" if index % 2 == 0 else "heading", heading_path=path, text=text, row_group=None)
        for index, (path, text) in enumerate(blocks)
    )
    return CleanDocument(
        source_id="s1",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        page_title=page_title,
        blocks=chosen,
        extracted_chars=sum(len(block.text) for block in chosen),
    )


def _sentence_chunks(cfg: AppConfig, source_spec_factory) -> list[Chunk]:
    doc = _sections_doc([("", LONG_SECTION)])
    return chunk_document(doc, source_spec_factory(), cfg)


def test_every_chunk_respects_the_token_bound(tmp_config: AppConfig, source_spec_factory) -> None:
    doc = _sections_doc(
        [
            ("", SENTENCE * 12),
            ("Fees > Exit load", "Exit load of 1% if redeemed within 1 year."),
            ("Fees", "Expense ratio 0.5% as on 30 Sep 2026."),
        ]
    )
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert chunks
    for chunk in chunks:
        assert count_tokens(chunk.text) <= tmp_config.chunking.max_tokens
        assert chunk.token_count == max(1, math.ceil(chunk.char_len / 4))


def test_chunk_ids_are_deterministic_for_the_same_document(tmp_config: AppConfig, source_spec_factory) -> None:
    doc = _sections_doc([("", SENTENCE * 12), ("Fees", "Expense ratio 0.5% as on 30 Sep 2026.")])
    first = chunk_document(doc, source_spec_factory(), tmp_config)
    second = chunk_document(doc, source_spec_factory(), tmp_config)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_content_addressing_changes_only_the_edited_chunk(tmp_config: AppConfig, source_spec_factory) -> None:
    blocks = [
        ("Fees > Exit load", "Exit load of 1% if redeemed within 1 year."),
        ("Benchmark", "The benchmark is NIFTY 100 Total Return Index."),
        ("Objective", "The scheme seeks long-term capital appreciation."),
    ]
    spec = source_spec_factory()
    before = chunk_document(_sections_doc(blocks), spec, tmp_config)
    assert len(before) == 3
    edited = list(blocks)
    edited[1] = ("Benchmark", "The benchmark is NIFTY 50 Total Return Index.")
    after = chunk_document(_sections_doc(edited), spec, tmp_config)
    assert [chunk.chunk_id for chunk in after if chunk.section_heading == "Benchmark"] != [
        chunk.chunk_id for chunk in before if chunk.section_heading == "Benchmark"
    ]
    assert [chunk.chunk_id for chunk in after if chunk.section_heading != "Benchmark"] == [
        chunk.chunk_id for chunk in before if chunk.section_heading != "Benchmark"
    ]


def test_table_row_group_keeps_every_row_and_repeats_the_header(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    rows = [
        Block("heading", "Holdings", "Name | Weight", None),
        Block("table_row", "Holdings", "Name | Weight | ICICI Bank Ltd | 10.05%", 0),
        Block("table_row", "Holdings", "Name | Weight | HDFC Bank Ltd | 6.88%", 0),
        Block("table_row", "Holdings", "Name | Weight | Infosys Ltd | 2.76%", 0),
    ]
    doc = CleanDocument("s1", "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", "Fund", tuple(rows), 0)
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert len(chunks) == 1
    assert chunks[0].text.split("\n") == [row.text for row in rows[1:]]
    assert chunks[0].text.count("Name | Weight") == 3
    assert chunks[0].section_heading == "Holdings"


def test_large_table_row_group_splits_by_rows_and_never_mid_row(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    rows = [Block("table_row", "Holdings", f"Name | Weight | Fund number {n} | {n}.05%", 0) for n in range(40)]
    doc = CleanDocument("s1", "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", "Fund", tuple(rows), 0)
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert len(chunks) > 1
    assert sum(chunk.text.count("Fund number") for chunk in chunks) == 40
    for chunk in chunks:
        for line in chunk.text.split("\n"):
            assert re.fullmatch(r"Name \| Weight \| Fund number \d+ \| \d+\.\d+%", line)
            assert chunk.char_len <= tmp_config.chunking.size


def test_no_chunk_splits_a_number_or_an_as_on_date(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    doc = _sections_doc([("", SENTENCE * 12)])
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text in doc.blocks[0].text
        assert chunk.text.count("\u20b9") == len(re.findall(r"\u20b9\s?\d+(?:[.,]\d+)*", chunk.text))
        assert chunk.text.count("%") == len(re.findall(r"\d+(?:[.,]\d+)*\s?%", chunk.text))
        for match in re.finditer(r"as on ", chunk.text):
            assert re.match(r"as on \d{1,2} [A-Za-z]{3,9} \d{4}", chunk.text[match.start() :])
    assert doc.blocks[0].text.strip().endswith(chunks[-1].text)
    for stamp in ("1.03%", "\u20b91,165.78", "as on 30 Sep 2026"):
        assert stamp in "\n".join(chunk.text for chunk in chunks)
    assert {chunk.content_as_of for chunk in chunks} == {"2026-09-30"}


def test_labelled_block_below_size_yields_one_anchored_chunk(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    doc = _sections_doc([("Fees > Exit load", "Exit load of 1% if redeemed within 1 year.")])
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section_heading == "Exit load"
    assert chunk.text == "Exit load of 1% if redeemed within 1 year."
    assert chunk.content_as_of == ""
    assert chunk.chunk_index == 0


def test_groww_nav_date_is_copied_onto_undated_chunks_on_the_same_page(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    doc = _sections_doc(
        [
            ("NAV", "NAV: 29 Sep '26 ₹1,165.78 (HDFC Large Cap Fund - Direct Growth)"),
            ("Fees > Expense ratio", "Expense ratio 1.03% (HDFC Large Cap Fund - Direct Growth)"),
        ]
    )
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert {chunk.content_as_of for chunk in chunks} == {"2026-09-29"}
    assert any(chunk.section_heading == "Expense ratio" for chunk in chunks)


def test_four_digit_year_is_not_truncated_to_two_digits(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    r"""Regression: the year alternation listed `\d{2}` before `\d{4}`, so a real
    four-digit year was captured as its first two digits and "29 Sep 2026" was
    stamped 2020-09-29 instead of 2026-09-29.
    """
    doc = _sections_doc(
        [
            ("NAV", "Latest NAV as of 29 Sep 2026 is \u20b91,165.78 for the direct growth plan."),
            ("Fees > Expense ratio", "Expense ratio 1.03% (HDFC Large Cap Fund - Direct Growth)"),
        ]
    )
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert {chunk.content_as_of for chunk in chunks} == {"2026-09-29"}


def test_two_digit_year_with_apostrophe_still_resolves(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    doc = _sections_doc(
        [
            ("NAV", "NAV: 29 Sep \u201926 \u20b91,165.78 (HDFC Large Cap Fund - Direct Growth)"),
            ("Fees > Expense ratio", "Expense ratio 1.03% (HDFC Large Cap Fund - Direct Growth)"),
        ]
    )
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert {chunk.content_as_of for chunk in chunks} == {"2026-09-29"}


def test_hyphenated_future_maturity_date_is_not_treated_as_last_updated(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    """A maturity date like `18-May-2066` must never be reported as the page's
    last-updated value, so the hyphenated form stays unsupported on purpose.
    """
    doc = _sections_doc(
        [
            ("NAV", "Latest NAV as of 29 Sep 2026 is \u20b91,165.78 for the direct growth plan."),
            ("Debt > Maturity", "Maturity date 18-May-2066 for the floating rate income fund."),
        ]
    )
    chunks = chunk_document(doc, source_spec_factory(), tmp_config)
    assert {chunk.content_as_of for chunk in chunks} == {"2026-09-29"}


def test_consecutive_windows_overlap_on_a_whole_sentence(
    tmp_config: AppConfig, source_spec_factory
) -> None:
    chunks = _sentence_chunks(tmp_config, source_spec_factory)
    cap = tmp_config.chunking.max_tokens * 4
    assert len(chunks) >= 2
    sentences = {chunk.text.strip() for chunk in chunks}
    for previous, following in zip(chunks, chunks[1:]):
        assert previous.char_len <= cap
        assert following.char_len <= cap
        shared = _sentences(previous.text) & _sentences(following.text)
        assert shared, (previous.text[-120:], following.text[:120])
        if following is not chunks[-1]:
            assert following.char_len >= tmp_config.chunking.size - tmp_config.chunking.overlap
    assert len(sentences) == len(chunks)


def test_section_under_min_chars_is_dropped_and_counted(tmp_config: AppConfig, source_spec_factory) -> None:
    doc = _sections_doc(
        [
            ("Fees > Exit load", "Exit load of 1% if redeemed within 1 year."),
            ("Holdings > See all", "See All"),
            ("Holdings > Notes", "Rating 4"),
        ]
    )
    plan = plan_chunks(doc, source_spec_factory(), tmp_config)
    assert plan.dropped == 2
    assert [chunk.text for chunk in plan.chunks] == ["Exit load of 1% if redeemed within 1 year."]
    assert [chunk.section_heading for chunk in plan.chunks] == ["Exit load"]


def test_document_above_max_chunks_raises(tmp_config: AppConfig, source_spec_factory) -> None:
    blocks = [(f"Section {n}", f"Section {n} carries a sentence with a number {n}.05% in it.") for n in range(2_000)]
    doc = _sections_doc(blocks)
    with pytest.raises(ChunkingError, match="above max_chunks_per_page"):
        chunk_document(doc, source_spec_factory(), tmp_config)


def test_metadata_is_complete_and_scalars_only(tmp_config: AppConfig, source_spec_factory) -> None:
    chunks = _sentence_chunks(tmp_config, source_spec_factory)
    for index, chunk in enumerate(chunks):
        metadata = chunk.to_metadata()
        for key in (
            "chunk_id",
            "source_url",
            "doc_type",
            "topic",
            "publisher",
            "chunk_index",
            "char_len",
            "content_hash",
        ):
            assert metadata.get(key) not in (None, ""), key
        assert chunk.chunk_index == index
        assert chunk.content_hash == chunk.content_hash.strip().lower()
        assert not any(isinstance(value, (dict, list)) for value in metadata.values())


def test_artefact_round_trip(tmp_config: AppConfig, source_spec_factory) -> None:
    chunks = _sentence_chunks(tmp_config, source_spec_factory)
    write_chunks(chunks, tmp_config, dropped=3)
    path = tmp_config.paths.chunks_dir / CHUNKS_FILE
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == len(chunks)
    for record, chunk in zip(records, chunks):
        restored = Chunk.from_metadata(record, record["text"])
        metadata = restored.to_metadata()
        assert metadata["chunk_id"] == chunk.chunk_id
        assert metadata["content_hash"] == chunk.content_hash
        assert not any(isinstance(value, (dict, list)) for value in metadata.values())
    preview = (tmp_config.paths.chunks_dir / "chunks_preview.txt").read_text(encoding="utf-8")
    assert "dropped_chunks: 3" in preview
    assert "footer" in preview


def test_sources_csv_has_one_row_per_source(tmp_config: AppConfig, source_spec_factory) -> None:
    chunks = []
    for spec in tmp_config.sources:
        doc = _sections_doc([("", f"{spec.id} carries a sentence with a number 1.03% in it.")])
        chunks.extend(chunk_document(doc, spec, tmp_config))
    path = tmp_config.paths.chunks_dir / "sources.csv"
    write_sources_csv(chunks, path, cfg=tmp_config)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("source_id,url,publisher,doc_type,scheme_name")
    assert len(lines) == 1 + len(tmp_config.sources)
    for spec in tmp_config.sources:
        row = next(line for line in lines[1:] if line.startswith(f"{spec.id},"))
        assert spec.url in row
        assert spec.publisher in row
        assert row.endswith(",1," + row.split(",")[-1])


def test_topic_matching_is_word_boundary_aware(real_config: AppConfig) -> None:
    assert classify_topic("Fees", "The total expense is 0.5%", real_config) == "expense_ratio"
    assert classify_topic("Fees", "The TER of the scheme is 0.5%", real_config) == "expense_ratio"
    assert classify_topic("Post", "After the enterprise listing", real_config) == "general"
    assert classify_topic("Post", "It was 180c over two days", real_config) == "general"
    assert classify_topic("NAV", "Net asset value is ₹1,165.78", real_config) == "nav"
    assert classify_topic("Guide", "See the navigation menu", real_config) == "general"
    assert classify_topic("Tax", "Section 80c benefit", real_config) == "lockin_80c"
    assert classify_topic("Misc", "Nothing matches here", real_config) == "general"
    assert (
        classify_topic(
            "HDFC Small Cap Fund - Direct Growth > About HDFC Small Cap Fund Direct Growth",
            "Exit Load for units in excess of 15% of the investment, 1% will be charged for "
            "redemption within 1 year. Minimum SIP Investment is set to \u20b9100.",
            real_config,
        )
        == "general"
    )
    assert (
        classify_topic(
            "Investment Objective",
            "The scheme seeks long-term capital appreciation. Fund benchmark NIFTY 100 Total "
            "Return Index.",
            real_config,
        )
        == "benchmark"
    )
    assert (
        classify_topic(
            "Fund management > DM Dhruv Muchhal Jun 2023 - Present View details",
            "Also manages these schemes HDFC Flexi Cap Direct Plan Growth HDFC NIFTY 50 ETF",
            real_config,
        )
        == "general"
    )
    assert (
        classify_topic(
            "Compare similar funds",
            "Name | 1Y | 3Y | Fund Size(Cr) | ITI Small Cap Fund Direct Growth | 22.40%",
            real_config,
        )
        == "general"
    )


def test_classify_topic_ignores_the_owning_scheme_name(real_config) -> None:
    text = "Min. for SIP \u20b9500 (HDFC ELSS Tax Saver Fund - Direct Plan - Growth)"
    assert (
        classify_topic("Fund facts > Min. for SIP", text, real_config)
        == "plan_direct_vs_regular"
    )
    assert (
        classify_topic(
            "Fund facts > Min. for SIP",
            text,
            real_config,
            scheme="HDFC ELSS Tax Saver Fund - Direct Plan - Growth",
        )
        == "general"
    )


def test_topics_outside_the_controlled_vocabulary_are_rejected(tmp_config: AppConfig) -> None:
    from dataclasses import replace

    broken = replace(tmp_config, topics={**tmp_config.topics, "made_up_topic": ("anything",)})
    with pytest.raises(ConfigError):
        classify_topic("Fees", "Expense ratio 0.5%", broken)
    assert "general" in TOPIC_VOCABULARY


def _sentences(text: str) -> set[str]:
    return {part.strip() for part in re.split(r"(?<=\.)\s+", text.strip()) if part.strip()}
