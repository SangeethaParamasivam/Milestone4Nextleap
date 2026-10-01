from __future__ import annotations

import json
import re
from pathlib import Path

from app.models import CleanDocument, SourceSpec
from config import AppConfig
from ingest.cleaner import (
    REJECT_BELOW_MIN_PAGE_CHARS,
    clean_file,
    clean_html,
    dump_blocks,
    normalise_text,
)

FUND_HTML = """
<!doctype html>
<html lang="en">
<head>
  <title>HDFC Large Cap Fund - Direct Growth | Groww</title>
  <script>window.dataLayer = [{'page': 'Menu Scripts'}];</script>
  <style>.hidden { display: none; }</style>
</head>
<body>
  <nav>Menu</nav>
  <header>HDFC Large Cap Fund - Direct Growth</header>
  <aside>Related articles</aside>
  <form><input name="otp"/></form>
  <div class="cookie-banner">We use cookies. Accept all</div>
  <div class="advert-promo">Buy now, pay later</div>
  <div id="consentDialog">Manage consent</div>
  <div class="breadcrumbs">Home > Mutual Funds</div>
  <main>
    <h1>HDFC Large Cap Fund - Direct Growth</h1>
    <p>NAV as on 30 Sep 2026 is &#8377;1,165.78 with an expense ratio of 1.03% and a fund size of
       &#8377;39,933.36 Cr as on 30 Sep 2026.</p>
    <h2>Fees</h2>
    <h3>Exit load</h3>
    <p>Exit load of 1% if redeemed within 1 year, nil after 12 months.</p>
    <h2>Holdings</h2>
    <table>
      <tr><th>Name</th><th>Weight</th></tr>
      <tr><td>ICICI Bank Ltd</td><td>10.05%</td></tr>
      <tr><td>HDFC Bank Ltd</td><td>6.88%</td></tr>
    </table>
    <h2>Documents</h2>
    <ul>
      <li>Minimum SIP &#8377;500 as on 30 Sep 2026</li>
      <li>Minimum lump sum &#8377;1,00,000</li>
    </ul>
    <dl><dt>Benchmark</dt><dd>NIFTY 100 Total Return Index</dd></dl>
    <p>The fund is rated Very High risk and follows a large cap benchmark. Minimum SIP investment
       is set to &#8377;500 as on 30 Sep 2026 and the minimum lump sum investment is
       &#8377;1,00,000 as on 30 Sep 2026.</p>
    <footer>Contact Us</footer>
  </main>
  <script>console.log('Menu Footer')</script>
</body>
</html>
"""


def _texts(doc: CleanDocument) -> str:
    return "\n".join(block.text for block in doc.blocks)


def _numerals(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*%?", text)


def test_nav_header_footer_script_and_widgets_are_removed(source_spec_factory: SourceSpec) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    blob = _texts(doc)
    for chrome in (
        "Menu",
        "Related articles",
        "We use cookies",
        "Buy now, pay later",
        "Manage consent",
        "Home > Mutual Funds",
        "Contact Us",
        "otp",
        "window.dataLayer",
        "display: none",
    ):
        assert chrome not in blob, chrome
    assert "NAV as on 30 Sep 2026" in blob
    assert doc.page_title == "HDFC Large Cap Fund - Direct Growth"


def test_clean_html_extracts_a_lock_in_fact_from_next_data(source_spec_factory: SourceSpec) -> None:
    html = (
        '<html><body><main><h1>HDFC ELSS Tax Saver Fund - Direct Plan - Growth</h1>'
        '<p>Fund overview text.</p></main>'
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"lock_in":{"years":3,"months":0,"days":0}}</script></body></html>'
    )
    spec = source_spec_factory(source_id="s3", scheme_name="HDFC ELSS Tax Saver Fund - Direct Plan - Growth")
    doc = clean_html(html, spec)
    assert any(block.text.startswith("Lock-in period: 3 years") for block in doc.blocks)


def test_clean_html_ignores_a_null_lock_in(source_spec_factory: SourceSpec) -> None:
    html = (
        '<html><body><main><h1>HDFC Small Cap Fund - Direct Growth</h1>'
        '<p>Fund overview text.</p></main>'
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"lock_in":{"years":null,"months":null,"days":null}}</script></body></html>'
    )
    doc = clean_html(html, source_spec_factory(source_id="s4", scheme_name="HDFC Small Cap Fund - Direct Growth"))
    assert not any(block.text.startswith("Lock-in") for block in doc.blocks)


def test_fund_snapshot_is_split_into_per_field_blocks(source_spec_factory: SourceSpec) -> None:
    from ingest.cleaner import _split_fund_facts

    from app.models import Block

    snapshot = Block(
        "para",
        "",
        "NAV: 29 Sep '26 \u20b91,165.78 Min. for SIP \u20b9100 Fund size (AUM) "
        "\u20b939,933.36 Cr Expense ratio 1.03% Rating 4",
        None,
    )
    kept = Block("para", "", "+7.77 % 3Y annualised", None)
    spec = source_spec_factory(source_id="s1", scheme_name="HDFC Large Cap Fund - Direct Growth")
    blocks = _split_fund_facts([snapshot, kept], spec)
    facts = [b for b in blocks if b.heading_path.startswith("Fund facts")]
    assert [b.heading_path.split(" > ")[-1] for b in facts] == ["NAV", "Min. for SIP", "Fund size (AUM)", "Expense ratio"]
    assert any("Fund size (AUM) \u20b939,933.36 Cr (HDFC Large Cap Fund - Direct Growth)" == b.text for b in facts)
    assert all(b.text.endswith("(HDFC Large Cap Fund - Direct Growth)") for b in facts)
    assert not any("Rating" in b.text for b in facts)
    assert any(b.text == "+7.77 % 3Y annualised" for b in blocks)
    assert not any(b is snapshot for b in blocks)


def test_heading_path_is_the_ancestor_chain(source_spec_factory: SourceSpec) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    by_text = {block.text: block for block in doc.blocks}
    fees = by_text["Fees"]
    assert fees.kind == "heading"
    assert fees.heading_path == ""
    exit_load = next(block for block in doc.blocks if block.text == "Exit load")
    assert exit_load.heading_path == "Fees"
    para = next(block for block in doc.blocks if block.text.startswith("Exit load of 1%"))
    assert para.kind == "para"
    assert para.heading_path == "Fees > Exit load"
    sip_item = next(block for block in doc.blocks if block.text.startswith("Minimum SIP"))
    assert sip_item.kind == "list_item"
    assert sip_item.heading_path == "Documents"


def test_table_header_repeats_and_row_group_increments(source_spec_factory: SourceSpec) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    header_blocks = [block for block in doc.blocks if block.kind == "heading" and "Weight" in block.text]
    assert [block.text for block in header_blocks] == ["Name | Weight"]
    rows = [block for block in doc.blocks if block.kind == "table_row"]
    assert [block.text for block in rows] == [
        "Name | Weight | ICICI Bank Ltd | 10.05%",
        "Name | Weight | HDFC Bank Ltd | 6.88%",
    ]
    assert [block.row_group for block in rows] == [0, 0]
    second_table = (
        "<h2>A</h2><table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
        "<h2>B</h2><table><tr><th>Y</th></tr><tr><td>2</td></tr></table>"
    )
    multi = clean_html(second_table, source_spec_factory())
    assert [block.row_group for block in multi.blocks if block.kind == "table_row"] == [0, 1]


def test_row_group_resets_per_document_not_per_table(source_spec_factory: SourceSpec) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    groups = [block.row_group for block in doc.blocks if block.row_group is not None]
    assert groups == sorted(groups)
    assert len(set(groups)) == 1


def test_definition_block_joins_term_and_definition(source_spec_factory: SourceSpec) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    definition = next(block for block in doc.blocks if block.kind == "definition")
    assert definition.text == "Benchmark: NIFTY 100 Total Return Index"


def test_protected_spans_survive_normalisation_byte_for_byte() -> None:
    text = "1.5% | ₹500 | 1,00,000 | as on 30 Sep 2026 | 12.5 bps | Rs. 1.25 lakh | 1.03% | 0.005%"
    assert normalise_text(text) == text


def test_unicode_normalisation_folds_and_keeps_ordinals() -> None:
    folded = normalise_text("a\u00a0b\u200bc \u2014 \u2013 d\u2212e \u2018f\u2019 \u201cg\u201d h\u2026i\u202fj")
    assert folded == "a b c - - d-e 'f' \"g\" h...i j"
    assert normalise_text("  \u2014NAV\u2014  ") == "-NAV-"
    assert normalise_text("30\u00a0Sep\u200b2026") == "30 Sep 2026"


def test_short_page_writes_reject_log_with_reason(tmp_config: AppConfig, tmp_path: Path) -> None:
    spec = tmp_config.sources[0]
    raw = tmp_path / "raw" / spec.domain
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{spec.id}.html").write_text("<html><body><p>tiny</p></body></html>", encoding="utf-8")
    doc = clean_file(raw / f"{spec.id}.html", spec, tmp_config)
    assert doc.extracted_chars < tmp_config.chunking.min_page_chars
    records = [
        json.loads(line)
        for line in tmp_config.paths.rejected_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["source_id"] == spec.id
    assert records[0]["reason"] == REJECT_BELOW_MIN_PAGE_CHARS
    assert records[0]["extracted_chars"] == doc.extracted_chars
    assert records[0]["ts"].endswith("Z")


def test_clean_file_keeps_a_full_page_out_of_the_reject_log(
    tmp_config: AppConfig, tmp_path: Path
) -> None:
    spec = tmp_config.sources[0]
    raw = tmp_path / "raw" / spec.domain
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{spec.id}.html").write_text(FUND_HTML, encoding="utf-8")
    doc = clean_file(raw / f"{spec.id}.html", spec, tmp_config)
    assert doc.extracted_chars >= tmp_config.chunking.min_page_chars
    assert not tmp_config.paths.rejected_file.exists()
    dump = tmp_config.paths.chunks_dir / f"blocks_{spec.id}.txt"
    assert dump.exists()
    assert "extraction_path:" in dump.read_text(encoding="utf-8")


def test_extraction_is_lossless_for_every_numeral_in_document_order(
    source_spec_factory: SourceSpec,
) -> None:
    doc = clean_html(FUND_HTML, source_spec_factory())
    expected = _numerals(
        "NAV as on 30 Sep 2026 is 1,165.78 with an expense ratio of 1.03% and a fund size of "
        "39,933.36 Cr as on 30 Sep 2026. Exit load of 1% if redeemed within 1 year, nil after "
        "12 months. ICICI Bank Ltd 10.05% HDFC Bank Ltd 6.88% Minimum SIP 500 as on 30 Sep 2026 "
        "Minimum lump sum 1,00,000"
    )
    found = _numerals(_texts(doc))
    position = 0
    for numeral in expected:
        position = found.index(numeral, position) + 1
    assert position == len(found) or expected


def test_short_blocks_below_three_characters_are_dropped(source_spec_factory: SourceSpec) -> None:
    doc = clean_html("<main><h2>Fees</h2><p>-</p><p>Exit load of 1%</p></main>", source_spec_factory())
    assert [block.text for block in doc.blocks] == ["Fees", "Exit load of 1%"]


def test_dump_blocks_renders_index_kind_path_and_row_group(source_spec_factory: SourceSpec) -> None:
    rendered = dump_blocks(clean_html(FUND_HTML, source_spec_factory()))
    assert rendered.startswith("source_id: s1")
    assert "page_title: HDFC Large Cap Fund - Direct Growth" in rendered
    assert "extracted_chars:" in rendered
    assert re.search(r"\[0\] para \|  \| \d+ chars", rendered)
    assert re.search(r"\[\d+\] heading \|  \| 4 chars\nFees", rendered)
    assert re.search(r"\[2\] heading \| Fees \| 9 chars\nExit load", rendered)
    assert re.search(r"\| row_group=0", rendered)
