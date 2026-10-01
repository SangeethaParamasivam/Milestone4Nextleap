from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import pytest

from app.models import Chunk, Decision, RetrievalHit
from app.retriever import (
    build_where,
    fuse,
    resolve_scheme,
    resolve_topic,
    retrieve,
    scheme_aliases,
)
from config import REPO_ROOT, AppConfig

GOLDEN_FILE = REPO_ROOT / "data" / "golden_questions.jsonl"
CHROMA_DIR = REPO_ROOT / "data" / "chroma"

SCHEME_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("HDFC Large Cap Fund - Direct Growth", "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?"),
    ("HDFC Equity (Flexi Cap) Fund - Direct Growth", "What is the minimum SIP for HDFC Equity Flexi Cap Fund?"),
    ("HDFC ELSS Tax Saver Fund - Direct Plan - Growth", "What is the lock-in period in HDFC ELSS Tax Saver Fund?"),
    ("HDFC Small Cap Fund - Direct Growth", "What is the exit load on HDFC Small Cap Fund?"),
    ("HDFC Balanced Advantage Fund - Direct Growth", "What is the riskometer level of HDFC Balanced Advantage Fund?"),
)

OFF_TOPIC: tuple[str, ...] = (
    "Who won the 2024 cricket world cup?",
    "How do I cook pasta?",
    "What is the price of Bitcoin today?",
    "Who is the president of India?",
)

FACTUAL = Decision("FACTUAL", None, "manual")


def make_chunk(
    chunk_id: str,
    *,
    text: str = "chunk text",
    scheme_name: str = "HDFC Large Cap Fund - Direct Growth",
    topic: str = "general",
    content_as_of: str = "",
) -> Chunk:
    return Chunk(
        source_id="s1",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        publisher="Groww (mirroring HDFC MF official data)",
        source_domain="groww.in",
        page_title=scheme_name,
        doc_type="scheme_page",
        scheme_name=scheme_name,
        scheme_category="large_cap",
        plan="direct",
        option="growth",
        topic=topic,
        section_heading=topic,
        content_as_of=content_as_of,
        chunk_index=0,
        text=text,
        ingested_at="2026-01-01T00:00:00Z",
        token_count=len(text.split()),
    )


class FakeQueryResult:
    """Stands in for `ingest.vectorstore.QueryResult` so unit tests need no model or Chroma."""

    def __init__(self, pairs: Sequence[tuple[Mapping[str, Any], str, float]]) -> None:
        self.metadatas = [md for md, _, _ in pairs]
        self.documents = [text for _, text, _ in pairs]
        self.scores = [score for _, _, score in pairs]


def cosine_pairs(
    rows: Iterable[tuple[Chunk, float]],
) -> list[tuple[Mapping[str, Any], str, float]]:
    return [(chunk_metadata(chunk), chunk.text, score) for chunk, score in rows]


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    import dataclasses

    return {field.name: getattr(chunk, field.name) for field in dataclasses.fields(Chunk)}


@pytest.fixture
def patch_backend(monkeypatch: pytest.MonkeyPatch):
    """Replace the embedding and Chroma query with recorded fixtures."""

    def _install(rows_by_where: Sequence[tuple[Mapping[str, Any] | None, list[tuple[Chunk, float]]]]) -> None:
        calls: list[Mapping[str, object] | None] = []

        def fake_embed(text: str, cfg: AppConfig) -> Any:
            return text

        def fake_query(vector: Any, n: int, cfg: AppConfig, where: Mapping[str, object] | None) -> FakeQueryResult:
            calls.append(where)
            for recorded_where, rows in rows_by_where:
                if recorded_where == where:
                    return FakeQueryResult(cosine_pairs(rows[:n]))
            return FakeQueryResult([])

        monkeypatch.setattr("app.retriever.embed_query", fake_embed)
        monkeypatch.setattr("app.retriever.query_chunks", fake_query)
        return calls

    return _install


# --- scheme and topic resolution -------------------------------------------------


def test_resolve_scheme_prefers_the_longest_alias(tmp_config: AppConfig) -> None:
    assert resolve_scheme("HDFC Large Cap Fund - Direct Growth?", tmp_config) == (
        "HDFC Large Cap Fund - Direct Growth"
    )
    assert resolve_scheme("HDFC Equity (Flexi Cap) Fund?", tmp_config) == (
        "HDFC Equity (Flexi Cap) Fund - Direct Growth"
    )


def test_resolve_scheme_without_a_scheme_name(tmp_config: AppConfig) -> None:
    assert resolve_scheme("What is the expense ratio?", tmp_config) is None


def test_scheme_aliases_include_short_and_parenthesised_forms() -> None:
    aliases = scheme_aliases("HDFC Equity (Flexi Cap) Fund - Direct Growth")
    assert "hdfc equity flexi cap fund direct growth" in aliases
    assert "hdfc equity flexi cap" in aliases
    assert "flexi cap" in aliases
    assert "hdfc flexi cap" in aliases


def test_resolve_topic(tmp_config: AppConfig) -> None:
    assert resolve_topic("What is the expense ratio?", tmp_config) == "expense_ratio"
    assert resolve_topic("What is the exit load?", tmp_config) == "exit_load"
    assert resolve_topic("Tell me something unrelated", tmp_config) is None


# --- where clauses ---------------------------------------------------------------


def test_build_where_topic_and_scheme() -> None:
    assert build_where("expense_ratio", "HDFC Small Cap Fund - Direct Growth") == {
        "$and": [
            {"$or": [{"topic": "expense_ratio"}, {"topic": "general"}]},
            {"scheme_name": "HDFC Small Cap Fund - Direct Growth"},
        ]
    }


def test_build_where_none_when_nothing_resolves() -> None:
    assert build_where(None, None) is None


def test_build_where_bundles_related_topics() -> None:
    where = build_where("objective", "HDFC Equity (Flexi Cap) Fund - Direct Growth", {"objective": ("benchmark",)})
    assert where == {
        "$and": [
            {"$or": [{"topic": "benchmark"}, {"topic": "general"}, {"topic": "objective"}]},
            {"scheme_name": "HDFC Equity (Flexi Cap) Fund - Direct Growth"},
        ]
    }


# --- fusion and dedupe -----------------------------------------------------------


def dense_hits(rows: Sequence[tuple[Chunk, float]]) -> list[RetrievalHit]:
    return [
        RetrievalHit(chunk=chunk, score=score, dense_rank=rank, lexical_rank=None)
        for rank, (chunk, score) in enumerate(rows, start=1)
    ]


def test_fuse_combines_dense_and_lexical_ranks(tmp_config: AppConfig) -> None:
    chunks = [make_chunk("alpha", text="alpha chunk body"), make_chunk("beta", text="beta chunk body")]
    fused = fuse(dense_hits(list(zip(chunks, (0.9, 0.8)))), {chunks[0].chunk_id: 1, chunks[1].chunk_id: 2}, tmp_config)
    rrf_k = tmp_config.retrieval.rrf_k
    weight = tmp_config.retrieval.lexical_weight
    assert fused[0].score == pytest.approx(1 / (rrf_k + 1) + weight / (rrf_k + 1))
    assert fused[1].score == pytest.approx(1 / (rrf_k + 2) + weight / (rrf_k + 2))
    assert fused[0].dense_rank == 1
    assert fused[0].lexical_rank == 1


def test_fuse_keeps_dense_only_hits_without_a_lexical_bonus(tmp_config: AppConfig) -> None:
    chunks = [make_chunk("alpha", text="alpha chunk body"), make_chunk("beta", text="beta chunk body")]
    fused = fuse(dense_hits(list(zip(chunks, (0.9, 0.8)))), {chunks[0].chunk_id: 1}, tmp_config)
    by_id = {hit.chunk.chunk_id: hit for hit in fused}
    assert by_id[chunks[1].chunk_id].lexical_rank is None
    assert by_id[chunks[1].chunk_id].score == pytest.approx(1 / (tmp_config.retrieval.rrf_k + 2))


def test_dedupe_keeps_the_newer_content_as_of(tmp_config: AppConfig) -> None:
    from app.retriever import _dedupe

    text = "Expense ratio of the fund is 0.5 percent."
    older = make_chunk("old", text=text, content_as_of="2024-01-01")
    newer = make_chunk("new", text=text, content_as_of="2026-01-01")
    hits = dense_hits([(older, 0.9), (newer, 0.8)])
    assert [hit.chunk.content_as_of for hit in _dedupe(hits)] == ["2026-01-01"]


# --- retrieval behaviour (stubbed backend) ---------------------------------------


def test_right_scheme_in_top_k_for_every_scheme(patch_backend, real_config: AppConfig) -> None:
    for scheme_name, question in SCHEME_QUESTIONS:
        chunk = make_chunk(f"{scheme_name}::general::1", scheme_name=scheme_name)
        patch_backend([(None, [(chunk, 0.9)])])
        result = retrieve(question, FACTUAL, real_config)
        assert result.status == "ok"
        assert [hit.chunk.scheme_name for hit in result.hits] == [scheme_name]


def test_no_scheme_confusion_when_a_scheme_resolves(patch_backend, real_config: AppConfig) -> None:
    right = make_chunk("right", scheme_name="HDFC Small Cap Fund - Direct Growth", text="Exit load is nil.")
    wrong = make_chunk("wrong", scheme_name="HDFC Large Cap Fund - Direct Growth", text="Exit load is nil.")
    patch_backend([(None, [(right, 0.9), (wrong, 0.85)])])
    result = retrieve("What is the exit load on HDFC Small Cap Fund?", FACTUAL, real_config)
    assert {hit.chunk.scheme_name for hit in result.hits} == {"HDFC Small Cap Fund - Direct Growth"}


def test_topic_filter_includes_an_expense_ratio_chunk(patch_backend, real_config: AppConfig) -> None:
    expense = make_chunk("er", topic="expense_ratio", text="The expense ratio is 0.5 percent.")
    general = make_chunk("gen", topic="general", text="Fund overview and scheme details.")
    expected = build_where("expense_ratio", "HDFC Large Cap Fund - Direct Growth")
    patch_backend([(expected, [(expense, 0.9), (general, 0.8)]), (None, [(expense, 0.9)])])
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?", FACTUAL, real_config)
    assert result.filtered_topic == "expense_ratio"
    assert any(hit.chunk.topic == "expense_ratio" for hit in result.hits)


def test_filter_relaxation_falls_back_when_the_filter_returns_fewer_hits(
    patch_backend, real_config: AppConfig
) -> None:
    mislabelled = make_chunk("relax", topic="general", text="The lock-in period is three years.")
    where = build_where("lockin_80c", "HDFC ELSS Tax Saver Fund - Direct Plan - Growth")
    patch_backend([(where, []), (None, [(mislabelled, 0.9)])])
    result = retrieve("What is the lock-in period in HDFC ELSS Tax Saver Fund?", FACTUAL, real_config)
    assert result.filtered_topic == "relaxed"
    assert [hit.chunk.text for hit in result.hits] == [mislabelled.text]


def test_determinism(patch_backend, real_config: AppConfig) -> None:
    rows = [(make_chunk(f"c{i}", text=f"Statement download detail {i}"), 0.9 - i / 100) for i in range(4)]
    patch_backend([(None, rows)])
    question = "How do I download my capital gains statement?"
    first = [(hit.chunk.chunk_id, hit.score) for hit in retrieve(question, FACTUAL, real_config).hits]
    second = [(hit.chunk.chunk_id, hit.score) for hit in retrieve(question, FACTUAL, real_config).hits]
    assert first == second


def test_lexical_boost_ranks_the_80c_chunk_first(patch_backend, real_config: AppConfig) -> None:
    lockin = make_chunk("80c", text="Lock-in period is three years for section 80C tax benefit.")
    other = make_chunk("other", text="The fund follows a large cap benchmark index.")
    patch_backend([(None, [(other, 0.95), (lockin, 0.6)])])
    result = retrieve(
        "What is the lock-in for 80C in HDFC ELSS Tax Saver Fund?", FACTUAL, real_config
    )
    assert result.hits[0].chunk.text == lockin.text


def test_out_of_corpus_when_no_hit_passes_the_floor(patch_backend, real_config: AppConfig) -> None:
    weak = make_chunk("weak", text="Unrelated filler text about a lock-in period.")
    patch_backend([(None, [(weak, 0.05)])])
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?", FACTUAL, real_config)
    assert result.status == "out_of_corpus"
    assert result.hits == ()


def test_out_of_corpus_for_a_named_foreign_scheme(patch_backend, real_config: AppConfig) -> None:
    foreign = make_chunk("foreign", text="Expense ratio of the fund is 0.5 percent.", topic="expense_ratio")
    patch_backend([(None, [(foreign, 0.9)])])
    result = retrieve("What is the expense ratio of Parag Parflex Fund?", FACTUAL, real_config)
    assert result.status == "out_of_corpus"
    assert result.hits == ()


def test_out_of_corpus_for_off_topic_questions(patch_backend, real_config: AppConfig) -> None:
    decoy = make_chunk("decoy", text="HDFC fund page overview with an index and a statement section.")
    for question in OFF_TOPIC:
        patch_backend([(None, [(decoy, 0.95)])])
        result = retrieve(question, FACTUAL, real_config)
        assert result.status == "out_of_corpus", question
        assert result.hits == (), question


def test_out_of_corpus_when_the_collection_is_missing(
    monkeypatch: pytest.MonkeyPatch, real_config: AppConfig
) -> None:
    from ingest.vectorstore import CorpusMissingError

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise CorpusMissingError("no collection")

    monkeypatch.setattr("app.retriever.embed_query", lambda text, cfg: text)
    monkeypatch.setattr("app.retriever.query_chunks", boom)
    result = retrieve("What is the expense ratio of HDFC Large Cap Fund?", FACTUAL, real_config)
    assert result.status == "out_of_corpus"
    assert result.hits == ()


def test_retrieve_never_exceeds_top_k(patch_backend, real_config: AppConfig) -> None:
    rows = [
        (make_chunk(f"c{i}", text=f"Statement download detail number {i}"), 0.9 - i / 1000)
        for i in range(12)
    ]
    patch_backend([(None, rows)])
    result = retrieve("How do I download my statement?", FACTUAL, real_config)
    assert len(result.hits) <= real_config.retrieval.top_k


# --- real-corpus checks (require the built collection) ----------------------------


def corpus_available() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


requires_corpus = pytest.mark.skipif(
    not corpus_available(), reason="run the ingest pipeline to build data/chroma first"
)


def load_golden() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


@requires_corpus
def test_golden_questions_match_their_expected_outcome(real_config: AppConfig) -> None:
    for row in load_golden():
        result = retrieve(row["question"], FACTUAL, real_config)
        if row["expect_refusal"]:
            assert result.status == "out_of_corpus", row["id"]
            assert result.hits == (), row["id"]
        else:
            assert result.status == "ok", row["id"]
            assert result.hits, row["id"]


@requires_corpus
def test_real_corpus_resolves_the_right_scheme_per_question(real_config: AppConfig) -> None:
    for scheme_name, question in SCHEME_QUESTIONS:
        result = retrieve(question, FACTUAL, real_config)
        assert result.status == "ok", question
        assert scheme_name in {hit.chunk.scheme_name for hit in result.hits}, question


@requires_corpus
def test_real_corpus_rejects_off_topic_questions(real_config: AppConfig) -> None:
    for question in OFF_TOPIC:
        result = retrieve(question, FACTUAL, real_config)
        assert result.status == "out_of_corpus", question
        assert result.hits == (), question
