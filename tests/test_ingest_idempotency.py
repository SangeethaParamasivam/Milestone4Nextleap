from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import run_ingest
from app.models import Chunk
from config import AppConfig, load_config
from ingest.chunker import plan_chunks
from ingest.cleaner import clean_file
from ingest.embedder import (
    embed_documents,
    embed_query,
    embeddings_preview_path,
    get_model,
    real_token_count,
    write_embeddings_preview,
)
from ingest.loader import cached_path
from ingest.vectorstore import (
    COLLECTION_NAME,
    CorpusMissingError,
    corpus_hash,
    get_collection,
    prune,
    query_chunks,
    read_status,
    upsert_chunks,
    write_status,
)


@pytest.fixture(scope="module")
def store_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    """Real sources and the real cached model/HTML, but a throwaway Chroma dir, status file and chunks dir."""
    repo = load_config()
    base = tmp_path_factory.mktemp("phase4")
    paths = replace(
        repo.paths,
        raw_dir=repo.paths.raw_dir,
        chunks_dir=base / "chunks",
        persist_dir=base / "chroma",
        cache_dir=base / "cache",
        models_dir=repo.paths.models_dir,
        status_file=base / "status.json",
        rejected_file=base / "rejected.jsonl",
    )
    return replace(repo, paths=paths)


@pytest.fixture(scope="module")
def corpus(store_config: AppConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for spec in store_config.sources:
        doc = clean_file(cached_path(spec, store_config), spec, store_config)
        chunks.extend(plan_chunks(doc, spec, store_config).chunks)
    return chunks


@pytest.fixture(scope="module")
def corpus_vectors(store_config: AppConfig, corpus: list[Chunk]) -> np.ndarray:
    return embed_documents([chunk.text for chunk in corpus], store_config)


@pytest.fixture
def fresh_store(store_config: AppConfig, tmp_path: Path) -> AppConfig:
    """An empty collection of its own, so prune and upsert counts are unambiguous."""
    return replace(
        store_config,
        paths=replace(store_config.paths, persist_dir=tmp_path / "chroma", status_file=tmp_path / "status.json"),
    )


@pytest.fixture
def wire_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point `run_ingest.run()` at a chosen store and count its embed/upsert calls."""

    def _wire(cfg: AppConfig) -> dict[str, int]:
        counters = {"embed": 0, "upsert": 0}
        real_embed = run_ingest.embed_documents
        real_upsert = run_ingest.upsert_chunks

        def counting_embed(texts, config):
            counters["embed"] += 1
            return real_embed(texts, config)

        def counting_upsert(chunks, vectors, config):
            counters["upsert"] += 1
            return real_upsert(chunks, vectors, config)

        monkeypatch.setattr(run_ingest, "load_config", lambda *args, **kwargs: cfg)
        monkeypatch.setattr(run_ingest, "embed_documents", counting_embed)
        monkeypatch.setattr(run_ingest, "upsert_chunks", counting_upsert)
        monkeypatch.setattr(run_ingest, "sources_csv_path", lambda *args, **kwargs: tmp_path / "sources.csv")
        return counters

    return _wire


def test_two_runs_produce_identical_state(
    store_config: AppConfig, corpus: list[Chunk], corpus_vectors: np.ndarray, wire_run
) -> None:
    assert upsert_chunks(corpus, corpus_vectors, store_config) == len(corpus)
    collection = get_collection(store_config, create=False)
    first_ids = set(collection.get(include=[])["ids"])
    first_count = collection.count()

    assert upsert_chunks(corpus, corpus_vectors, store_config) == len(corpus)
    assert collection.count() == first_count == len(corpus)
    assert set(collection.get(include=[])["ids"]) == first_ids

    counters = wire_run(store_config)
    assert run_ingest.run() == 0
    assert counters["upsert"] == 1
    assert get_collection(store_config, create=False).count() == first_count
    assert set(get_collection(store_config, create=False).get(include=[])["ids"]) == first_ids


def test_prune_deletes_the_superseded_chunk_id(fresh_store: AppConfig, corpus: list[Chunk]) -> None:
    sample = corpus[:6]
    upsert_chunks(sample, embed_documents([c.text for c in sample], fresh_store), fresh_store)
    stale_id = sample[0].chunk_id

    edited = replace(sample[0], text=sample[0].text + " Updated on 30 Sep 2026.")
    assert edited.chunk_id != stale_id
    replacement = [edited, *sample[1:]]
    upsert_chunks(replacement, embed_documents([c.text for c in replacement], fresh_store), fresh_store)

    assert prune(replacement, fresh_store) == 1
    stored = set(get_collection(fresh_store, create=False).get(include=[])["ids"])
    assert stale_id not in stored
    assert edited.chunk_id in stored
    assert get_collection(fresh_store, create=False).count() == len(replacement)


def test_status_gate_skips_embed_and_force_re_embeds(fresh_store: AppConfig, wire_run, capsys) -> None:
    counters = wire_run(fresh_store)
    assert run_ingest.run() == 0
    assert counters["embed"] == 1
    assert counters["upsert"] == 1
    capsys.readouterr()

    assert run_ingest.run() == 0
    assert counters["embed"] == 1
    assert "corpus unchanged, skipping embed+store" in capsys.readouterr().out
    assert get_collection(fresh_store, create=False).count() == read_status(fresh_store).chunk_count

    assert run_ingest.run(force=True) == 0
    assert counters["embed"] == 2
    assert counters["upsert"] == 2


def test_dry_run_stops_before_embed(fresh_store: AppConfig, wire_run) -> None:
    counters = wire_run(fresh_store)
    assert run_ingest.run(dry_run=True) == 0
    assert counters["embed"] == 0
    assert counters["upsert"] == 0
    assert not fresh_store.paths.status_file.exists()


def test_metadata_values_are_scalars(corpus: list[Chunk]) -> None:
    for chunk in corpus[:20]:
        for value in chunk.to_metadata().values():
            assert isinstance(value, (str, int, float, bool))
            assert not isinstance(value, (list, tuple, dict))
    assert "text" not in corpus[0].to_metadata()


def test_real_token_count_is_within_the_bound_for_every_chunk(store_config: AppConfig, corpus: list[Chunk]) -> None:
    counts = [real_token_count(chunk.text, store_config) for chunk in corpus]
    assert max(counts) <= store_config.chunking.max_tokens
    assert get_model(store_config).max_seq_length >= max(counts)


def test_embed_documents_shape_dtype_and_unit_norm(store_config: AppConfig, corpus: list[Chunk]) -> None:
    sample = [chunk.text for chunk in corpus[:5]]
    vectors = embed_documents(sample, store_config)
    assert vectors.shape == (5, store_config.embedding.dim) == (5, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    query = embed_query("What is the expense ratio of HDFC Large Cap Fund - Direct Growth?", store_config)
    assert query.shape == (384,)
    assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-5
    assert embed_documents([], store_config).shape == (0, 384)


def test_ingest_runs_without_network(store_config: AppConfig, no_network, wire_run) -> None:
    assert run_ingest.run() == 0
    assert get_collection(store_config, create=False).count() > 0


def test_status_json_has_exactly_the_six_keys(store_config: AppConfig, corpus: list[Chunk]) -> None:
    status = write_status(store_config, corpus[:3], {"s1": "a" * 64, "s2": "b" * 64})
    record = json.loads(store_config.paths.status_file.read_text(encoding="utf-8"))
    assert set(record) == {"corpus_hash", "chunk_count", "model", "collection", "last_ingest_at", "source_hashes"}
    assert record["collection"] == COLLECTION_NAME
    assert record["model"] == store_config.embedding.model
    assert "groww.in" not in json.dumps(record).replace("https://", "")
    assert read_status(store_config) == status


def test_corpus_hash_is_order_independent() -> None:
    assert corpus_hash({"s2": "b", "s1": "a"}) == corpus_hash({"s1": "a", "s2": "b"})
    assert corpus_hash({"s1": "a"}) != corpus_hash({"s1": "b"})


def test_query_chunks_returns_scores_and_documents(
    store_config: AppConfig, corpus: list[Chunk], corpus_vectors: np.ndarray
) -> None:
    upsert_chunks(corpus, corpus_vectors, store_config)
    result = query_chunks(
        embed_query("Exit load of HDFC Large Cap Fund - Direct Growth", store_config), 3, store_config, None
    )
    assert len(result.ids) == len(result.documents) == len(result.scores) == 3
    assert all(0.0 <= score <= 1.0001 for score in result.scores)
    assert result.scores == tuple(sorted(result.scores, reverse=True))
    assert set(result.metadatas[0]) == set(corpus[0].to_metadata())


def test_missing_collection_raises(store_config: AppConfig, tmp_path: Path) -> None:
    empty = replace(store_config, paths=replace(store_config.paths, persist_dir=tmp_path / "nowhere"))
    with pytest.raises(CorpusMissingError):
        get_collection(empty, create=False)


def test_embeddings_preview_lists_first_five_rows_and_ten_dims(
    store_config: AppConfig, corpus: list[Chunk]
) -> None:
    vectors = embed_documents([chunk.text for chunk in corpus[:5]], store_config)
    path = write_embeddings_preview(vectors, [chunk.text for chunk in corpus[:5]], store_config)
    assert path == embeddings_preview_path(store_config) == store_config.paths.persist_dir.parent / "embeddings_preview.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("[")]
    assert len(rows) == 5
    for index, line in enumerate(rows):
        head = line.split("first_10:")[1].split()
        assert len(head) == 10
        assert "l2_norm=1.0000" in line
    assert any("text:" in line for line in lines)
