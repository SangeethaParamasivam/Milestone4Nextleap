from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Hashable, Mapping, Protocol, Sequence

import numpy as np

from app.models import Chunk, IngestStatus
from config import AppConfig

COLLECTION_NAME = "mf_faqs"
COLLECTION_METADATA: Mapping[str, Hashable] = {"hnsw:space": "cosine"}
STATUS_KEYS = ("corpus_hash", "chunk_count", "model", "collection", "last_ingest_at", "source_hashes")


class CorpusMissingError(RuntimeError):
    pass


class ChromaClient(Protocol):
    def get_or_create_collection(self, name: str, metadata: Mapping[str, Hashable] | None = ...) -> "Collection": ...
    def get_collection(self, name: str) -> "Collection": ...


class Collection(Protocol):
    def upsert(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Mapping[str, Hashable]], embeddings: Sequence[Sequence[float]]) -> None: ...
    def delete(self, ids: Sequence[str]) -> None: ...
    def count(self) -> int: ...
    def get(self, include: Sequence[str] = ...) -> Mapping[str, object]: ...
    def query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int, where: Mapping[str, object] | None = ..., include: Sequence[str] = ...) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class QueryResult:
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[Mapping[str, Hashable], ...]
    scores: tuple[float, ...]


@lru_cache(maxsize=4)
def _persistent_client(persist_dir: str) -> ChromaClient:
    import chromadb

    return chromadb.PersistentClient(path=persist_dir)


def get_client(cfg: AppConfig) -> ChromaClient:
    """One persistent client per `paths.persist_dir`; the store lives on disk, never in memory."""
    cfg.paths.persist_dir.mkdir(parents=True, exist_ok=True)
    return _persistent_client(str(cfg.paths.persist_dir))


def get_collection(cfg: AppConfig, *, create: bool) -> Collection:
    client = get_client(cfg)
    if create:
        return client.get_or_create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        raise CorpusMissingError(f"collection {COLLECTION_NAME} not found in {cfg.paths.persist_dir}") from error
    if collection.count() == 0:
        raise CorpusMissingError(f"collection {COLLECTION_NAME} is empty in {cfg.paths.persist_dir}")
    return collection


def upsert_chunks(chunks: Sequence[Chunk], vectors: np.ndarray, cfg: AppConfig) -> int:
    """Always upsert by `chunk_id`; `text` is the document, `to_metadata()` stays scalar-only."""
    if len(chunks) != len(vectors):
        raise CorpusMissingError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        return 0
    collection = get_collection(cfg, create=True)
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.to_metadata() for chunk in chunks],
        embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
    )
    return len(chunks)


def prune(chunks: Sequence[Chunk], cfg: AppConfig) -> int:
    """Delete stored ids that are no longer in the corpus, so an edited page leaves no orphans."""
    collection = get_collection(cfg, create=False)
    stored = collection.get(include=[])
    ids = [str(value) for value in stored.get("ids", [])]
    keep = {chunk.chunk_id for chunk in chunks}
    stale = [chunk_id for chunk_id in ids if chunk_id not in keep]
    if stale:
        collection.delete(ids=stale)
    return len(stale)


def query_chunks(vector: np.ndarray, n: int, cfg: AppConfig, where: Mapping[str, object] | None) -> QueryResult:
    collection = get_collection(cfg, create=False)
    raw = collection.query(
        query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = tuple(str(value) for value in _first(raw, "ids"))
    documents = tuple(str(value) for value in _first(raw, "documents"))
    metadatas = _metadatas(raw)
    scores = tuple(round(1.0 - float(distance), 6) for distance in _first(raw, "distances"))
    return QueryResult(ids=ids, documents=documents, metadatas=metadatas, scores=scores)


def corpus_hash(source_hashes: Mapping[str, str]) -> str:
    payload = "\n".join(f"{source_id}:{digest}" for source_id, digest in sorted(source_hashes.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_status(cfg: AppConfig, chunks: Sequence[Chunk], source_hashes: Mapping[str, str]) -> IngestStatus:
    status = IngestStatus(
        corpus_hash=corpus_hash(source_hashes),
        chunk_count=len(chunks),
        model=cfg.embedding.model,
        collection=COLLECTION_NAME,
        last_ingest_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_hashes=dict(source_hashes),
    )
    path = cfg.paths.status_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(_status_record(status), indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return status


def read_status(cfg: AppConfig) -> IngestStatus | None:
    path = cfg.paths.status_file
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return IngestStatus(
        corpus_hash=str(record.get("corpus_hash", "")),
        chunk_count=int(record.get("chunk_count", 0)),
        model=str(record.get("model", "")),
        collection=str(record.get("collection", "")),
        last_ingest_at=str(record.get("last_ingest_at", "")),
        source_hashes={str(key): str(value) for key, value in dict(record.get("source_hashes", {})).items()},
    )


def _status_record(status: IngestStatus) -> Mapping[str, object]:
    return {
        "corpus_hash": status.corpus_hash,
        "chunk_count": status.chunk_count,
        "model": status.model,
        "collection": status.collection,
        "last_ingest_at": status.last_ingest_at,
        "source_hashes": dict(status.source_hashes),
    }


def _first(raw: Mapping[str, object], key: str) -> Sequence[object]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        return ()
    first = values[0]
    return first if isinstance(first, list) else ()


def _metadatas(raw: Mapping[str, object]) -> tuple[Mapping[str, Hashable], ...]:
    return tuple(
        dict(entry) if isinstance(entry, dict) else {}
        for entry in _first(raw, "metadatas")
    )
