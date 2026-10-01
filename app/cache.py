"""Answer cache (implementation §9.6.3). In-memory plus optional JSONL at `paths.cache_dir`.

The key is `sha256(f"{model}|{normalised_question}|{sorted_chunk_ids}")`, so a corpus rebuild or
a model change misses the cache. Only validated answers are stored: a suppressed result is never
cached, so a fix in the validators is picked up on the next run. Nothing sensitive is written —
no question or answer text, only the key and the validated payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from app.models import Answer, RetrievalHit
from config import AppConfig

_LOG = logging.getLogger("app.cache")

CACHE_VERSION = "v1"
_MEMORY: dict[str, Answer] = {}


def cache_key(question: str, hits: Sequence[RetrievalHit], model: str) -> str:
    """`sha256(version | model | normalised question | sorted "chunk_id@source_url" pairs)`.

    `chunk_id` is derived from the chunk text, so two sources carrying identical text share an id;
    the URL is therefore part of the key, because the answer renders it. The version prefix means
    a change to how an `Answer` is built invalidates existing entries instead of serving a stale
    payload.
    """
    from app.guard import normalise_question

    payload = "|".join(
        [
            CACHE_VERSION,
            model,
            normalise_question(question),
            ",".join(
                sorted(f"{hit.chunk.chunk_id}@{hit.chunk.source_url}" for hit in hits)
            ),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_file(cfg: AppConfig) -> Path:
    return cfg.paths.cache_dir / "answers.jsonl"


def lookup(key: str, cfg: AppConfig) -> Answer | None:
    """Return a cached answer from memory, then from the JSONL file, marked as a cache hit."""
    if key in _MEMORY:
        return _as_hit(_MEMORY[key])
    path = _cache_file(cfg)
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("key") == key:
                return _as_hit(_answer_from_record(record))
    except (OSError, json.JSONDecodeError) as error:
        _LOG.warning("cache read failed: %s", type(error).__name__)
    return None


def _as_hit(answer: Answer) -> Answer:
    """A cache hit is always reported with `was_cached=True`, however it was found."""
    if answer.was_cached:
        return answer
    return replace(answer, was_cached=True)


def store(key: str, answer: Answer, cfg: AppConfig) -> None:
    """Persist a validated answer. Suppressed answers are never cached."""
    if answer.status != "ok":
        return
    _MEMORY[key] = answer
    path = _cache_file(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record_from_answer(key, answer), sort_keys=True) + "\n")
    except OSError as error:
        _LOG.warning("cache write failed: %s", type(error).__name__)


def clear_memory() -> None:
    """Test hook: drop the in-process cache."""
    _MEMORY.clear()


def _record_from_answer(key: str, answer: Answer) -> dict[str, object]:
    return {
        "key": key,
        "text": answer.text,
        "source_url": answer.source_url,
        "doc_type": answer.doc_type,
        "publisher": answer.publisher,
        "content_as_of": answer.content_as_of,
        "used_chunk_ids": list(answer.used_chunk_ids),
        "status": answer.status,
    }


def _answer_from_record(record: dict[str, object]) -> Answer:
    return Answer(
        text=str(record["text"]),
        source_url=str(record["source_url"]),
        doc_type=str(record["doc_type"]),
        publisher=str(record["publisher"]),
        content_as_of=str(record["content_as_of"]),
        used_chunk_ids=tuple(record["used_chunk_ids"]),
        status=str(record["status"]),
        was_cached=True,
    )
