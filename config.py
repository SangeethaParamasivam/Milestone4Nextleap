from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from app.models import SourceSpec

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

DEFAULT_PATHS: Mapping[str, str] = {
    "raw_dir": "data/raw",
    "chunks_dir": "data/chunks",
    "persist_dir": "data/chroma",
    "cache_dir": "data/cache",
    "models_dir": "data/models",
    "status_file": "data/status.json",
    "rejected_file": "data/rejected.jsonl",
}

DEFAULT_CHUNKING: Mapping[str, Any] = {
    "size": 900,
    "overlap": 150,
    "min_chars": 40,
    "max_chunks_per_page": 400,
    "max_tokens": 250,
    "min_page_chars": 400,
}

DEFAULT_RETRIEVAL: Mapping[str, Any] = {
    "top_k": 5,
    "min_score": 0.28,
    "candidate_multiplier": 4,
    "lexical_weight": 0.6,
    "rrf_k": 60,
    "scheme_affinity": True,
}

DEFAULT_EMBEDDING: Mapping[str, Any] = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dim": 384,
    "batch_size": 32,
    "normalize": True,
    "offline_env": True,
}

DEFAULT_GENERATION: Mapping[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 300,
    "seed": 42,
    "max_sentences": 3,
    "max_words": 60,
    "retries_on_citation_failure": 1,
    "retries_on_http_error": 1,
}

DEFAULT_TOPICS: Mapping[str, tuple[str, ...]] = {
    "expense_ratio": ("expense ratio", "ter", "total expense", "charges"),
    "exit_load": ("exit load", "exit charges"),
    "min_sip": ("minimum sip", "min sip", "minimum instal"),
    "lockin_80c": ("lock-in", "lock in", "lockin", "80c", "section 80c"),
    "riskometer": ("riskometer",),
    "benchmark": ("benchmark", "index"),
    "statements_download": ("statement", "tax document", "how to download", "capital gains"),
    "nav": ("nav", "net asset value"),
    "plan_direct_vs_regular": ("direct vs regular", "direct plan", "regular plan"),
    "objective": ("objective", "about the scheme", "scheme profile"),
    "entry_load": ("entry load",),
    "min_lump_sum": ("minimum lump sum", "lump sum"),
}

DEFAULT_MEMORY: Mapping[str, Any] = {
    "max_messages": 10,
    "rewrite": True,
    "max_rewrite_chars": 300,
    "max_rewrite_tokens": 120,
    "min_rewrite_words": 2,
}

DEFAULT_DEICTIC_TRIGGERS: tuple[str, ...] = (
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "them",
    "they",
    "he",
    "she",
    "his",
    "her",
    "same",
    "other",
    "another",
    "also",
    "and",
)

DEFAULT_FRESHNESS_WARN_DAYS = 90


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkingParams:
    size: int
    overlap: int
    min_chars: int
    max_chunks_per_page: int
    max_tokens: int
    min_page_chars: int


@dataclass(frozen=True)
class RetrievalParams:
    top_k: int
    min_score: float
    candidate_multiplier: int
    lexical_weight: float
    rrf_k: int
    scheme_affinity: bool


@dataclass(frozen=True)
class EmbeddingParams:
    model: str
    dim: int
    batch_size: int
    normalize: bool
    offline_env: bool


@dataclass(frozen=True)
class GenerationParams:
    temperature: float
    max_tokens: int
    seed: int
    max_sentences: int
    max_words: int
    retries_on_citation_failure: int
    retries_on_http_error: int


@dataclass(frozen=True)
class MemoryParams:
    max_messages: int
    rewrite: bool
    max_rewrite_chars: int
    max_rewrite_tokens: int
    min_rewrite_words: int
    deictic_triggers: tuple[str, ...] = DEFAULT_DEICTIC_TRIGGERS


@dataclass(frozen=True)
class RefusalRules:
    advice: tuple[str, ...]
    performance: tuple[str, ...]
    pii_patterns: tuple[str, ...]
    educational_links: Mapping[str, str]


@dataclass(frozen=True)
class PathParams:
    raw_dir: Path
    chunks_dir: Path
    persist_dir: Path
    cache_dir: Path
    models_dir: Path
    status_file: Path
    rejected_file: Path


@dataclass(frozen=True)
class AppConfig:
    sources: tuple[SourceSpec, ...]
    allowed_domains: frozenset[str]
    chunking: ChunkingParams
    retrieval: RetrievalParams
    embedding: EmbeddingParams
    generation: GenerationParams
    topics: dict[str, tuple[str, ...]]
    refusal: RefusalRules
    memory: MemoryParams
    out_of_corpus: frozenset[str]
    paths: PathParams
    freshness_warn_days: int
    groq_model: str = DEFAULT_GROQ_MODEL
    related_topics: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def source_by_id(self, source_id: str) -> SourceSpec:
        for spec in self.sources:
            if spec.id == source_id:
                return spec
        raise ConfigError(f"unknown source id: {source_id}")


def _base_dir(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "config" else config_path.parent


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> AppConfig:
    """Resolve configuration with precedence real env > .env file > config.yaml > code default.

    Relative `paths:` entries resolve against the repository root, taken as the parent of
    `config/` when the file lives in a directory named `config`, otherwise its own directory.
    """
    config_path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    load_dotenv(ENV_FILE, override=False)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {config_path}")
    base = _base_dir(config_path)
    allowed_domains = frozenset(
        str(domain).strip().lower() for domain in raw.get("allowed_domains", []) if str(domain).strip()
    )
    sources = _build_sources(raw.get("sources", []), allowed_domains)
    chunking = _with_env(
        _build_params(ChunkingParams, raw.get("chunking"), DEFAULT_CHUNKING),
        "CHUNK_SIZE",
    )
    chunking = _with_env(chunking, "CHUNK_OVERLAP")
    retrieval = _with_env(
        _build_params(RetrievalParams, raw.get("retrieval"), DEFAULT_RETRIEVAL),
        "TOP_K",
    )
    retrieval = _with_env(retrieval, "MIN_SCORE")
    raw_retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
    embedding = _with_env(
        _build_params(EmbeddingParams, raw.get("embedding"), DEFAULT_EMBEDDING),
        "EMBED_MODEL",
    )
    memory = _with_env(
        _build_memory(raw.get("memory")), "MEMORY_MAX_MESSAGES"
    )
    return AppConfig(
        sources=sources,
        allowed_domains=allowed_domains,
        chunking=chunking,
        retrieval=retrieval,
        embedding=embedding,
        generation=_build_params(GenerationParams, raw.get("generation"), DEFAULT_GENERATION),
        topics=_build_topics(raw.get("topics")),
        refusal=_build_refusal(raw.get("refusal")),
        memory=memory,
        out_of_corpus=frozenset(
            str(name).strip().lower()
            for name in (raw.get("out_of_corpus") or {}).get("known_foreign_schemes", [])
            if str(name).strip()
        ),
        paths=_build_paths(raw.get("paths"), base),
        freshness_warn_days=_coerce(
            (raw.get("freshness") or {}).get("warn_after_days"), DEFAULT_FRESHNESS_WARN_DAYS
        ),
        groq_model=_env_str("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
        related_topics=_build_related_topics(raw_retrieval.get("related_topics")),
    )


def get_groq_api_key() -> str | None:
    value = os.getenv("GROQ_API_KEY")
    return value.strip() if value and value.strip() else None


def get_groq_model() -> str:
    return _env_str("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def _build_sources(raw_sources: object, allowed_domains: frozenset[str]) -> tuple[SourceSpec, ...]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError("config must define a non-empty `sources` list")
    specs: list[SourceSpec] = []
    seen_ids: set[str] = set()
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise ConfigError(f"each source must be a mapping, got {type(entry).__name__}")
        missing = [key for key in ("id", "url", "domain", "doc_type", "publisher") if not entry.get(key)]
        if missing:
            raise ConfigError(f"source {entry.get('id', '<no id>')} missing keys: {', '.join(missing)}")
        source_id = str(entry["id"])
        if source_id in seen_ids:
            raise ConfigError(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)
        url = str(entry["url"])
        host = (urlsplit(url).hostname or "").lower()
        domain = str(entry["domain"]).strip().lower()
        if urlsplit(url).scheme != "https":
            raise ConfigError(f"source {source_id}: only https URLs are allowed ({url})")
        if not host:
            raise ConfigError(f"source {source_id}: url has no host ({url})")
        if not allowed_domains:
            raise ConfigError("config must define a non-empty `allowed_domains` list")
        if host not in allowed_domains or domain not in allowed_domains:
            raise ConfigError(
                f"source {source_id}: host {host!r} (domain {domain!r}) is not in allowed_domains"
            )
        specs.append(
            SourceSpec(
                id=source_id,
                url=url,
                domain=domain,
                doc_type=str(entry["doc_type"]),
                scheme_name=_opt(entry.get("scheme_name")),
                scheme_category=_opt(entry.get("scheme_category")),
                plan=_opt(entry.get("plan")),
                option=_opt(entry.get("option")),
                publisher=str(entry["publisher"]),
            )
        )
    return tuple(specs)


def _build_paths(raw_paths: object, base: Path) -> PathParams:
    values = raw_paths if isinstance(raw_paths, dict) else {}
    resolved: dict[str, Path] = {}
    for key, default in DEFAULT_PATHS.items():
        raw_value = values.get(key, default)
        env_value = _env_str(key.upper()) if key.endswith(("_dir", "_file")) else None
        chosen = env_value or str(raw_value)
        path = Path(chosen).expanduser()
        resolved[key] = path if path.is_absolute() else (base / path).resolve()
    return PathParams(**resolved)


def _build_topics(raw_topics: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_topics, dict) or not raw_topics:
        return dict(DEFAULT_TOPICS)
    topics: dict[str, tuple[str, ...]] = {}
    for topic, phrases in raw_topics.items():
        if isinstance(phrases, str):
            raise ConfigError(f"topics.{topic} must be a list of phrases")
        topics[str(topic)] = tuple(str(phrase) for phrase in phrases)
    return topics


def _build_related_topics(raw_topics: object) -> dict[str, tuple[str, ...]]:
    """`retrieval.related_topics`: a topic may also match chunks of the listed topics.

    The vocabulary itself is not extended here; every key and every value must already exist in
    `cfg.topics`, or the retrieval where-clause would silently match an unknown field.
    """
    if not isinstance(raw_topics, dict) or not raw_topics:
        return {}
    related: dict[str, tuple[str, ...]] = {}
    for topic, others in raw_topics.items():
        if isinstance(others, str):
            raise ConfigError(f"retrieval.related_topics.{topic} must be a list of topics")
        related[str(topic)] = tuple(str(other) for other in others)
    return related


def _build_refusal(raw_refusal: object) -> RefusalRules:
    values = raw_refusal if isinstance(raw_refusal, dict) else {}
    advice = tuple(str(p) for p in values.get("advice", []))
    performance = tuple(str(p) for p in values.get("performance", []))
    pii_patterns = tuple(str(p) for p in values.get("pii_patterns", []))
    links_raw = values.get("educational_links")
    links = (
        {str(k): str(v) for k, v in links_raw.items()}
        if isinstance(links_raw, dict)
        else {"advice": "", "performance": "", "pii": ""}
    )
    return RefusalRules(
        advice=advice,
        performance=performance,
        pii_patterns=pii_patterns,
        educational_links=dict(links),
    )


def _build_memory(raw: object) -> MemoryParams:
    """`memory:` section, with the deictic trigger words kept as an inspectable list in config.yaml.

    The triggers gate the rewrite call: a question that names its subject is already standalone, so
    it is retrieved as typed and no LLM call is made for it.
    """
    values = raw if isinstance(raw, dict) else {}
    triggers = values.get("deictic_triggers")
    words = (
        tuple(str(word).strip().lower() for word in triggers if str(word).strip())
        if isinstance(triggers, list)
        else DEFAULT_DEICTIC_TRIGGERS
    )
    return MemoryParams(
        max_messages=_coerce(values.get("max_messages"), DEFAULT_MEMORY["max_messages"]),
        rewrite=_coerce(values.get("rewrite"), DEFAULT_MEMORY["rewrite"]),
        max_rewrite_chars=_coerce(
            values.get("max_rewrite_chars"), DEFAULT_MEMORY["max_rewrite_chars"]
        ),
        max_rewrite_tokens=_coerce(
            values.get("max_rewrite_tokens"), DEFAULT_MEMORY["max_rewrite_tokens"]
        ),
        min_rewrite_words=_coerce(
            values.get("min_rewrite_words"), DEFAULT_MEMORY["min_rewrite_words"]
        ),
        deictic_triggers=words,
    )


def _build_params(cls: type, raw: object, defaults: Mapping[str, Any]) -> Any:
    values = raw if isinstance(raw, dict) else {}
    return cls(**{key: _coerce(values.get(key), default) for key, default in defaults.items()})


def _coerce(value: object, default: Any) -> Any:
    """Coerce a YAML/env value to the type of the code default that stands in for it."""
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            raise ConfigError(f"expected a boolean, got {value!r}")
        return bool(value)
    if isinstance(default, int):
        return _as_int(value, default)
    if isinstance(default, float):
        return _as_float(value, default)
    return _as_str(value, default)


def _as_str(value: object, default: Any = None) -> Any:
    return default if value is None else str(value)


def _as_int(value: object, default: Any = None) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"expected an integer, got {value!r}") from exc


def _as_float(value: object, default: Any = None) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"expected a number, got {value!r}") from exc


def _opt(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env_str(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


_ENV_OVERRIDES: Mapping[str, str] = {
    "TOP_K": "retrieval.top_k",
    "MIN_SCORE": "retrieval.min_score",
    "CHUNK_SIZE": "chunking.size",
    "CHUNK_OVERLAP": "chunking.overlap",
    "EMBED_MODEL": "embedding.model",
    "MEMORY_MAX_MESSAGES": "memory.max_messages",
}


def _with_env(params: Any, env_name: str) -> Any:
    raw = _env_str(env_name)
    if raw is None:
        return params
    field_name = _ENV_OVERRIDES[env_name].split(".")[1]
    return replace(params, **{field_name: _coerce(raw, getattr(params, field_name))})
