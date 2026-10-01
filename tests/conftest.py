from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import requests
import yaml

from app.models import Block, CleanDocument, SourceSpec
from config import ENV_FILE, REPO_ROOT, load_config

CONFIG_TEXT = """
sources:
  - {id: s1, url: "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", domain: groww.in,
     doc_type: scheme_page, scheme_name: "HDFC Large Cap Fund - Direct Growth", scheme_category: large_cap,
     plan: direct, option: growth, publisher: "Groww (mirroring HDFC MF official data)"}
  - {id: s2, url: "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth", domain: groww.in,
     doc_type: scheme_page, scheme_name: "HDFC Equity (Flexi Cap) Fund - Direct Growth",
     scheme_category: flexi_cap, plan: direct, option: growth,
     publisher: "Groww (mirroring HDFC MF official data)"}
allowed_domains: [groww.in, hdfcmf.com, sebi.gov.in, amfiindia.com, myamfi.in]
paths: {raw_dir: data/raw, chunks_dir: data/chunks, persist_dir: data/chroma, cache_dir: data/cache,
        models_dir: data/models, status_file: data/status.json, rejected_file: data/rejected.jsonl}
chunking: {size: 900, overlap: 150, min_chars: 40, max_chunks_per_page: 400, max_tokens: 250,
           min_page_chars: 400}
retrieval: {top_k: 5, min_score: 0.28, candidate_multiplier: 4, lexical_weight: 0.6, rrf_k: 60,
            scheme_affinity: true}
embedding: {model: "sentence-transformers/all-MiniLM-L6-v2", dim: 384, batch_size: 32,
            normalize: true, offline_env: true}
generation: {temperature: 0.0, max_tokens: 300, seed: 42, max_sentences: 3, max_words: 60,
             retries_on_citation_failure: 1, retries_on_http_error: 1}
topics:
  exit_load: ["exit load", "exit charges"]
  expense_ratio: ["expense ratio", "ter", "total expense", "charges"]
  nav: ["nav", "net asset value"]
refusal:
  advice: ["should i", "recommend"]
  performance: ["return", "cagr"]
  pii_patterns: ["\\\\b(otp|cvv)\\\\b"]
  educational_links:
    advice: "https://www.investor.gov.in/"
    performance: "https://www.investor.gov.in/"
    pii: "https://www.investor.gov.in/"
out_of_corpus: {known_foreign_schemes: ["parag parflex"]}
freshness: {warn_after_days: 90}
"""

ENV_KEYS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "TOP_K",
    "MIN_SCORE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MEMORY_MAX_MESSAGES",
    "PERSIST_DIR",
    "RAW_DIR",
    "EMBED_MODEL",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    load_config.cache_clear()


@pytest.fixture
def real_config():
    return load_config()


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(text: str = CONFIG_TEXT, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        load_config.cache_clear()
        return path

    return _write


@pytest.fixture
def tmp_config(write_config):
    return load_config(write_config())


@pytest.fixture
def clean_doc_factory():
    def _make(
        source_id: str = "s1",
        url: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        blocks: tuple[Block, ...] | None = None,
        page_title: str = "HDFC Large Cap Fund - Direct Growth",
    ) -> CleanDocument:
        chosen = blocks if blocks is not None else (
            Block(kind="heading", heading_path="Fees", text="Fees", row_group=None),
            Block(kind="para", heading_path="Fees > Expense ratio", text="Expense ratio: 0.5%", row_group=None),
        )
        return CleanDocument(
            source_id=source_id,
            url=url,
            page_title=page_title,
            blocks=chosen,
            extracted_chars=sum(len(block.text) for block in chosen),
        )

    return _make


@pytest.fixture
def source_spec_factory():
    def _make(source_id: str = "s1", domain: str = "groww.in", **overrides: Any) -> SourceSpec:
        base: dict[str, Any] = {
            "id": source_id,
            "url": f"https://{domain}/mutual-funds/{source_id}",
            "domain": domain,
            "doc_type": "scheme_page",
            "scheme_name": "HDFC Large Cap Fund - Direct Growth",
            "scheme_category": "large_cap",
            "plan": "direct",
            "option": "growth",
            "publisher": "Groww (mirroring HDFC MF official data)",
        }
        base.update(overrides)
        return SourceSpec(**base)

    return _make


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(requests.Session, "request", _blocked)
    import httpx

    monkeypatch.setattr(httpx.Client, "send", _blocked)
    monkeypatch.setattr(httpx.Client, "request", _blocked)


@pytest.fixture
def repo_dotenv() -> Iterator[Path]:
    """Provide a temporary repo-root `.env`, restoring any pre-existing file afterwards."""
    path = Path(ENV_FILE)
    original: bytes | None = path.read_bytes() if path.exists() else None
    yield path
    if original is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(original)


def yaml_without(text: str, *keys: str) -> str:
    raw = yaml.safe_load(text)
    for key in keys:
        raw.pop(key, None)
    return yaml.safe_dump(raw, sort_keys=False)


REAL_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
