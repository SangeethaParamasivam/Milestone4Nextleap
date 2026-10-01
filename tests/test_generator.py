from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Sequence

import pytest

import app.cache as cache_module
from app.cache import cache_key, lookup, store
from app.generator import (
    MISSING_KEY_MESSAGE,
    MissingApiKeyError,
    SAFE_FALLBACK,
    answer,
    safe_fallback,
)
from app.models import Answer, Chunk, Decision, RetrievalHit, RetrievalResult
from app.prompts import SYSTEM_PROMPT, build_context_blocks, build_user_prompt, citation_allowlist
from config import AppConfig, load_config

URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
OTHER_URL = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
CHUNK_TEXT = "The expense ratio of the fund is 1.5% plus additional charges."


def make_hit(url: str = URL, text: str = CHUNK_TEXT, as_of: str = "2026-01-15") -> RetrievalHit:
    chunk = Chunk(
        source_id="s1",
        source_url=url,
        publisher="Groww (mirroring HDFC MF official data)",
        source_domain="groww.in",
        page_title="HDFC Large Cap Fund - Direct Growth",
        doc_type="scheme_page",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        scheme_category="large_cap",
        plan="direct",
        option="growth",
        topic="expense_ratio",
        section_heading="Fees",
        content_as_of=as_of,
        chunk_index=0,
        text=text,
        ingested_at="2026-01-16T00:00:00Z",
        token_count=len(text.split()),
    )
    return RetrievalHit(chunk=chunk, score=0.026, dense_rank=1, lexical_rank=1)


def result(hits: Sequence[RetrievalHit] = ()) -> RetrievalResult:
    return RetrievalResult(
        hits=tuple(hits), status="ok" if hits else "out_of_corpus",
        filtered_topic="expense_ratio", total_candidates=len(hits),
    )


FACTUAL = Decision("FACTUAL", None, "manual")
GOOD = f"The expense ratio of the fund is 1.5%.\nSource: {URL}\nLast updated from sources: 2026-01-15"


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeClient:
    """Returns scripted completions in order; raises `AssertionError` when exhausted."""

    def __init__(self, script: Sequence[str]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = self

    @property
    def completions(self) -> "FakeClient":
        return self

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("unexpected extra LLM call")
        return FakeResponse(self.script.pop(0))


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Never read or write the real `data/cache/answers.jsonl` from a test."""
    cache_module.clear_memory()
    yield
    cache_module.clear_memory()


@pytest.fixture
def uncached_config(real_config: AppConfig, tmp_path):
    """The real config with the answer cache redirected into a temporary directory."""
    paths = dataclasses.replace(real_config.paths, cache_dir=tmp_path / "cache")
    return dataclasses.replace(real_config, paths=paths)


@pytest.fixture
def real_config(uncached_config: AppConfig) -> AppConfig:
    """Shadow the shared fixture so no test here touches the real `data/cache/answers.jsonl`."""
    return uncached_config


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch):
    def _install(script: Sequence[str], *, missing_key: bool = False) -> FakeClient:
        if missing_key:
            def boom() -> Any:
                raise MissingApiKeyError(MISSING_KEY_MESSAGE)

            monkeypatch.setattr("app.generator._client", boom)
            return FakeClient([])
        client = FakeClient(script)
        monkeypatch.setattr("app.generator._client", lambda: client)
        return client

    return _install


# --- prompts ---------------------------------------------------------------------


def test_system_prompt_is_frozen() -> None:
    flat = re.sub(r"\s+", " ", SYSTEM_PROMPT).strip()
    assert flat.startswith(
        "You are a facts-only mutual fund FAQ assistant for a fixed set of HDFC Mutual Fund schemes."
    )
    assert "I don't have that in my sources." in flat
    assert "never state a figure no retrieved chunk shows" in flat
    assert "Keep the answer to at most 3 sentences." in flat
    assert 'End with exactly one line "Source: <url>".' in flat
    assert 'End with "Last updated from sources: <content_as_of>".' in flat


def test_context_blocks_use_the_verbatim_format() -> None:
    blocks = build_context_blocks([make_hit()])
    assert blocks.startswith(f"[1] source_url: {URL}")
    assert "doc_type: scheme_page | publisher: Groww (mirroring HDFC MF official data) | content_as_of: 2026-01-15" in blocks
    assert CHUNK_TEXT in blocks


def test_context_blocks_render_unknown_when_no_chunk_has_a_date() -> None:
    blocks = build_context_blocks([make_hit(as_of="")])
    assert "content_as_of: unknown" in blocks
    assert "Last updated from sources: unknown" in blocks


def test_context_blocks_of_nothing() -> None:
    assert build_context_blocks([]) == ""


def test_user_prompt_carries_the_question_and_the_allowlist() -> None:
    prompt = build_user_prompt("What is the expense ratio?", [make_hit()])
    assert "What is the expense ratio?" in prompt
    assert f"Use only these source URLs: ['{URL}']" in prompt
    assert CHUNK_TEXT in prompt


def test_citation_allowlist_excludes_unconfigured_urls() -> None:
    hits = [make_hit(url=URL), make_hit(url="https://example.com/x")]
    assert citation_allowlist(hits, load_config()) == {URL}


# --- cache -----------------------------------------------------------------------


def test_cache_round_trip(tmp_config: AppConfig) -> None:
    key = cache_key("What is the expense ratio?", [make_hit()], "test-model")
    assert lookup(key, tmp_config) is None
    store(key, safe_fallback(tmp_config), tmp_config)
    assert lookup(key, tmp_config) is None, "suppressed answers must never be cached"
    good = Answer(
        text="body", source_url=URL, doc_type="scheme_page", publisher="p",
        content_as_of="2026-01-15", used_chunk_ids=("c1",), status="ok", was_cached=False,
    )
    store(key, good, tmp_config)
    found = lookup(key, tmp_config)
    assert found is not None and found.text == "body" and found.was_cached


def test_cache_key_depends_on_model_question_and_chunks() -> None:
    base = cache_key("q", [make_hit()], "m1")
    assert base == cache_key("q", [make_hit()], "m1")
    assert base != cache_key("q", [make_hit()], "m2")
    assert base != cache_key("other", [make_hit()], "m1")
    assert base != cache_key("q", [make_hit(url=OTHER_URL)], "m1")


def test_cache_key_is_order_independent_over_chunks(tmp_config: AppConfig) -> None:
    hits = [make_hit(), make_hit(url=OTHER_URL)]
    assert cache_key("q", hits, "m") == cache_key("q", list(reversed(hits)), "m")


def test_cache_file_is_jsonl(tmp_config: AppConfig) -> None:
    key = cache_key("q", [make_hit()], "m")
    store(key, Answer("t", URL, "d", "p", "2026-01-15", ("c",), "ok", False), tmp_config)
    lines = (tmp_config.paths.cache_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["key"] == key


# --- generator -------------------------------------------------------------------


def test_refusal_intent_short_circuits_without_an_llm_call(
    patch_client, real_config: AppConfig
) -> None:
    client = patch_client([])
    refusal = Decision("ADVICE", "https://www.investor.gov.in/", "advice_keyword")
    response = answer("Should I buy this fund?", refusal, result([make_hit()]), real_config)
    assert response.status == "refused_advice"
    assert client.calls == []


def test_out_of_corpus_short_circuits_without_an_llm_call(
    patch_client, real_config: AppConfig
) -> None:
    client = patch_client([])
    response = answer("What is the price of Bitcoin?", FACTUAL, result(), real_config)
    assert response.status == "out_of_corpus"
    assert response.text == "I don't have that in my sources."
    assert client.calls == []


def test_missing_api_key_raises_a_clear_error(patch_client, real_config: AppConfig) -> None:
    patch_client([], missing_key=True)
    with pytest.raises(MissingApiKeyError) as caught:
        answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert str(caught.value) == MISSING_KEY_MESSAGE


def test_valid_answer_is_returned_with_provenance(patch_client, real_config: AppConfig) -> None:
    client = patch_client([GOOD])
    response = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert response.status == "ok"
    assert response.text == "The expense ratio of the fund is 1.5%."
    assert response.source_url == URL
    assert response.doc_type == "scheme_page"
    assert response.content_as_of == "2026-01-15"
    assert response.used_chunk_ids == (make_hit().chunk.chunk_id,)
    assert client.calls[0]["temperature"] == real_config.generation.temperature
    assert client.calls[0]["max_tokens"] == real_config.generation.max_tokens
    assert client.calls[0]["seed"] == real_config.generation.seed
    assert client.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT


def test_second_call_is_served_from_the_cache(patch_client, real_config: AppConfig) -> None:
    client = patch_client([GOOD])
    first = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    second = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert len(client.calls) == 1
    assert second.was_cached is True
    assert second.text == first.text


def test_use_cache_false_bypasses_the_cache(patch_client, real_config: AppConfig) -> None:
    client = patch_client([GOOD, GOOD])
    answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config, use_cache=False)
    assert len(client.calls) == 2


def test_citation_failure_retries_once_then_falls_back(patch_client, real_config: AppConfig) -> None:
    client = patch_client([f"No source here.\nLast updated from sources: 2026-01-15", f"Still nothing.\nLast updated from sources: 2026-01-15"])
    response = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert response.status == "suppressed"
    assert response.text.startswith(SAFE_FALLBACK)
    assert len(client.calls) == 2, "citation failures earn exactly one retry"
    assert "Cite one of these URLs" in client.calls[1]["messages"][1]["content"]


def test_citation_retry_can_succeed(patch_client, real_config: AppConfig) -> None:
    patch_client([f"No source here.\nLast updated from sources: 2026-01-15", GOOD])
    response = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert response.status == "ok"


def test_unsourced_number_suppresses_immediately(patch_client, real_config: AppConfig) -> None:
    client = patch_client([f"The expense ratio is 9.9%.\nSource: {URL}\nLast updated from sources: 2026-01-15"])
    response = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert response.status == "suppressed"
    assert len(client.calls) == 1, "non-citation violations get no retry"


def test_asof_mismatch_suppresses_immediately(patch_client, real_config: AppConfig) -> None:
    client = patch_client([f"The expense ratio is 1.5%.\nSource: {URL}\nLast updated from sources: 1999-01-01"])
    response = answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert response.status == "suppressed"
    assert len(client.calls) == 1


def test_suppressed_result_is_not_cached(patch_client, real_config: AppConfig) -> None:
    client = patch_client([f"The expense ratio is 9.9%.\nSource: {URL}\nLast updated from sources: 2026-01-15"])
    key = cache_key("What is the expense ratio?", [make_hit()], "llama-3.1-8b-instant")
    answer("What is the expense ratio?", FACTUAL, result([make_hit()]), real_config)
    assert len(client.calls) == 1
    assert lookup(key, real_config) is None


def test_safe_fallback_carries_the_educational_link(real_config: AppConfig) -> None:
    fallback = safe_fallback(real_config)
    assert fallback.status == "suppressed"
    assert fallback.source_url == real_config.refusal.educational_links["advice"]
    assert fallback.content_as_of == "unknown"
    assert fallback.text == SAFE_FALLBACK
