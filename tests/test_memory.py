from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import pytest

import ask as ask_module
import app.cache as cache_module
from app.memory import (
    ASSISTANT,
    USER,
    ConversationMemory,
    needs_rewrite,
    resolve_question,
    rewrite_question,
)
from app.models import Answer, Chunk, Decision, RetrievalHit, RetrievalResult
from app.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt
from config import AppConfig, load_config

URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
FOLLOW_UP = "what about its fees?"
FOLLOW_UP_REWRITE = "What is the expense ratio of the HDFC Large Cap Fund?"
FIRST_QUESTION = "What is the expense ratio of the HDFC Large Cap Fund?"
FIRST_ANSWER = "The expense ratio of HDFC Large Cap Fund - Direct Growth is 1.03%."


class FakeUsage:
    prompt_tokens = 20
    completion_tokens = 12


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
    """Returns scripted rewrites in order; raises when the script is exhausted."""

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
            raise AssertionError("unexpected extra rewrite call")
        return FakeResponse(self.script.pop(0))


class ExplodingClient(FakeClient):
    def create(self, **kwargs: Any) -> FakeResponse:
        raise RuntimeError("groq is down")


@pytest.fixture(autouse=True)
def _isolated_cache():
    cache_module.clear_memory()
    yield
    cache_module.clear_memory()


@pytest.fixture
def uncached_config(real_config: AppConfig, tmp_path) -> AppConfig:
    paths = dataclasses.replace(real_config.paths, cache_dir=tmp_path / "cache")
    return dataclasses.replace(real_config, paths=paths)


@pytest.fixture
def real_config(uncached_config: AppConfig) -> AppConfig:
    return uncached_config


@pytest.fixture
def patch_rewriter(monkeypatch: pytest.MonkeyPatch):
    def _install(script: Sequence[str], *, broken: bool = False) -> FakeClient:
        client = ExplodingClient([]) if broken else FakeClient(script)
        monkeypatch.setattr("app.generator._client", lambda: client)
        return client

    return _install


def seeded_memory(max_messages: int = 10) -> ConversationMemory:
    memory = ConversationMemory(max_messages)
    memory.add_exchange(FIRST_QUESTION, FIRST_ANSWER)
    return memory


def make_hit() -> RetrievalHit:
    chunk = Chunk(
        source_id="s1",
        source_url=URL,
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
        content_as_of="2026-09-29",
        chunk_index=0,
        text="Expense ratio 1.03% (HDFC Large Cap Fund - Direct Growth)",
        ingested_at="2026-09-29T00:00:00Z",
        token_count=9,
    )
    return RetrievalHit(chunk=chunk, score=0.026, dense_rank=1, lexical_rank=1)


# --- the window --------------------------------------------------------------------


def test_memory_keeps_the_last_ten_messages() -> None:
    memory = ConversationMemory(max_messages=10)
    for index in range(6):
        memory.add_exchange(f"question {index}", f"answer {index}")
    assert len(memory) == 10
    assert [message.role for message in memory.messages] == [USER, ASSISTANT] * 5
    assert memory.messages[0].text == "question 1"
    assert memory.messages[-1].text == "answer 5"


def test_memory_uses_the_configured_window(real_config: AppConfig) -> None:
    assert ConversationMemory(real_config.memory.max_messages).max_messages == 10


def test_memory_rejects_a_zero_window() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(0)


def test_memory_normalises_and_drops_blanks() -> None:
    memory = ConversationMemory(10)
    memory.add(USER, "  what   about\nits fees? ")
    memory.add(ASSISTANT, "   ")
    assert memory.messages[0].text == "what about its fees?"
    assert len(memory) == 1


def test_memory_clear_and_bool() -> None:
    memory = seeded_memory()
    assert memory
    memory.clear()
    assert not memory
    assert memory.messages == ()


# --- when a rewrite is requested ---------------------------------------------------


def test_needs_rewrite_is_false_without_history(real_config: AppConfig) -> None:
    assert needs_rewrite(FOLLOW_UP, None, real_config) is False
    assert needs_rewrite(FOLLOW_UP, ConversationMemory(10), real_config) is False


def test_needs_rewrite_is_false_for_a_standalone_question(real_config: AppConfig) -> None:
    assert needs_rewrite(FIRST_QUESTION, seeded_memory(), real_config) is False


def test_needs_rewrite_is_true_for_a_deictic_follow_up(real_config: AppConfig) -> None:
    memory = seeded_memory()
    assert needs_rewrite(FOLLOW_UP, memory, real_config) is True
    assert needs_rewrite("And its lock-in?", memory, real_config) is True
    assert needs_rewrite("Is the same true for the small cap fund?", memory, real_config) is True


def test_needs_rewrite_is_true_for_a_too_short_question(real_config: AppConfig) -> None:
    assert needs_rewrite("NAV?", seeded_memory(), real_config) is True


def test_needs_rewrite_respects_whole_words(real_config: AppConfig) -> None:
    assert needs_rewrite("What is the exit load?", seeded_memory(), real_config) is False


def test_needs_rewrite_is_false_when_disabled(real_config: AppConfig) -> None:
    disabled = dataclasses.replace(real_config, memory=dataclasses.replace(real_config.memory, rewrite=False))
    assert needs_rewrite(FOLLOW_UP, seeded_memory(), disabled) is False


# --- the rewrite itself -------------------------------------------------------------


def test_rewrite_returns_the_question_unchanged_without_a_client(
    real_config: AppConfig, patch_rewriter
) -> None:
    client = patch_rewriter([])  # any call would raise AssertionError
    assert rewrite_question(FOLLOW_UP, None, real_config) == FOLLOW_UP
    assert client.calls == []


def test_rewrite_resolves_the_follow_up_from_history(
    real_config: AppConfig, patch_rewriter
) -> None:
    client = patch_rewriter([FOLLOW_UP_REWRITE])
    rewritten = rewrite_question(FOLLOW_UP, seeded_memory(), real_config)
    assert rewritten == FOLLOW_UP_REWRITE
    assert client.calls[0]["model"] == real_config.groq_model
    assert client.calls[0]["messages"][0]["content"] == REWRITE_SYSTEM_PROMPT
    prompt = client.calls[0]["messages"][1]["content"]
    assert f"User: {FIRST_QUESTION}" in prompt
    assert f"Assistant: {FIRST_ANSWER}" in prompt
    assert f"Follow-up question: {FOLLOW_UP}" in prompt


def test_rewrite_strips_labels_quotes_and_commentary(
    real_config: AppConfig, patch_rewriter
) -> None:
    patch_rewriter(
        [
            'Question: "What is the expense ratio of the HDFC Large Cap Fund?"\n'
            "I resolved 'its' to the previously discussed fund."
        ]
    )
    assert rewrite_question(FOLLOW_UP, seeded_memory(), real_config) == FOLLOW_UP_REWRITE


def test_rewrite_truncates_an_overlong_reply(real_config: AppConfig, patch_rewriter) -> None:
    long_cfg = dataclasses.replace(
        real_config, memory=dataclasses.replace(real_config.memory, max_rewrite_chars=40)
    )
    patch_rewriter(["What is the expense ratio of the HDFC Large Cap Fund today exactly?"])
    rewritten = rewrite_question(FOLLOW_UP, seeded_memory(), long_cfg)
    assert len(rewritten) <= 40
    assert rewritten == "What is the expense ratio of the HDFC"


def test_rewrite_falls_back_to_the_question_on_an_empty_reply(
    real_config: AppConfig, patch_rewriter
) -> None:
    patch_rewriter(["   \n  "])
    assert rewrite_question(FOLLOW_UP, seeded_memory(), real_config) == FOLLOW_UP


def test_rewrite_falls_back_when_the_model_repeats_the_question(
    real_config: AppConfig, patch_rewriter
) -> None:
    patch_rewriter([FOLLOW_UP])
    assert rewrite_question(FOLLOW_UP, seeded_memory(), real_config) == FOLLOW_UP


def test_rewrite_falls_back_when_the_call_fails(real_config: AppConfig, patch_rewriter) -> None:
    patch_rewriter([], broken=True)
    assert rewrite_question(FOLLOW_UP, seeded_memory(), real_config) == FOLLOW_UP


def test_rewrite_sends_only_the_kept_window(real_config: AppConfig, patch_rewriter) -> None:
    memory = seeded_memory()
    for index in range(5):
        memory.add_exchange(f"later question {index}", f"later answer {index}")
    client = patch_rewriter([FOLLOW_UP_REWRITE])
    rewrite_question(FOLLOW_UP, memory, real_config)
    prompt = client.calls[0]["messages"][1]["content"]
    assert FIRST_QUESTION not in prompt
    assert "later question 4" in prompt
    assert prompt.count("User:") == 5


# --- resolution order ----------------------------------------------------------------


def test_resolve_rewrites_a_follow_up_and_reclassifies(real_config: AppConfig, patch_rewriter) -> None:
    patch_rewriter([FOLLOW_UP_REWRITE])
    resolved = resolve_question(FOLLOW_UP, seeded_memory(), real_config)
    assert resolved.original == FOLLOW_UP
    assert resolved.question == FOLLOW_UP_REWRITE
    assert resolved.rewritten is True
    assert resolved.decision.intent == "FACTUAL"


def test_resolve_leaves_a_standalone_question_alone(real_config: AppConfig, patch_rewriter) -> None:
    client = patch_rewriter([])
    resolved = resolve_question(FIRST_QUESTION, seeded_memory(), real_config)
    assert resolved.question == FIRST_QUESTION
    assert resolved.rewritten is False
    assert client.calls == []


def test_resolve_refuses_on_the_raw_question_before_any_rewrite(
    real_config: AppConfig, patch_rewriter
) -> None:
    client = patch_rewriter([])
    for question, intent in [
        ("what about its 3 year returns?", "PERFORMANCE"),
        ("should i buy its sibling fund?", "ADVICE"),
        ("what is its PAN number?", "PII"),
    ]:
        resolved = resolve_question(question, seeded_memory(), real_config)
        assert resolved.decision.intent == intent
        assert resolved.rewritten is False
    assert client.calls == []


def test_resolve_refuses_a_rewrite_that_resolves_out_of_scope(
    real_config: AppConfig, patch_rewriter
) -> None:
    patch_rewriter(["What is the expense ratio of the Parag Parflex Fund?"])
    resolved = resolve_question(FOLLOW_UP, seeded_memory(), real_config)
    assert resolved.rewritten is True
    assert resolved.decision.intent == "OUT_OF_CORPUS"


# --- prompt construction --------------------------------------------------------------


def test_build_rewrite_prompt_labels_every_turn() -> None:
    prompt = build_rewrite_prompt(FOLLOW_UP, ((USER, "q1"), (ASSISTANT, "a1")))
    assert prompt.index("User: q1") < prompt.index("Assistant: a1") < prompt.index(FOLLOW_UP)
    assert "Standalone version of the follow-up question:" in prompt


# --- CLI wiring --------------------------------------------------------------------


def test_ask_once_answers_the_follow_up_with_the_rewritten_question(
    real_config: AppConfig, patch_rewriter, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    patch_rewriter([FOLLOW_UP_REWRITE])
    seen: dict[str, Any] = {}

    def fake_retrieve(question: str, decision: Decision, cfg: AppConfig) -> RetrievalResult:
        seen["question"] = question
        return RetrievalResult(hits=(make_hit(),), status="ok", filtered_topic="expense_ratio", total_candidates=1)

    def fake_answer(question: str, decision: Decision, retrieval: RetrievalResult, cfg: AppConfig, **kwargs: Any) -> Answer:
        seen["answered"] = question
        return Answer(
            text="The expense ratio of HDFC Large Cap Fund - Direct Growth is 1.03%.",
            source_url=URL,
            doc_type="scheme_page",
            publisher="Groww",
            content_as_of="2026-09-29",
            used_chunk_ids=("s1::expense_ratio::9083d8ab",),
            status="ok",
            was_cached=False,
        )

    monkeypatch.setattr(ask_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(ask_module, "answer", fake_answer)

    memory = seeded_memory()
    code = ask_module.ask_once(FOLLOW_UP, real_config, False, True, memory, True)
    assert code == 0
    assert seen["question"] == FOLLOW_UP_REWRITE
    assert seen["answered"] == FOLLOW_UP_REWRITE
    out = capsys.readouterr().out
    assert "[rewrite] what about its fees? -> What is the expense ratio of the HDFC Large Cap Fund?" in out
    assert "1.03%" in out
    # the typed wording is what a later pronoun refers to, so history keeps the user's own words
    assert memory.messages[-2].text == FOLLOW_UP
    assert len(memory) == 4


def test_ask_once_without_memory_never_rewrites(
    real_config: AppConfig, patch_rewriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = patch_rewriter([])
    seen: dict[str, Any] = {}

    def fake_retrieve(question: str, decision: Decision, cfg: AppConfig) -> RetrievalResult:
        seen["question"] = question
        return RetrievalResult(hits=(make_hit(),), status="ok", filtered_topic=None, total_candidates=1)

    monkeypatch.setattr(ask_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        ask_module,
        "answer",
        lambda question, decision, retrieval, cfg, **kwargs: Answer(
            text="ok", source_url=URL, doc_type="scheme_page", publisher="Groww",
            content_as_of="2026-09-29", used_chunk_ids=(), status="ok", was_cached=False,
        ),
    )
    assert ask_module.ask_once(FOLLOW_UP, real_config, False, True) == 0
    assert seen["question"] == FOLLOW_UP
    assert client.calls == []


def test_ask_once_refuses_a_follow_up_that_is_a_refusal(
    real_config: AppConfig, patch_rewriter, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    client = patch_rewriter([])

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("retrieval must not run for a refused question")

    monkeypatch.setattr(ask_module, "retrieve", boom)
    memory = seeded_memory()
    assert ask_module.ask_once("what about its 3 year returns?", real_config, False, True, memory) == 0
    assert "I don't state or compute returns." in capsys.readouterr().out
    assert client.calls == []
    assert len(memory) == 2  # a refusal states no subject, so nothing is stored


def test_render_memory_shows_the_window() -> None:
    memory = seeded_memory()
    rendered = ask_module.render_memory(memory)
    assert rendered.startswith("[memory] 2/10 messages")
    assert "[1] user: What is the expense ratio" in rendered
    assert ask_module.render_memory(ConversationMemory(10)) == "[memory] empty"


def test_memory_section_is_configurable(write_config) -> None:
    path = write_config()
    text = path.read_text(encoding="utf-8").replace(
        "generation:", "memory: {max_messages: 4, rewrite: false, deictic_triggers: [that]}\ngeneration:", 1
    )
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.memory.max_messages == 4
    assert cfg.memory.rewrite is False
    assert cfg.memory.deictic_triggers == ("that",)


def test_memory_defaults_apply_without_a_config_section(write_config) -> None:
    path = write_config()
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.memory.max_messages == 10
    assert cfg.memory.rewrite is True
    assert "its" in cfg.memory.deictic_triggers


def test_env_overrides_the_memory_window(monkeypatch: pytest.MonkeyPatch, write_config) -> None:
    write_config()
    monkeypatch.setenv("MEMORY_MAX_MESSAGES", "3")
    load_config.cache_clear()
    assert load_config().memory.max_messages == 3
