from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app.cache as cache_module
from app.generator import MISSING_KEY_MESSAGE, MissingApiKeyError
from app.memory import ConversationMemory
from app.models import Answer, Chunk, Decision, RetrievalHit, RetrievalResult
from app.pipeline import (
    DEFAULT_DEPS,
    OUT_OF_CORPUS_TEXT,
    PipelineDeps,
    TurnResult,
    answer_question,
    corpus_is_ready,
    run_turn,
    staleness_warning,
)
from config import AppConfig

URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
QUESTION = "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?"
FOLLOW_UP = "what about its fund size?"
FOLLOW_UP_REWRITE = "What is the fund size of the HDFC Large Cap Fund?"
UI_PATH = Path(__file__).resolve().parent.parent / "app" / "ui.py"


@pytest.fixture(autouse=True)
def _isolated_cache():
    cache_module.clear_memory()
    yield
    cache_module.clear_memory()


@pytest.fixture
def real_config(real_config: AppConfig, tmp_path) -> AppConfig:
    paths = dataclasses.replace(real_config.paths, cache_dir=tmp_path / "cache")
    return dataclasses.replace(real_config, paths=paths)


def make_hit(chunk_id: str = "s1::expense_ratio::9083d8ab") -> RetrievalHit:
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
        section_heading="Expense ratio",
        content_as_of="2026-09-29",
        chunk_index=0,
        text="Expense ratio 1.03% (HDFC Large Cap Fund - Direct Growth)",
        ingested_at="2026-09-29T00:00:00Z",
        token_count=9,
    )
    return RetrievalHit(chunk=chunk, score=0.026, dense_rank=1, lexical_rank=1)


def ok_answer() -> Answer:
    return Answer(
        text="The expense ratio for HDFC Large Cap Fund - Direct Growth is 1.03%.",
        source_url=URL,
        doc_type="scheme_page",
        publisher="Groww (mirroring HDFC MF official data)",
        content_as_of="2026-09-29",
        used_chunk_ids=("s1::expense_ratio::9083d8ab",),
        status="ok",
        was_cached=False,
    )


def make_deps(
    log: list[str],
    *,
    intent: str = "FACTUAL",
    hits: tuple[RetrievalHit, ...] = (make_hit(),),
    generator_error: Exception | None = None,
    rewritten: str | None = None,
) -> PipelineDeps:
    def classifier(question: str, cfg: AppConfig) -> Decision:
        log.append("classify")
        return Decision(intent, None, "stub")

    def retriever(question: str, decision: Decision, cfg: AppConfig) -> RetrievalResult:
        log.append("retrieve")
        return RetrievalResult(
            hits=hits,
            status="ok" if hits else "out_of_corpus",
            filtered_topic="expense_ratio",
            total_candidates=len(hits) or 20,
        )

    def generator(question: str, decision: Decision, retrieval: RetrievalResult, cfg: AppConfig) -> Answer:
        log.append("generate")
        if generator_error is not None:
            raise generator_error
        return ok_answer()

    def rewriter(question: str, memory: ConversationMemory, cfg: AppConfig) -> str:
        log.append("rewrite")
        return rewritten if rewritten is not None else question

    return PipelineDeps(retriever=retriever, generator=generator, classifier=classifier, rewriter=rewriter)


# --- ordering ----------------------------------------------------------------------


def test_happy_path_order(real_config: AppConfig) -> None:
    log: list[str] = []
    result = answer_question(QUESTION, real_config, make_deps(log))
    assert log == ["classify", "retrieve", "generate"]
    assert result.status == "ok"
    assert result.text == ok_answer().text


def test_refusal_short_circuits(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, intent="ADVICE")
    result = answer_question("Should I buy HDFC Large Cap Fund?", real_config, deps)
    assert log == ["classify"]
    assert result.status == "refused_advice"
    assert "recommend" in result.text


def test_out_of_corpus_short_circuits(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, hits=())
    result = answer_question("Who won the 2024 cricket world cup?", real_config, deps)
    assert log == ["classify", "retrieve"]
    assert result.status == "out_of_corpus"
    assert result.text == OUT_OF_CORPUS_TEXT


def test_missing_key_becomes_a_suppressed_answer(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, generator_error=MissingApiKeyError(MISSING_KEY_MESSAGE))
    result = answer_question(QUESTION, real_config, deps)
    assert log == ["classify", "retrieve", "generate"]
    assert result.status == "suppressed"
    assert result.text == MISSING_KEY_MESSAGE


def test_default_deps_are_the_real_collaborators() -> None:
    assert DEFAULT_DEPS.classifier.__name__ == "classify"
    assert DEFAULT_DEPS.retriever.__name__ == "retrieve"
    assert DEFAULT_DEPS.rewriter.__name__ == "rewrite_question"


# --- the memory window through the orchestrator --------------------------------------


def test_follow_up_is_rewritten_before_retrieval(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, rewritten=FOLLOW_UP_REWRITE)
    memory = ConversationMemory(10)
    memory.add_exchange(QUESTION, ok_answer().text)
    turn = run_turn(FOLLOW_UP, real_config, deps, memory=memory)
    assert log == ["classify", "rewrite", "classify", "retrieve", "generate"]
    assert turn.question == FOLLOW_UP_REWRITE
    assert turn.answer.status == "ok"
    assert turn.retrieval is not None


def test_standalone_question_is_never_rewritten(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, rewritten=FOLLOW_UP_REWRITE)
    memory = ConversationMemory(10)
    memory.add_exchange(QUESTION, ok_answer().text)
    run_turn(QUESTION, real_config, deps, memory=memory)
    assert log == ["classify", "retrieve", "generate"]


def test_a_refused_follow_up_never_reaches_the_rewriter(real_config: AppConfig) -> None:
    log: list[str] = []
    deps = make_deps(log, intent="PERFORMANCE", rewritten=FOLLOW_UP_REWRITE)
    memory = ConversationMemory(10)
    memory.add_exchange(QUESTION, ok_answer().text)
    turn = run_turn("what about its 3 year returns?", real_config, deps, memory=memory)
    assert log == ["classify"]
    assert turn.answer.status == "refused_performance"
    assert turn.retrieval is None


def test_turn_result_carries_the_evidence(real_config: AppConfig) -> None:
    log: list[str] = []
    turn = run_turn(QUESTION, real_config, make_deps(log))
    assert turn.intent == "FACTUAL"
    assert turn.reason_code == "factual"
    assert turn.retrieval is not None
    assert [hit.chunk.chunk_id for hit in turn.retrieval.hits] == ["s1::expense_ratio::9083d8ab"]
    assert turn.answer.source_url == URL


def test_refusal_turn_has_no_context_panel_data(real_config: AppConfig) -> None:
    log: list[str] = []
    turn = run_turn("should i buy it?", real_config, make_deps(log, intent="ADVICE"))
    assert turn.retrieval is None
    assert turn.reason_code == "stub"


# --- corpus readiness ----------------------------------------------------------------


def test_corpus_is_ready_false_without_status(real_config: AppConfig, tmp_path) -> None:
    paths = dataclasses.replace(real_config.paths, status_file=tmp_path / "missing.json")
    assert corpus_is_ready(dataclasses.replace(real_config, paths=paths)) is False


def test_corpus_is_ready_true_with_status_and_chunks(real_config: AppConfig, tmp_path, monkeypatch) -> None:
    status_file = tmp_path / "status.json"
    status_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("app.pipeline.read_status", lambda cfg: object())
    monkeypatch.setattr("app.pipeline.get_collection", lambda cfg, create: _FakeCollection(215))
    paths = dataclasses.replace(real_config.paths, status_file=status_file)
    assert corpus_is_ready(dataclasses.replace(real_config, paths=paths)) is True


def test_corpus_is_ready_false_when_the_store_is_gone(real_config: AppConfig, monkeypatch) -> None:
    def boom(cfg: AppConfig, *, create: bool) -> Any:
        raise RuntimeError("collection not found")

    monkeypatch.setattr("app.pipeline.read_status", lambda cfg: object())
    monkeypatch.setattr("app.pipeline.get_collection", boom)
    assert corpus_is_ready(real_config) is False


def test_corpus_is_ready_false_for_an_empty_collection(real_config: AppConfig, monkeypatch) -> None:
    monkeypatch.setattr("app.pipeline.read_status", lambda cfg: object())
    monkeypatch.setattr("app.pipeline.get_collection", lambda cfg, create: _FakeCollection(0))
    assert corpus_is_ready(real_config) is False


class _FakeCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


# --- freshness ----------------------------------------------------------------------


def test_staleness_warning_only_past_the_configured_window(real_config: AppConfig) -> None:
    from datetime import date

    today = date(2026, 9, 30)
    assert staleness_warning("2026-09-29", real_config, today=today) is None
    assert staleness_warning("2026-01-01", real_config, today=today) is not None


def test_staleness_warning_ignores_unknown_and_future_dates(real_config: AppConfig) -> None:
    from datetime import date

    today = date(2026, 9, 30)
    assert staleness_warning("unknown", real_config, today=today) is None
    assert staleness_warning("", real_config, today=today) is None
    assert staleness_warning("29 Sep 2026", real_config, today=today) is None
    assert staleness_warning("2027-01-01", real_config, today=today) is None


# --- the Streamlit surface (implementation §11.3) --------------------------------------

UI_ALLOWED_IMPORTS = frozenset(
    {
        "app.disclaimer",
        "app.generator",
        "app.memory",
        "app.models",
        "app.pipeline",
        "config",
        "logging",
        "streamlit",
        "time",
        "__future__",
    }
)


def refusal_turn(question: str) -> TurnResult:
    return TurnResult(
        question=question,
        answer=Answer(
            text=(
                "I answer only published facts about these schemes and can't recommend, compare or "
                "rate funds or suggest a portfolio."
            ),
            source_url="https://www.investor.gov.in/",
            doc_type="",
            publisher="",
            content_as_of="unknown",
            used_chunk_ids=(),
            status="refused_advice",
            was_cached=False,
        ),
        retrieval=None,
        intent="ADVICE",
        reason_code="advice:recommend",
    )


def grounded_turn(question: str, *, content_as_of: str = "2026-09-29") -> TurnResult:
    return TurnResult(
        question=question,
        answer=dataclasses.replace(ok_answer(), content_as_of=content_as_of),
        retrieval=make_retrieval(),
        intent="FACTUAL",
        reason_code="factual",
    )


def make_retrieval() -> RetrievalResult:
    return RetrievalResult(
        hits=(make_hit(),),
        status="ok",
        filtered_topic="expense_ratio",
        total_candidates=20,
    )


def build_ui(
    monkeypatch: pytest.MonkeyPatch,
    turn: TurnResult,
    *,
    corpus_ready: bool = True,
    api_key: str | None = "test-key",
) -> AppTest:
    """Run `app/ui.py` headlessly with the corpus, the key and the pipeline stubbed out."""
    st.cache_resource.clear()
    monkeypatch.setattr("app.pipeline.corpus_is_ready", lambda cfg: corpus_ready)
    monkeypatch.setattr("config.get_groq_api_key", lambda: api_key)
    monkeypatch.setattr("app.pipeline.run_turn", lambda q, cfg, deps=DEFAULT_DEPS, memory=None: turn)
    app = AppTest.from_file(str(UI_PATH), default_timeout=30)
    app.run()
    assert not app.exception
    return app


def ask(app: AppTest, question: str) -> None:
    app.text_input(key="question").set_value(question)
    [button for button in app.button if button.label == "Ask"][0].click().run()
    assert not app.exception


def test_ui_startup_renders_the_required_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, refusal_turn(QUESTION))
    assert [title.value for title in app.title] == ["Mutual fund FAQ assistant"]
    captions = [caption.value for caption in app.caption]
    assert any("HDFC Large Cap" in caption and "facts-only" in caption for caption in captions)
    assert "Facts-only. No investment advice." in captions
    assert any(caption.startswith("Facts-only. No investment advice. This assistant") for caption in captions)
    labels = [button.label for button in app.button]
    assert labels.count("What is the expense ratio of HDFC Large Cap Fund - Direct Growth?") == 1
    assert labels.count("What is the lock-in period in HDFC ELSS Tax Saver Fund?") == 1
    assert labels.count("How do I download my capital gains statement?") == 1
    assert labels.count("Is there an exit load on HDFC Small Cap Fund - Direct Growth?") == 1
    assert labels.count("What is the riskometer level of HDFC Balanced Advantage Fund?") == 1
    assert len([label for label in labels if label.endswith("?")]) == 5
    assert app.text_input[0].label == "Ask a factual question about these schemes"
    assert app.text_input[0].max_chars == 300
    assert [button.label for button in app.sidebar.button] == ["Clear chat"]


def test_ui_answer_block_shows_the_citation_type_and_date(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION))
    ask(app, QUESTION)
    assistant = [message for message in app.chat_message if message.name == "assistant"][0]
    assert assistant.markdown[0].value == ok_answer().text
    assert assistant.markdown[1].value == f"[Source: {URL}]({URL})"
    captions = [caption.value for caption in app.caption]
    assert "Source document type: scheme_page" in captions
    assert "Last updated from sources: 2026-09-29" in captions
    assert app.text_input[0].value == ""


def test_ui_shows_a_sources_expander_under_every_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION))
    ask(app, QUESTION)
    assert [expander.label for expander in app.expander] == ["Sources"]
    markdown = [element.value for element in app.markdown]
    assert any("score 0.026" in value and "Expense ratio" in value for value in markdown)
    captions = [caption.value for caption in app.caption]
    assert f"{URL} | scheme_page | as of 2026-09-29" in captions
    assert [block.value for block in app.code] == [make_hit().chunk.text]


def test_ui_keeps_every_turn_in_the_history(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION))
    ask(app, QUESTION)
    ask(app, FOLLOW_UP)
    assert [(message.name, message.markdown[0].value) for message in app.chat_message] == [
        ("user", QUESTION),
        ("assistant", ok_answer().text),
        ("user", FOLLOW_UP),
        ("assistant", ok_answer().text),
    ]
    assert len(app.expander) == 2
    assert len(app.session_state["messages"]) == 2


def test_ui_refusal_has_no_context_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, refusal_turn("Should I buy HDFC Large Cap Fund?"))
    ask(app, "Should I buy HDFC Large Cap Fund?")
    assistant = [message for message in app.chat_message if message.name == "assistant"][0]
    assert "can't recommend" in assistant.markdown[0].value
    assert assistant.markdown[1].value == "[Source: https://www.investor.gov.in/](https://www.investor.gov.in/)"
    assert not app.expander
    assert "Last updated from sources: unknown" in [caption.value for caption in app.caption]


def test_ui_shows_unknown_when_the_cited_chunk_has_no_date(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION, content_as_of=""))
    ask(app, QUESTION)
    assert "Last updated from sources: unknown" in [caption.value for caption in app.caption]
    assert not app.info


def test_ui_session_memory_is_a_ten_message_window(monkeypatch: pytest.MonkeyPatch, real_config: AppConfig) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION))
    ask(app, QUESTION)
    memory = app.session_state["memory"]
    assert isinstance(memory, ConversationMemory)
    assert memory.max_messages == 10 == real_config.memory.max_messages
    assert memory.turns() == (("user", QUESTION), ("assistant", ok_answer().text))
    assert any("last 10 messages" in caption.value for caption in app.sidebar.caption)


def test_ui_clear_chat_empties_the_history_and_the_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION))
    ask(app, QUESTION)
    assert app.session_state["messages"]
    [button for button in app.sidebar.button if button.label == "Clear chat"][0].click().run()
    assert not app.exception
    assert not app.chat_message
    assert app.session_state["messages"] == []
    assert len(app.session_state["memory"]) == 0
    assert not app.expander


def test_ui_warns_when_the_corpus_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, refusal_turn(QUESTION), corpus_ready=False)
    assert [warning.value for warning in app.warning] == ["Run: python run_ingest.py"]
    assert not app.chat_message


def test_ui_stops_when_the_api_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, refusal_turn(QUESTION), api_key=None)
    assert [error.value for error in app.error] == [MISSING_KEY_MESSAGE]
    assert not app.chat_message


def test_ui_shows_a_stale_source_warning_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_ui(monkeypatch, grounded_turn(QUESTION, content_as_of="2020-01-01"))
    ask(app, QUESTION)
    assert [info.value for info in app.info] == [
        "These sources were last updated on 2020-01-01, more than 90 days ago. Verify the details "
        "with the source link."
    ]
    assert [expander.label for expander in app.expander] == ["Sources"]


def test_ui_imports_no_client_library() -> None:
    roots = set()
    for line in UI_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            roots.add(stripped[len("import ") :].split(" as ")[0].split(".")[0].strip())
        elif stripped.startswith("from ") and " import " in stripped:
            roots.add(stripped[len("from ") :].split(" import ")[0].strip())
    assert roots <= UI_ALLOWED_IMPORTS
    assert not roots & {"chromadb", "groq", "requests", "re", "httpx"}
