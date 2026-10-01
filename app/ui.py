"""Streamlit chat UI (implementation §11.3). Presentation only.

Every question, every answer and every retrieved chunk comes from `app.pipeline`, which owns the
guard, the memory window, retrieval and generation. This module holds no prompt text, no regex and
no threshold, and reaches no client library or pipeline stage of its own: the only imports are
`app.pipeline`, `app.memory`, `app.models`, `app.generator`'s key message, `app.disclaimer` and
`config`.

Run it with:

    .\\.venv\\Scripts\\python.exe -m streamlit run app/ui.py
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from app.disclaimer import DISCLAIMER, SHORT_NOTE
from app.generator import MISSING_KEY_MESSAGE
from app.memory import ConversationMemory
from app.models import RetrievalResult
from app.pipeline import corpus_is_ready, run_turn, staleness_warning
from config import AppConfig, get_groq_api_key, get_groq_model, load_config

_LOG = logging.getLogger("app.ui")

WELCOME = (
    "Ask published facts about five HDFC Mutual Fund schemes: HDFC Large Cap, HDFC Equity "
    "(Flexi Cap), HDFC ELSS Tax Saver, HDFC Small Cap and HDFC Balanced Advantage. Answers come "
    "only from the official scheme pages and are facts-only: no advice, no comparisons, no returns."
)
EXAMPLES = (
    "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "Is there an exit load on HDFC Small Cap Fund - Direct Growth?",
    "What is the riskometer level of HDFC Balanced Advantage Fund?",
)
QUESTION_LABEL = "Ask a factual question about these schemes"
QUESTION_PLACEHOLDER = "e.g. What is the NAV of the HDFC Small Cap Fund?"
MAX_QUESTION_CHARS = 300
NO_CORPUS_WARNING = "Run: python run_ingest.py"
QUESTION_KEY = "question"
PENDING_KEY = "pending_question"

MESSAGES = "messages"
MEMORY = "memory"


@st.cache_resource(show_spinner=False)
def load_assets() -> tuple[AppConfig, bool]:
    """Config and corpus readiness, resolved once per process instead of once per rerun."""
    cfg = load_config()
    return cfg, corpus_is_ready(cfg)


def main() -> None:
    cfg, ready = load_assets()
    if not ready:
        st.warning(NO_CORPUS_WARNING)
        st.stop()
    if not get_groq_api_key():
        st.error(MISSING_KEY_MESSAGE)
        st.stop()
    if st.session_state.get(MESSAGES) is None:
        st.session_state[MESSAGES] = []
    if st.session_state.get(MEMORY) is None:
        st.session_state[MEMORY] = ConversationMemory(cfg.memory.max_messages)

    st.title("Mutual fund FAQ assistant")
    st.caption(WELCOME)
    render_sidebar(cfg)
    render_history(cfg)
    st.caption(SHORT_NOTE)
    question = render_input()
    if question:
        answer_turn(cfg, question)
    st.caption(DISCLAIMER)


def render_sidebar(cfg: AppConfig) -> None:
    st.sidebar.header("Session")
    st.sidebar.caption(f"model: {get_groq_model()}")
    st.sidebar.caption(f"retrieval context window: last {cfg.memory.max_messages} messages")
    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state[MESSAGES] = []
        st.session_state[MEMORY] = ConversationMemory(cfg.memory.max_messages)
        st.rerun()


def render_history(cfg: AppConfig) -> None:
    """Every past turn: the question as typed, the answer, and the sources behind that answer."""
    for turn in st.session_state[MESSAGES]:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        answer = turn["answer"]
        with st.chat_message("assistant"):
            st.markdown(answer.text)
            if answer.source_url:
                st.markdown(f"[Source: {answer.source_url}]({answer.source_url})")
            else:
                st.markdown("Source: none")
            if answer.doc_type:
                st.caption(f"Source document type: {answer.doc_type}")
            as_of = answer.content_as_of or "unknown"
            st.caption(f"Last updated from sources: {as_of}")
            if answer.status == "ok":
                warning = staleness_warning(answer.content_as_of, cfg)
                if warning:
                    st.info(warning)
            render_sources(turn["retrieval"])


def render_sources(retrieval: RetrievalResult | None) -> None:
    """The chunks the answer was built from, behind an expander under the answer."""
    if retrieval is None or not retrieval.hits:
        return
    with st.expander("Sources", expanded=False):
        st.caption(
            f"{len(retrieval.hits)} chunk(s) | topic={retrieval.filtered_topic or 'none'} "
            f"| candidates={retrieval.total_candidates}"
        )
        for rank, hit in enumerate(retrieval.hits, start=1):
            chunk = hit.chunk
            st.markdown(
                f"**{rank}. {chunk.chunk_id}** - score {hit.score:.3f} - {chunk.section_heading}"
            )
            st.caption(
                f"{chunk.source_url} | {chunk.doc_type} | as of {chunk.content_as_of or 'unknown'}"
            )
            st.code(chunk.text, language="text")


def render_input() -> str:
    """Five fixed examples that fill the field, the single question input and the submit button."""
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if column.button(example, use_container_width=True):
            st.session_state[QUESTION_KEY] = example
    st.text_input(
        QUESTION_LABEL,
        placeholder=QUESTION_PLACEHOLDER,
        max_chars=MAX_QUESTION_CHARS,
        key=QUESTION_KEY,
    )
    if st.button("Ask", type="primary", on_click=hand_off_question):
        return st.session_state.pop(PENDING_KEY, "")
    return ""


def hand_off_question() -> None:
    """Move the typed question into the pending slot and empty the field, before the rerun.

    A widget's session-state entry cannot be written once the widget exists in the running
    script, so the submit button hands the question over in a callback instead: the body then
    reads the pending copy, and the next run starts with an empty field.
    """
    st.session_state[PENDING_KEY] = st.session_state.get(QUESTION_KEY, "")
    st.session_state[QUESTION_KEY] = ""


def answer_turn(cfg: AppConfig, question: str) -> None:
    """Run one turn through the pipeline, then show it as the newest entry in the history."""
    question = question.strip()
    if not question:
        return
    memory: ConversationMemory = st.session_state[MEMORY]
    started = time.perf_counter()
    turn = run_turn(question, cfg, memory=memory)
    latency_ms = int((time.perf_counter() - started) * 1000)
    _LOG.info(
        "intent=%s reason_code=%s latency_ms=%d", turn.intent, turn.reason_code, latency_ms
    )
    memory.add_exchange(question, turn.answer.text)
    st.session_state[MESSAGES].append(
        {"question": question, "answer": turn.answer, "retrieval": turn.retrieval}
    )
    st.rerun()


main()
