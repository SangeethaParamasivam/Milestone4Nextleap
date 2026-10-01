"""The single question-to-answer orchestrator (implementation §11.2).

Both the Streamlit UI and `evaluate.py` call `answer_question`, so there is exactly one code path
through guard, retrieval, generation and validation (A9, NFR-8). This module holds no presentation
and no retrieval logic of its own: it sequences the collaborators in `PipelineDeps`, which is also
the injection seam the tests and the evaluation harness stub.

Order is fixed: `classify → (refuse unless FACTUAL) → resolve follow-up wording → retrieve →
(out_of_corpus) → generate`. The conversation memory never runs ahead of the guard - the raw
question is classified first, so a refusal is decided on what the user actually typed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

from app.guard import classify, refusal_answer
from app.generator import MissingApiKeyError, answer
from app.memory import ConversationMemory, resolve_question, rewrite_question
from app.models import Answer, Decision, RetrievalResult
from app.retriever import retrieve
from config import AppConfig
from ingest.vectorstore import get_collection, read_status

_LOG = logging.getLogger("app.pipeline")

OUT_OF_CORPUS_TEXT = "I don't have that in my sources."


@dataclass(frozen=True)
class PipelineDeps:
    retriever: Callable[[str, Decision, AppConfig], RetrievalResult]
    generator: Callable[..., Answer]
    classifier: Callable[[str, AppConfig], Decision]
    rewriter: Callable[[str, ConversationMemory, AppConfig], str]


@dataclass(frozen=True)
class TurnResult:
    """What one turn produced: the wording that was actually searched, the answer, and its evidence.

    The UI needs the `retrieval` to render the sources panel, which `answer_question` alone cannot
    return without breaking its `-> Answer` contract.
    """

    question: str
    answer: Answer
    retrieval: RetrievalResult | None
    intent: str
    reason_code: str


def _default_generator(question: str, decision: Decision, retrieval: RetrievalResult, cfg: AppConfig) -> Answer:
    return answer(question, decision, retrieval, cfg)


DEFAULT_DEPS = PipelineDeps(
    retriever=retrieve,
    generator=_default_generator,
    classifier=classify,
    rewriter=rewrite_question,
)


def answer_question(
    question: str,
    cfg: AppConfig,
    deps: PipelineDeps = DEFAULT_DEPS,
    *,
    memory: ConversationMemory | None = None,
) -> Answer:
    """Answer one question, or explain why there is no answer.

    `memory` is the session window: when it holds a follow-up, the question is rewritten to a
    standalone form before retrieval and the rewrite is what the generator sees. Every early exit
    is an `Answer`, never an exception, so a caller renders refusals and out-of-corpus messages the
    same way it renders a grounded answer.
    """
    return run_turn(question, cfg, deps, memory=memory).answer


def run_turn(
    question: str,
    cfg: AppConfig,
    deps: PipelineDeps = DEFAULT_DEPS,
    *,
    memory: ConversationMemory | None = None,
) -> TurnResult:
    """`answer_question` plus the evidence and the guard outcome behind the answer.

    `classify → (refuse unless FACTUAL) → resolve follow-up wording → retrieve → (out_of_corpus) →
    generate`, with `MissingApiKeyError` turned into a `suppressed` answer so the UI and
    `evaluate.py` render a missing key identically instead of crashing.
    """
    resolved = resolve_question(
        question, memory, cfg, classifier=deps.classifier, rewriter=deps.rewriter
    )
    decision = resolved.decision
    if decision.intent != "FACTUAL":
        _LOG.info("intent=%s reason_code=%s", decision.intent, decision.reason_code)
        return TurnResult(
            question=resolved.question,
            answer=refusal_answer(decision),
            retrieval=None,
            intent=decision.intent,
            reason_code=decision.reason_code,
        )

    result = deps.retriever(resolved.question, decision, cfg)
    if result.status == "out_of_corpus" or not result.hits:
        _LOG.info("intent=%s reason_code=out_of_corpus", decision.intent)
        return TurnResult(
            question=resolved.question,
            answer=Answer(
                text=OUT_OF_CORPUS_TEXT,
                source_url="",
                doc_type="",
                publisher="",
                content_as_of="unknown",
                used_chunk_ids=(),
                status="out_of_corpus",
                was_cached=False,
            ),
            retrieval=result,
            intent=decision.intent,
            reason_code="out_of_corpus",
        )

    try:
        response = deps.generator(resolved.question, decision, result, cfg)
    except MissingApiKeyError as exc:
        _LOG.info("intent=%s reason_code=missing_api_key", decision.intent)
        return TurnResult(
            question=resolved.question,
            answer=Answer(
                text=str(exc),
                source_url="",
                doc_type="",
                publisher="",
                content_as_of="unknown",
                used_chunk_ids=(),
                status="suppressed",
                was_cached=False,
            ),
            retrieval=result,
            intent=decision.intent,
            reason_code="missing_api_key",
        )
    reason = "rewritten_follow_up" if resolved.rewritten else "factual"
    _LOG.info("intent=%s reason_code=%s", decision.intent, reason)
    return TurnResult(
        question=resolved.question,
        answer=response,
        retrieval=result,
        intent=decision.intent,
        reason_code=reason,
    )


def corpus_is_ready(cfg: AppConfig) -> bool:
    """True when `status.json` exists and the collection it describes still holds chunks.

    Both halves matter: a status file without a collection means the store was deleted, and a
    collection without a status file means the run never finished.
    """
    if read_status(cfg) is None:
        return False
    try:
        return get_collection(cfg, create=False).count() > 0
    except Exception as error:  # noqa: BLE001 - a missing or unreadable store is simply not ready
        _LOG.info("corpus not ready reason_code=%s", type(error).__name__)
        return False


def staleness_warning(content_as_of: str, cfg: AppConfig, *, today: date | None = None) -> str | None:
    """A one-line freshness warning, or `None` when the date is fresh, absent or unparseable.

    Never blocks the answer (PRD §14 Q3): an unknown date is not a stale one, so a chunk without a
    `content_as_of` produces no warning. `today` is injectable so the boundary is testable.
    """
    try:
        stamped = date.fromisoformat(content_as_of.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    reference = today if today is not None else date.today()
    if stamped > reference:
        return None
    if (reference - stamped) <= timedelta(days=cfg.freshness_warn_days):
        return None
    return (
        f"These sources were last updated on {stamped.isoformat()}, more than "
        f"{cfg.freshness_warn_days} days ago. Verify the details with the source link."
    )
