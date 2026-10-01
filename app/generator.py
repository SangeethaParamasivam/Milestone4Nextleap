"""Groq generation with the two post-LLM gates and one terminal fallback (implementation §9.6).

Groq is the only LLM provider. Retrieval, guardrails and validation stay offline; this is the only
module that performs a network call, and it never logs the question or the answer text (§9.6.11).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Sequence

from app.cache import cache_key, lookup, store
from app.guard import refusal_answer
from app.models import Answer, Decision, RetrievalHit, RetrievalResult
from app.prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from app.validators import (
    INSUFFICIENT_CONTEXT,
    Violation,
    is_citation_failure,
    is_insufficient_context,
    is_suppressing,
    normalise_url,
    split_answer_lines,
    validate_answer,
)
from config import AppConfig, get_groq_api_key, get_groq_model

_LOG = logging.getLogger("app.generator")

INSUFFICIENT_CONTEXT_STATUS = "insufficient_context"

SAFE_FALLBACK = (
    "I can't give a verified answer from my sources. Please refer to the scheme's official "
    "factsheet and the investor-education resources linked below."
)
MISSING_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
)


class MissingApiKeyError(RuntimeError):
    """Raised when `GROQ_API_KEY` is absent. The UI catches this; `evaluate.py` prints it and exits 2."""


def safe_fallback(cfg: AppConfig) -> Answer:
    """The one terminal fallback of §9.6.9, plus the `advice` educational link."""
    link = cfg.refusal.educational_links.get("advice", "")
    return Answer(
        text=SAFE_FALLBACK,
        source_url=link,
        doc_type="",
        publisher="",
        content_as_of="unknown",
        used_chunk_ids=(),
        status="suppressed",
        was_cached=False,
    )


def answer(
    question: str,
    decision: Decision,
    retrieval: RetrievalResult,
    cfg: AppConfig,
    *,
    use_cache: bool = True,
) -> Answer:
    """Return a grounded `Answer` for a `FACTUAL` question whose retrieval succeeded."""
    if decision.intent != "FACTUAL":
        return refusal_answer(decision)
    if retrieval.status == "out_of_corpus" or not retrieval.hits:
        return Answer(
            text="I don't have that in my sources.",
            source_url="",
            doc_type="",
            publisher="",
            content_as_of="unknown",
            used_chunk_ids=(),
            status="out_of_corpus",
            was_cached=False,
        )

    hits = retrieval.hits[: cfg.retrieval.top_k]
    model = get_groq_model()
    key = cache_key(question, hits, model)
    cached = lookup(key, cfg) if use_cache else None
    if cached is not None:
        _log_call(intent=decision.intent, n_hits=len(hits), latency_ms=0, prompt_tokens=0,
                  completion_tokens=0, violations=(), status="ok", was_cached=True)
        return cached

    started = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    try:
        client = _client()
    except MissingApiKeyError:
        raise

    attempt_text = ""
    for attempt in range(cfg.generation.retries_on_citation_failure + 1):
        user_prompt = (
            build_user_prompt(question, hits)
            if attempt == 0
            else build_retry_prompt(question, hits, attempt_text)
        )
        try:
            completion_text, usage = _call(client, model, user_prompt, cfg)
        except _TransientError:
            completion_text, usage = _call_with_http_retry(client, model, user_prompt, cfg)
        prompt_tokens += usage[0]
        completion_tokens += usage[1]

        violations = validate_answer(completion_text, hits, question, cfg)
        if not violations:
            result = _build_answer(completion_text, hits, cfg, cached=False)
            if use_cache:
                store(key, result, cfg)
            _log_call(
                intent=decision.intent, n_hits=len(hits),
                latency_ms=_elapsed_ms(started), prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, violations=(), status="ok", was_cached=False,
            )
            return result

        if is_citation_failure(violations):
            if attempt < cfg.generation.retries_on_citation_failure:
                attempt_text = _citation_reason(violations)
                continue
            _log_and_fallback(
                decision, hits, started, prompt_tokens, completion_tokens, violations, cfg
            )
            return safe_fallback(cfg)
        if is_insufficient_context(violations):
            return _log_insufficient(
                completion_text, decision, hits, started, prompt_tokens, completion_tokens, violations
            )
        if is_suppressing(violations):
            return _log_and_fallback(
                decision, hits, started, prompt_tokens, completion_tokens, violations, cfg
            )
        result = _build_answer(completion_text, hits, cfg, cached=False)
        if use_cache:
            store(key, result, cfg)
        _log_call(
            intent=decision.intent, n_hits=len(hits),
            latency_ms=_elapsed_ms(started), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            violations=tuple(v.code for v in violations), status="ok", was_cached=False,
        )
        return result

    result = safe_fallback(cfg)
    _log_call(
        intent=decision.intent, n_hits=len(hits), latency_ms=_elapsed_ms(started),
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        violations=("citation_invalid_after_retry",), status=result.status, was_cached=False,
    )
    return result


def _log_and_fallback(
    decision: Decision,
    hits: Sequence[RetrievalHit],
    started: float,
    prompt_tokens: int,
    completion_tokens: int,
    violations: Sequence[Violation],
    cfg: AppConfig,
) -> Answer:
    """The terminal fallback, with its single structured log line (§9.6.9)."""
    result = safe_fallback(cfg)
    _log_call(
        intent=decision.intent, n_hits=len(hits),
        latency_ms=_elapsed_ms(started), prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        violations=tuple(v.code for v in violations),
        status=result.status, was_cached=False,
    )
    return result


def _log_insufficient(
    text: str,
    decision: Decision,
    hits: Sequence[RetrievalHit],
    started: float,
    prompt_tokens: int,
    completion_tokens: int,
    violations: Sequence[Violation],
) -> Answer:
    """The model reported that the retrieved context does not answer the question.

    The model's own sentence is kept because it is more useful than the generic fallback, but the
    status is not `ok`: nothing was answered, and this result is never cached.
    """
    body, _, _ = split_answer_lines(text)
    cited = hits[0].chunk if hits else None
    result = Answer(
        text=body,
        source_url=cited.source_url if cited is not None else "",
        doc_type=cited.doc_type if cited is not None else "",
        publisher=cited.publisher if cited is not None else "",
        content_as_of=(cited.content_as_of if cited is not None and cited.content_as_of else "unknown"),
        used_chunk_ids=tuple(hit.chunk.chunk_id for hit in hits),
        status=INSUFFICIENT_CONTEXT_STATUS,
        was_cached=False,
    )
    _log_call(
        intent=decision.intent, n_hits=len(hits),
        latency_ms=_elapsed_ms(started), prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        violations=tuple(v.code for v in violations),
        status=result.status, was_cached=False,
    )
    return result


class _TransientError(RuntimeError):
    """Marker for a 429/5xx response, which §9.6.6 retries with backoff."""


def _client():
    api_key = get_groq_api_key()
    if not api_key:
        raise MissingApiKeyError(MISSING_KEY_MESSAGE)
    from groq import Groq

    return Groq(api_key=api_key)


def _call(client, model: str, user_prompt: str, cfg: AppConfig) -> tuple[str, tuple[int, int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    generation = cfg.generation
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=generation.temperature,
            max_tokens=generation.max_tokens,
            seed=generation.seed,
        )
    except (TypeError, ValueError) as error:
        if "seed" not in str(error).lower():
            raise
        _LOG.info("seed unsupported by model=%s, retrying without it", model)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=generation.temperature,
            max_tokens=generation.max_tokens,
        )
    except Exception as error:
        if _is_transient(error):
            raise _TransientError(str(error)) from error
        raise
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return response.choices[0].message.content or "", (prompt_tokens, completion_tokens)


def _call_with_http_retry(
    client, model: str, user_prompt: str, cfg: AppConfig
) -> tuple[str, tuple[int, int]]:
    """One retry after `2 s * attempt` plus jitter, then let the fallback happen (§9.6.6)."""
    last: _TransientError | None = None
    for attempt in range(cfg.generation.retries_on_http_error + 1):
        try:
            return _call(client, model, user_prompt, cfg)
        except _TransientError as error:
            last = error
            if attempt < cfg.generation.retries_on_http_error:
                time.sleep(2 * attempt + random.uniform(0, 0.5))
    raise _TransientError(str(last)) from last


def _is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    return isinstance(status, int) and 500 <= status <= 599


def _build_answer(text: str, hits: Sequence[RetrievalHit], cfg: AppConfig, cached: bool) -> Answer:
    """`text` is the body only; `Answer.render()` re-emits the Source and as-of lines in order.

    The model returns the full three-part text, but `Answer.render()` is frozen in `app/models.py`
    and already appends the Source and as-of lines itself, so storing the body alone is what keeps
    the rendered answer free of duplicates (§9.6.10).
    """
    body, source_line, asof_line = split_answer_lines(text)
    cited = normalise_url(source_line) if source_line else ""
    chunk = next(
        (hit.chunk for hit in hits if normalise_url(hit.chunk.source_url) == cited),
        hits[0].chunk,
    )
    return Answer(
        text=body,
        source_url=chunk.source_url,
        doc_type=chunk.doc_type,
        publisher=chunk.publisher,
        content_as_of=asof_line or chunk.content_as_of or "unknown",
        used_chunk_ids=tuple(hit.chunk.chunk_id for hit in hits[: cfg.retrieval.top_k]),
        status="ok",
        was_cached=cached,
    )


def _citation_reason(violations: Sequence[Violation]) -> str:
    reasons = {
        "citation_missing": "cited no source.",
        "citation_not_retrieved": "cited a URL that is not in the provided context.",
        "citation_not_in_allowlist": "cited a URL that is not in the allowlist.",
        "citation_multiple": "cited more than one source.",
    }
    return " ".join(reasons.get(violation.code, "cited an invalid source.") for violation in violations)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_call(
    *,
    intent: str,
    n_hits: int,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    violations: tuple[str, ...],
    status: str,
    was_cached: bool,
) -> None:
    """One structured line per call. Never the question, never the answer text (§9.6.11)."""
    _LOG.info(
        "intent=%s n_hits=%d latency_ms=%d prompt_tokens=%d completion_tokens=%d "
        "violations=%s status=%s cached=%s",
        intent, n_hits, latency_ms, prompt_tokens, completion_tokens,
        ",".join(violations) or "none", status, int(was_cached),
    )
