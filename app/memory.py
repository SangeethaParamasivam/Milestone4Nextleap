"""Conversation memory and follow-up question rewriting for the interactive CLI.

`ConversationMemory` keeps the last `cfg.memory.max_messages` user/assistant messages of the
session, so a follow-up such as "what about its fees?" can be retrieved as "What is the expense
ratio of the HDFC Large Cap Fund?" instead of an unresolvable pronoun soup.

Three rules keep this safe and cheap:

* the deterministic guard always sees the raw question first, so PII, advice, performance and
  out-of-corpus refusals are decided on what the user actually typed, never on a rewrite;
* a rewrite is requested only when the session has history and the question is elliptical - it
  carries a configured deictic trigger, or is too short to stand alone - so an ordinary question
  is retrieved exactly as typed and costs no extra call;
* any failure, empty reply or unusable rewrite falls back to the original question, so memory can
  never turn a working answer into a failure.

The rewrite is a Groq call like the answer itself, and reuses the single client factory in
`app.generator`; this module adds no new network dependency of its own.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.guard import classify, normalise_question
from app.models import Decision
from app.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt
from config import AppConfig, get_groq_model

_LOG = logging.getLogger("app.memory")

USER = "user"
ASSISTANT = "assistant"

_LEADING_LABELS = re.compile(
    r"^(?:standalone\s+version\s+of\s+the\s+follow[- ]?up\s+question|rewritten\s+question|"
    r"standalone\s+question|question|follow[- ]?up|q)\s*[:\-]\s*",
    re.IGNORECASE,
)
_TRIGGER_PATTERNS: dict[str, re.Pattern[str]] = {}


@dataclass(frozen=True)
class Message:
    role: str
    text: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.role, self.text)


@dataclass(frozen=True)
class ResolvedQuestion:
    """What the pipeline should run on, and how it got there."""

    original: str
    question: str
    decision: Decision
    rewritten: bool = False


class ConversationMemory:
    """The last `max_messages` messages of the session, oldest dropped first.

    A `deque(maxlen=...)` is used so the cap holds on the data structure itself: a caller cannot
    grow the window by accident, and ten messages is five question/answer pairs, which is the
    whole point of remembering the previous exchange rather than a transcript.
    """

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        self.max_messages = max_messages
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, role: str, text: str) -> None:
        cleaned = " ".join(text.split())
        if cleaned:
            self._messages.append(Message(role=role, text=cleaned))

    def add_exchange(self, question: str, answer: str) -> None:
        """Store the question exactly as the user typed it, next to the answer it produced.

        The typed wording is what a later pronoun refers to, so a rewrite is never stored in its
        place - that would make the history drift away from the conversation.
        """
        self.add(USER, question)
        self.add(ASSISTANT, answer)

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def turns(self) -> tuple[tuple[str, str], ...]:
        return tuple(message.as_tuple() for message in self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)


def has_history(memory: ConversationMemory | None) -> bool:
    return memory is not None and len(memory) > 0


def needs_rewrite(question: str, memory: ConversationMemory | None, cfg: AppConfig) -> bool:
    """True when the question cannot be retrieved as typed.

    Two deterministic signals, both checked before any LLM call: a deictic trigger from
    `memory.deictic_triggers` ("its", "that fund", "the same"), or a question shorter than
    `memory.min_rewrite_words` words, which is elliptical by construction ("And NAV?").
    """
    if not cfg.memory.rewrite or not has_history(memory):
        return False
    words = normalise_question(question).split()
    if len(words) < cfg.memory.min_rewrite_words:
        return True
    return any(_trigger_pattern(word).search(question) is not None for word in cfg.memory.deictic_triggers)


def resolve_question(
    question: str,
    memory: ConversationMemory | None,
    cfg: AppConfig,
    *,
    classifier: Callable[[str, AppConfig], Decision] = classify,
    rewriter: Callable[[str, ConversationMemory, AppConfig], str] | None = None,
) -> ResolvedQuestion:
    """Classify the raw question, then rewrite it for retrieval when it is a follow-up.

    Order matters: `classifier` runs on the raw text, so a refusal is never reached through a model
    rewrite, and the rewritten question is re-classified so a resolution that drags in a foreign
    scheme name is still refused. The two callables are parameters so `app.pipeline` can inject its
    own seams in tests instead of reaching past them to the real guard and the real Groq client.
    """
    decision = classifier(question, cfg)
    if decision.intent != "FACTUAL" or not needs_rewrite(question, memory, cfg):
        return ResolvedQuestion(original=question, question=question, decision=decision)
    assert memory is not None
    resolve = rewriter if rewriter is not None else rewrite_question
    rewritten = resolve(question, memory, cfg)
    if rewritten == question:
        return ResolvedQuestion(original=question, question=question, decision=decision)
    return ResolvedQuestion(
        original=question,
        question=rewritten,
        decision=classifier(rewritten, cfg),
        rewritten=True,
    )


def rewrite_question(
    question: str,
    memory: ConversationMemory | None,
    cfg: AppConfig,
) -> str:
    """The standalone form of `question`, or `question` itself if it cannot be produced.

    The model is asked for one line and is believed only as far as a retrieval query can be: a
    reply is stripped of labels and quotes, truncated, and rejected when it is empty, when it
    repeats the conversation, or when it is longer than `memory.max_rewrite_chars`.
    """
    if not has_history(memory):
        return question
    assert memory is not None
    prompt = build_rewrite_prompt(question, memory.turns())
    try:
        from app.generator import _client

        client = _client()
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=cfg.memory.max_rewrite_tokens,
        )
        raw = response.choices[0].message.content or ""
    except Exception as error:  # noqa: BLE001 - memory must never break the answer path
        _LOG.info("rewrite unavailable reason_code=%s", type(error).__name__)
        return question

    candidate = _clean_rewrite(raw, cfg)
    if not candidate or candidate.lower() in {question.lower()}:
        return question
    _LOG.info("rewritten follow-up original_words=%d rewritten_words=%d", len(question.split()), len(candidate.split()))
    return candidate


def _clean_rewrite(raw: str, cfg: AppConfig) -> str:
    for line in raw.splitlines():
        text = _LEADING_LABELS.sub("", line.strip()).strip().strip("\"'").strip()
        if not text:
            continue
        if len(text) > cfg.memory.max_rewrite_chars:
            text = text[: cfg.memory.max_rewrite_chars].rsplit(" ", 1)[0].strip()
        return text
    return ""


def _trigger_pattern(word: str) -> re.Pattern[str]:
    compiled = _TRIGGER_PATTERNS.get(word)
    if compiled is None:
        compiled = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        _TRIGGER_PATTERNS[word] = compiled
    return compiled
