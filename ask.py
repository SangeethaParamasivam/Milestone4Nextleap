r"""Interactive CLI: guard -> retrieve -> Groq answer, with the retrieved chunks shown.

    .\.venv\Scripts\python.exe ask.py
    .\.venv\Scripts\python.exe ask.py "What is the expense ratio of HDFC Large Cap Fund?"
    .\.venv\Scripts\python.exe ask.py --show-chunks --no-cache "What is the exit load?"

The question is embedded with the same local model that built the corpus and the chunks come
from the persistent Chroma collection. Groq is used only to turn those chunks into a short
grounded answer, and every answer passes the two post-LLM gates in `app/validators.py`.

The session remembers the last `memory.max_messages` messages (`app/memory.py`), so a follow-up
is retrieved as a standalone question: after "What is the expense ratio of HDFC Large Cap Fund?",
"what about its lock-in?" is rewritten before retrieval. The guard always sees the raw question,
and `--no-memory` turns the rewriting off.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from app.guard import refusal_answer
from app.generator import MissingApiKeyError, answer
from app.memory import ConversationMemory, resolve_question
from app.models import Answer, RetrievalResult
from app.retriever import retrieve
from config import AppConfig, ConfigError, get_groq_model, load_config

BANNER = "Mutual fund FAQ assistant. Type 'exit' to quit."
EXIT_WORDS = {"exit", "quit", ":q", "bye"}
HISTORY_WORDS = {"history", "context"}
RESET_WORDS = {"reset", "clear"}
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "filelock")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask.py", description="Ask a grounded question about the corpus.")
    parser.add_argument("question", nargs="*", help="ask one question and exit instead of looping")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved chunks before the answer")
    parser.add_argument("--show-rewrite", action="store_true", help="print the standalone question a follow-up was rewritten to")
    parser.add_argument("--no-memory", action="store_true", help="forget the conversation and retrieve every question as typed")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the answer cache")
    parser.add_argument("--quiet", action="store_true", help="hide the per-call generator log line")
    return parser


def render_chunks(result: RetrievalResult, cfg: AppConfig) -> str:
    """The chunks actually sent to the model, so the answer can be checked against its evidence."""
    if not result.hits:
        return "[chunks] none retrieved"
    lines = [
        f"[chunks] {len(result.hits)} hit(s) | topic={result.filtered_topic or 'none'} "
        f"| candidates={result.total_candidates}"
    ]
    for index, hit in enumerate(result.hits, start=1):
        chunk = hit.chunk
        as_of = chunk.content_as_of or "unknown"
        lines.append(
            f"  [{index}] score={hit.score:.4f} scheme={chunk.scheme_name} topic={chunk.topic} "
            f"as_of={as_of}"
        )
        lines.append(f"      {chunk.chunk_id}")
        lines.append(f"      {chunk.text}")
    return "\n".join(lines)


def ask_once(
    question: str,
    cfg: AppConfig,
    show_chunks: bool,
    use_cache: bool = True,
    memory: ConversationMemory | None = None,
    show_rewrite: bool = False,
) -> int:
    """Run the full pipeline for one question and print the result. Returns a process exit code.

    `memory` is optional: without it every question is retrieved exactly as typed, which is the
    behaviour of a one-shot invocation. Only answered exchanges are remembered - a refusal or an
    out-of-corpus message states no subject for a later pronoun to resolve, so it is not stored.
    """
    resolved = resolve_question(question, memory, cfg)
    decision = resolved.decision
    if decision.intent != "FACTUAL":
        print(refusal_answer(decision).render())
        return 0
    if show_rewrite and resolved.rewritten:
        print(f"[rewrite] {resolved.original} -> {resolved.question}")

    result = retrieve(resolved.question, decision, cfg)
    if show_chunks:
        print(render_chunks(result, cfg))
    if result.status == "out_of_corpus" or not result.hits:
        print(
            Answer(
                text="I don't have that in my sources.",
                source_url="",
                doc_type="",
                publisher="",
                content_as_of="unknown",
                used_chunk_ids=(),
                status="out_of_corpus",
                was_cached=False,
            ).render()
        )
        return 0

    try:
        response = answer(resolved.question, decision, result, cfg, use_cache=use_cache)
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print()
    print(response.render())
    if show_chunks:
        print(f"\n[answer] status={response.status} cached={int(response.was_cached)} "
              f"chunks_used={len(response.used_chunk_ids)}")
    if memory is not None:
        memory.add_exchange(question, response.text)
    return 0


def render_memory(memory: ConversationMemory) -> str:
    """The kept window, so the user can see what a follow-up would be resolved against."""
    if not len(memory):
        return "[memory] empty"
    lines = [f"[memory] {len(memory)}/{memory.max_messages} messages"]
    for index, message in enumerate(memory.messages, start=1):
        lines.append(f"  [{index}] {message.role}: {message.text}")
    return "\n".join(lines)


def configure_output() -> None:
    """Force UTF-8 on stdout and silence third-party log noise.

    Chunk text contains `₹` and en dashes, which the default Windows console codec cannot encode,
    and the HuggingFace and HTTP libraries log every request at INFO.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_output()
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    use_cache = not args.no_cache

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.question:
        return ask_once(" ".join(args.question).strip(), cfg, args.show_chunks, use_cache)

    memory = None if args.no_memory else ConversationMemory(cfg.memory.max_messages)
    print(BANNER)
    print(f"model: {get_groq_model()}")
    if memory is not None:
        print(f"memory: last {memory.max_messages} messages | 'history' shows them, 'reset' forgets them")
    while True:
        try:
            raw = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw.lower() in EXIT_WORDS:
            return 0
        if memory is not None and raw.lower() in HISTORY_WORDS:
            print(render_memory(memory))
            continue
        if memory is not None and raw.lower() in RESET_WORDS:
            memory.clear()
            print("[memory] cleared")
            continue
        code = ask_once(raw, cfg, args.show_chunks, use_cache, memory, args.show_rewrite)
        if code != 0:
            return code


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
