"""Deterministic offline retrieval (architecture §6.2, implementation §8).

Order of operations: resolve scheme and topic, dense pass, lexical pass, RRF fusion,
scheme affinity, dedupe, `min_score` floor, scope check, then filter relaxation (A8). No LLM
import and no network: the question is embedded with the same local model that built the
corpus (`ingest.embedder.embed_query`) and the vectors come from the persistent Chroma
collection.

**Relevance threshold (`min_score`) — a documented deviation from §8.3 step 9.** §8.3 applies
the floor to the RRF *fused* score, but RRF is a rank score with no absolute scale, so it
cannot express "relevant enough". Measured here, every golden question's top hit lands in
0.0245-0.0263 regardless of relevance and the out-of-corpus Q9 scores *above* several valid
answers. A floor on that value either rejects everything or accepts everything.

The floor is therefore applied to the **dense cosine score** (`1 - distance`), which is the
absolute relevance signal, inside each pass. The fused score then only decides *ranking*, and
`hit.score` carries it for display and ordering.

**Scope check.** Cosine alone is not a scope test either: off-topic questions such as "What is
the price of Bitcoin today?" reach 0.377 and "Who is the president of India?" 0.352, above
valid in-corpus questions as low as 0.358, because MiniLM maps every finance-shaped question
near the HDFC pages. So a question that resolves no in-scope scheme must additionally show
lexical evidence against a retrieved chunk, or name a scheme from `out_of_corpus`. This is the
"resolved_scheme is None and lexical_ranks empty, combined with a low best score" case that
§8.4 requires, expressed as a conjunctive check.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping, Sequence

import numpy as np

from app.guard import normalise_question
from app.models import Chunk, Decision, RetrievalHit, RetrievalResult
from config import AppConfig
from ingest.embedder import embed_query
from ingest.topics import phrase_pattern
from ingest.vectorstore import CorpusMissingError, query_chunks

_LOG = logging.getLogger("app.retriever")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPLIST = frozenset(
    {
        "the", "and", "for", "what", "is", "of", "my", "in", "a", "to", "how", "do", "i",
        "fund", "funds", "me", "are", "was", "were", "be", "on", "it", "its", "that", "this",
    }
)
_GENERAL_TOPIC = "general"
_RELAXED = "relaxed"
_QUESTION_WORDS = frozenset(
    {"who", "which", "whose", "whom", "when", "where", "why", "tell", "give", "list", "please", "does"}
)
_GENERIC_WORDS = frozenset({"india", "indian", "hdfc", "fund", "funds", "plan", "scheme", "invest"})
_PLAN_OPTION_WORDS = frozenset({"direct", "regular", "growth", "dividend", "plan", "option", "fund"})
_PARENTHESISED = re.compile(r"\(([^)]+)\)")


def resolve_scheme(question: str, cfg: AppConfig) -> str | None:
    """The `scheme_name` of the longest alias occurring in the question, else `None`."""
    haystack = normalise_question(question)
    best: tuple[int, str] | None = None
    for spec in cfg.sources:
        for alias in scheme_aliases(spec.scheme_name):
            length = len(alias.split())
            if phrase_pattern(alias).search(haystack) is None:
                continue
            if best is None or length > best[0]:
                best = (length, spec.scheme_name)
    return best[1] if best is not None else None


def scheme_aliases(scheme_name: str | None) -> tuple[str, ...]:
    """Full name plus shortened forms: without plan/option words, without a trailing "fund", and any parenthesised qualifier."""
    if not scheme_name:
        return ()
    full = normalise_question(scheme_name)
    if not full:
        return ()
    aliases = [full]
    stripped = " ".join(token for token in full.split() if token not in _PLAN_OPTION_WORDS)
    without_fund = " ".join(
        token for token in stripped.split() if token not in _PLAN_OPTION_WORDS - {"fund"}
    )
    trimmed = without_fund[:-5] if without_fund.endswith(" fund") else without_fund
    for candidate in (stripped, trimmed):
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    for qualifier in _PARENTHESISED.findall(scheme_name):
        normalised = normalise_question(qualifier)
        if normalised and normalised not in aliases:
            aliases.append(normalised)
        prefixed = normalise_question(f"hdfc {qualifier}")
        if prefixed and prefixed not in aliases:
            aliases.append(prefixed)
    return tuple(aliases)


def resolve_topic(question: str, cfg: AppConfig) -> str | None:
    """The first configured topic whose phrase occurs in the question, else `None`."""
    haystack = normalise_question(question)
    for topic, phrases in cfg.topics.items():
        for phrase in phrases:
            if phrase_pattern(phrase).search(haystack) is not None:
                return topic
    return None


def build_where(
    topic: str | None,
    scheme: str | None,
    related_topics: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object] | None:
    """Chroma where clause for the first pass: the topic, its related topics, `general` and the scheme.

    A topic filter also admits the topic itself, any `retrieval.related_topics` bundled with it and
    the `general` fallback. The bundling exists because Groww's "Investment Objective" section
    yields a single chunk that carries both the objective sentence and the "Fund benchmark ..."
    line but only one `topic` label; without it an `objective` question is stranded from the very
    chunk that would answer it, since the filter is never relaxed while it returns `top_k` hits.
    """
    if topic is None and scheme is None:
        return None
    clauses: list[dict[str, object]] = []
    if topic is not None:
        matched = {topic, _GENERAL_TOPIC}
        matched.update((related_topics or {}).get(topic, ()))
        clauses.append({"$or": [{"topic": name} for name in sorted(matched)]})
    if scheme is not None:
        clauses.append({"scheme_name": scheme})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def fuse(
    dense: Sequence[RetrievalHit],
    lexical_ranks: Mapping[str, int],
    cfg: AppConfig,
) -> list[RetrievalHit]:
    """Reciprocal-rank fusion. A dense hit with no lexical rank keeps only its dense term."""
    rrf_k = cfg.retrieval.rrf_k
    fused: list[RetrievalHit] = []
    for hit in dense:
        lexical_rank = lexical_ranks.get(hit.chunk.chunk_id)
        score = 1.0 / (rrf_k + hit.dense_rank)
        if lexical_rank is not None:
            score += cfg.retrieval.lexical_weight / (rrf_k + lexical_rank)
        fused.append(
            RetrievalHit(
                chunk=hit.chunk,
                score=score,
                dense_rank=hit.dense_rank,
                lexical_rank=lexical_rank,
            )
        )
    return fused


def retrieve(question: str, decision: Decision, cfg: AppConfig) -> RetrievalResult:
    """Embed the question with the corpus model, fuse the passes and apply the `min_score` floor."""
    scheme = resolve_scheme(question, cfg)
    topic = resolve_topic(question, cfg)
    n_results = max(cfg.retrieval.top_k * cfg.retrieval.candidate_multiplier, 20)
    vector = embed_query(question, cfg)

    try:
        filtered = _pass(question, vector, build_where(topic, scheme, cfg.related_topics), scheme, cfg, n_results)
        total = filtered.total_candidates
        hits = filtered.hits
        used_topic = topic or ""
        if len(hits) < cfg.retrieval.top_k:
            relaxed = _pass(question, vector, None, scheme, cfg, n_results)
            if len(relaxed.hits) > len(hits):
                hits = relaxed.hits
                total = max(total, relaxed.total_candidates)
                used_topic = _RELAXED
    except CorpusMissingError:
        _LOG.info("intent=%s status=out_of_corpus reason=corpus_missing", decision.intent)
        return RetrievalResult(hits=(), status="out_of_corpus", filtered_topic="", total_candidates=0)

    hits = _apply_scheme_affinity(hits, scheme, cfg)
    hits = _dedupe(hits)
    hits = hits[: cfg.retrieval.top_k]
    if not hits or (scheme is None and not _is_on_topic(question, hits, cfg)):
        status = "out_of_corpus"
        hits = ()
    else:
        status = "ok"
    _LOG.info(
        "intent=%s status=%s scheme=%s topic=%s filtered_topic=%s n_hits=%d candidates=%d",
        decision.intent, status, scheme or "", topic or "", used_topic, len(hits), total,
    )
    return RetrievalResult(
        hits=tuple(hits),
        status=status,
        filtered_topic=used_topic,
        total_candidates=total,
    )


def _pass(
    question: str,
    vector: np.ndarray,
    where: Mapping[str, object] | None,
    scheme: str | None,
    cfg: AppConfig,
    n_results: int,
) -> RetrievalResult:
    dense = [hit for hit in _dense_hits(vector, n_results, cfg, where) if hit.score >= cfg.retrieval.min_score]
    tokens = _query_tokens(question, scheme, cfg)
    lexical_ranks = _lexical_ranks(dense, tokens)
    return RetrievalResult(
        hits=tuple(fuse(dense, lexical_ranks, cfg)),
        status="ok",
        filtered_topic="",
        total_candidates=len(dense),
    )


def _dense_hits(
    vector: np.ndarray,
    n_results: int,
    cfg: AppConfig,
    where: Mapping[str, object] | None,
) -> list[RetrievalHit]:
    result = query_chunks(vector, n_results, cfg, where)
    return [
        RetrievalHit(
            chunk=Chunk.from_metadata(md, text),
            score=score,
            dense_rank=index,
            lexical_rank=None,
        )
        for index, (md, text, score) in enumerate(
            zip(result.metadatas, result.documents, result.scores), start=1
        )
    ]


def _query_tokens(question: str, scheme: str | None, cfg: AppConfig) -> frozenset[str]:
    """Rare tokens of the question, plus the alias tokens of the resolved scheme only.

    Alias tokens are filtered by the same stoplist: scheme names contribute words like "fund",
    "direct" and "plan" that appear across every page, and leaving them in lets an unrelated
    chunk earn a lexical rank that competes with the genuinely distinctive match.
    """
    tokens = {token for token in _TOKEN.findall(normalise_question(question)) if len(token) >= 3}
    alias_tokens = (
        {
            token
            for spec in cfg.sources
            if spec.scheme_name == scheme
            for alias in scheme_aliases(spec.scheme_name)
            for token in alias.split()
        }
        if scheme is not None
        else set()
    )
    return frozenset((tokens | alias_tokens) - _STOPLIST)


def _lexical_ranks(dense: Sequence[RetrievalHit], tokens: frozenset[str]) -> dict[str, int]:
    scored = [
        (hit, len(tokens & set(_TOKEN.findall(hit.chunk.normalised_text))))
        for hit in dense
    ]
    ranked = sorted(
        (pair for pair in scored if pair[1] > 0),
        key=lambda pair: (-pair[1], pair[0].dense_rank),
    )
    return {hit.chunk.chunk_id: rank for rank, (hit, _) in enumerate(ranked, start=1)}


def _apply_scheme_affinity(
    hits: Sequence[RetrievalHit], scheme: str | None, cfg: AppConfig
) -> list[RetrievalHit]:
    """Drop chunks from a conflicting scheme unless that would empty the set (architecture §6.2)."""
    if scheme is None or not cfg.retrieval.scheme_affinity:
        return list(hits)
    kept = [hit for hit in hits if hit.chunk.scheme_name in (scheme, "")]
    if kept:
        return kept
    _LOG.info("scheme=%s affinity_relaxed=1 n_hits=%d", scheme, len(hits))
    return list(hits)


def _dedupe(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """Collapse identical normalised text, keeping the newest `content_as_of` then the best score."""
    grouped: dict[str, list[RetrievalHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.chunk.content_hash, []).append(hit)
    winners = [max(group, key=_dedupe_key) for group in grouped.values()]
    return sorted(winners, key=lambda hit: (-hit.score, hit.chunk.chunk_id))


def _is_on_topic(question: str, hits: Sequence[RetrievalHit], cfg: AppConfig) -> bool:
    """For a question naming no in-scope scheme, demand lexical evidence or a foreign-scheme match.

    Cosine alone is not a scope test: measured best-cosine for out-of-topic questions reaches
    0.38, above several valid in-corpus questions, because MiniLM places every finance-shaped
    question near the HDFC pages. §8.4 therefore requires the extra signal that a question with
    no resolved scheme must either overlap the corpus lexically or name a known foreign scheme.
    """
    if _names_foreign_scheme(question, cfg):
        return False
    tokens = _query_tokens(question, None, cfg) - _QUESTION_WORDS
    if not tokens:
        return False
    return any(
        _content_match(tokens, hit) for hit in hits
    )


def _content_match(tokens: frozenset[str], hit: RetrievalHit) -> bool:
    """A token counts only when it is not a generic word that appears in most of the corpus."""
    words = set(_TOKEN.findall(hit.chunk.normalised_text))
    return bool((tokens & words) - _GENERIC_WORDS)


def _names_foreign_scheme(question: str, cfg: AppConfig) -> bool:
    """True when the question names a scheme outside the corpus and names no in-scope scheme."""
    haystack = normalise_question(question)
    return any(phrase_pattern(name).search(haystack) is not None for name in sorted(cfg.out_of_corpus))


def _dedupe_key(hit: RetrievalHit) -> tuple[str, float]:
    return (hit.chunk.content_as_of, hit.score)
