from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from app.models import Chunk, CleanDocument, SourceSpec
from config import AppConfig, ConfigError, load_config
from ingest.chunker import plan_chunks, sources_csv_path, write_chunks, write_sources_csv
from ingest.cleaner import clean_file, is_rejected
from ingest.embedder import embed_documents, real_token_count, write_embeddings_preview
from ingest.loader import cached_path, fetch_all, meta_path, render_report
from ingest.vectorstore import corpus_hash, get_collection, prune, read_status, upsert_chunks, write_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_ingest.py", description="Ingestion pipeline for the MF FAQ corpus.")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--force", action="store_true", help="ignore the raw cache and refetch")
    parser.add_argument("--sources", default=None, help="comma-separated source ids, e.g. s1,s2")
    parser.add_argument("--dry-run", action="store_true", help="stop after the chunk artefacts")
    return parser


def embed_text(chunk: Chunk) -> str:
    """The string embedded for a chunk: heading, owning scheme name, then body.

    Two blind spots are closed here without touching the stored/labeled text:

    * the heading is embedded so a section's subject is visible to the encoder (the
      "Investment Objective" body never contains the word "objective");
    * the scheme name is embedded because question sentences are fund-specific — a pure fact
      chunk that never repeats the product name cannot match "HDFC Equity Flexi Cap" queries
      against summary chunk, which lists the name in its body.
    """
    return f"{chunk.section_heading}\n{chunk.scheme_name}\n{chunk.text}" if chunk.section_heading else chunk.text


def select_sources(cfg: AppConfig, requested: str | None) -> tuple[SourceSpec, ...]:
    if not requested:
        return cfg.sources
    wanted = [token.strip() for token in requested.split(",") if token.strip()]
    known = {spec.id for spec in cfg.sources}
    unknown = [token for token in wanted if token not in known]
    if unknown:
        raise ConfigError(f"unknown source ids: {', '.join(unknown)}")
    return tuple(cfg.source_by_id(token) for token in wanted)


def run(
    config_path: str | Path | None = None,
    *,
    force: bool = False,
    sources: str | None = None,
    dry_run: bool = False,
) -> int:
    """Phases 2-4 pipeline: load config, fetch the allowlist, clean every fetched page, chunk the
    cleaned pages, then embed and upsert them into the persistent Chroma collection and write
    `data/status.json`.

    A page below `min_page_chars` is logged to `data/rejected.jsonl` and the run exits 1, but every
    other source is still fetched, cleaned and chunked first. `--dry-run` stops after the chunk
    artefacts; a `--sources` subset skips the status gate and the prune.
    """
    cfg = load_config(config_path)
    specs = select_sources(cfg, sources)
    subset = sources is not None
    results = fetch_all(specs, cfg, force=force)
    print(render_report(results))
    by_id = {result.source_id: result for result in results}
    rejected: list[str] = []
    documents: list[tuple[SourceSpec, CleanDocument]] = []
    for spec in specs:
        result = by_id[spec.id]
        if result.status != "ok":
            continue
        doc = clean_file(cached_path(spec, cfg), spec, cfg)
        if is_rejected(doc, cfg):
            rejected.append(f"{spec.id}({doc.extracted_chars})")
            continue
        documents.append((spec, doc))
    if not results:
        print("[load] no sources requested, nothing fetched")
        return 1
    if not all(result.status == "ok" for result in results):
        failed = [result for result in results if result.status != "ok"]
        print(
            f"[load] {len(failed)} of {len(results)} sources did not reach status ok: "
            + ", ".join(f"{result.source_id}={result.status}" for result in failed)
        )
        return 1
    if rejected:
        print(
            f"[clean] {len(rejected)} of {len(results)} pages below min_page_chars="
            f"{cfg.chunking.min_page_chars}: " + ", ".join(rejected)
        )
        return 1
    if not documents:
        print("[clean] no usable documents to chunk")
        return 1
    chunks = _write_chunk_artefacts(documents, cfg, config_path)
    if dry_run:
        print("[embed] dry-run, stopped after chunk artefacts")
        return 0
    return _embed_and_store(chunks, cfg, specs=specs, subset=subset, force=force)


def _write_chunk_artefacts(
    documents: list[tuple[SourceSpec, CleanDocument]], cfg: AppConfig, config_path: str | Path | None
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    dropped = 0
    for spec, doc in documents:
        plan = plan_chunks(doc, spec, cfg)
        chunks.extend(plan.chunks)
        dropped += plan.dropped
        print(
            f"[chunk] {spec.id} chunks={len(plan.chunks)} dropped={plan.dropped} "
            f"size={cfg.chunking.size} overlap={cfg.chunking.overlap}"
        )
    write_chunks(chunks, cfg, dropped=dropped)
    write_sources_csv(chunks, sources_csv_path(config_path), cfg=cfg)
    print(f"[chunk] total={len(chunks)} dropped={dropped} -> chunks.jsonl, chunks_preview.txt, sources.csv")
    return tuple(chunks)


def _embed_and_store(
    chunks: tuple[Chunk, ...], cfg: AppConfig, *, specs: tuple[SourceSpec, ...], subset: bool, force: bool
) -> int:
    source_hashes = _source_hashes(specs, cfg)
    digest = corpus_hash(source_hashes)
    status = read_status(cfg)
    if not force and not subset and status is not None and status.corpus_hash == digest:
        print(f"corpus unchanged, skipping embed+store (corpus_hash={digest[:8]}, chunk_count={status.chunk_count})")
        return 0
    offending = [
        chunk.chunk_id
        for chunk in chunks
        if real_token_count(chunk.text, cfg) > cfg.chunking.max_tokens
    ]
    if offending:
        print(
            f"[embed] {len(offending)} chunks exceed max_tokens={cfg.chunking.max_tokens} with the real "
            f"tokenizer: " + ", ".join(offending[:5]) + ("..." if len(offending) > 5 else "")
        )
        return 1
    vectors = embed_documents([embed_text(chunk) for chunk in chunks], cfg)
    written = upsert_chunks(chunks, vectors, cfg)
    preview = write_embeddings_preview(vectors, [embed_text(chunk) for chunk in chunks], cfg)
    removed = 0 if subset else prune(chunks, cfg)
    stored = get_collection(cfg, create=False).count()
    new_status = write_status(cfg, chunks, source_hashes)
    print(
        f"[embed] vectors={vectors.shape[0]}x{vectors.shape[1]} model={cfg.embedding.model} "
        f"real_token_max={max(real_token_count(chunk.text, cfg) for chunk in chunks)}"
    )
    print(f"[embed] upserted={written} pruned={removed} stored={stored} -> {cfg.paths.persist_dir}")
    print(f"[embed] preview -> {preview.name}")
    print(f"[store] corpus_hash={new_status.corpus_hash[:8]} chunk_count={new_status.chunk_count}")
    return 0


def _source_hashes(specs: tuple[SourceSpec, ...], cfg: AppConfig) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for spec in specs:
        record = meta_path(spec, cfg)
        if record.is_file():
            hashes[spec.id] = str(json.loads(record.read_text(encoding="utf-8")).get("sha256", ""))
    return hashes


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        return run(args.config, force=args.force, sources=args.sources, dry_run=args.dry_run)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
