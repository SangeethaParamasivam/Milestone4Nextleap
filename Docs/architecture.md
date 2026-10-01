# Architecture — Milestone 4: Mutual Fund FAQ RAG Chatbot (Facts-Only Q&A)

| Field | Value |
|---|---|
| Document | `Docs/architecture.md` |
| Version | 0.1 (Draft) |
| Date | 2026-09-30 |
| Status | Ready for review |
| Derived from | `Docs/PRD.md` v0.1, `Docs/ProblemStatement.txt` |
| Audience | Implementation team, reviewer/evaluator |

---

## 1. Purpose & Scope of This Document

The PRD defines **what** the system must do. This document defines **how** it is built:
component boundaries, module contracts, data flow, state on disk, failure handling, and the
mapping from every functional requirement to the component that satisfies it.

Non-goals of this document: product copy, prompt wording (see PRD §6.5), final configuration
values (see `config.yaml`), and operational runbooks beyond what is needed to build M1–M10.

### 1.1 Architectural drivers (ranked)

| # | Driver | Architectural consequence |
|---|---|---|
| D1 | **Grounding is the product** — every number must be traceable to a chunk | Ingestion is content-addressed and auditable; generation never sees raw text, only retrieved chunk text + metadata |
| D2 | **Refusal correctness is a hard gate (G4, 0 false accepts)** | A deterministic rules layer runs *before* the LLM and can short-circuit the pipeline entirely; the LLM is never asked to adjudicate refusal |
| D3 | **The LLM is untrusted** | Two independent post-LLM gates (citation validation, output pattern filter) that can suppress any answer |
| D4 | **Inspectability / auditability** (G3, G6, G7, NFR-6) | Chunk text is a first-class persisted artefact, not an internal detail; every answer carries provenance metadata |
| D5 | **Determinism & reproducibility** (G6, NFR-3) | Content-addressed chunk IDs, idempotent upsert, `status.json` marker, pinned dependency versions, response cache |
| D6 | **Local-first, one network dependency** (NFR-2) | Embedding and storage are fully local; the only outbound calls are ingestion HTTP fetches and the Groq generation call |
| D7 | **No PII, ever** (NFR-5, §8) | Questions are never persisted; PII detection is pre-LLM (abort) and post-LLM (suppress); logging is redacted |
| D8 | **Module isolation per RAG stage** (NFR-8) | One module per pipeline stage with a narrow, typed contract; the UI layer holds no business logic |

---

## 2. System Context

```
                    ┌──────────────────────────────────────────────┐
                    │              User (browser)                 │
                    └───────────────────────┬──────────────────────┘
                                            │ HTTP (localhost)
                    ┌───────────────────────▼──────────────────────┐
                    │   Streamlit UI  —  app/ui.py                │
                    │   welcome · 3 examples · input · disclaimer │
                    │   answer · Source · Retrieved context panel  │
                    └───────────────────────┬──────────────────────┘
                                            │ in-process call
                    ┌───────────────────────▼──────────────────────┐
                    │        Application Service  —  app/          │
                    │  guard → retriever → generator → validation   │
                    └───┬───────────────────────┬──────────────────┘
                        │ local                │ HTTPS
        ┌───────────────▼─────────────┐   ┌─────▼────────────────────┐
        │  Embedder (MiniLM-L6-v2)   │   │  Groq LLM API            │
        │  sentence-transformers      │   │  llama-3.1-8b-instant    │
        │  384-dim, CPU, offline      │   │  (single generation call)│
        └───────────────┬─────────────┘   └──────────────────────────┘
                        │
        ┌───────────────▼─────────────┐
        │  ChromaDB (persistent)      │      Local disk only
        │  collection: mf_faqs        │
        └─────────────────────────────┘

  ── Offline / build-time only ──
  ingest/  →  allowlisted official pages (groww.in, hdfcmf.com, sebi.gov.in,
              amfiindia.com, myamfi.in)  →  data/raw/  →  data/chunks/  →  ChromaDB
  evaluate.py → reports/eval_report.md
```

**Trust boundaries**

| Boundary | Data crossing | Control |
|---|---|---|
| Internet → loader | HTML from allowlisted domains | Domain allowlist (FR-1.4), descriptive UA, rate limit, retries, raw cache |
| LLM → application | Generated answer text | Citation validation (FR-6.3), number/performance/PII filter (FR-6.4) |
| User → application | Free-text question | PII + intent rules pre-LLM (FR-7), no persistence (FR-7.6) |
| Application → disk | Chunks, vectors, status | Content-addressed IDs, atomic writes, `.gitignore` for volatile dirs |

---

## 3. High-Level Design

### 3.1 Two pipelines, one shared model

```
INGESTION (batch, run once — FR-4.4)
  config.yaml allowlist
        │
        ▼
  [Load]    ingest/loader.py      fetch → data/raw/<domain>/<slug>.html   + fetch report
        │
        ▼
  [Clean]   ingest/cleaner.py     DOM → block list (heading/para/list/table) + normalised text
        │                       reject log for under-length pages (FR-2.5)
        ▼
  [Chunk]   ingest/chunker.py     section-aware 900/150 char windows → Block objects
        │                       → data/chunks/chunks.jsonl + chunks_preview.txt
        ▼
  [Embed]   ingest/embedder.py    all-MiniLM-L6-v2, 384-dim, batched, token-length assertion
        │
        ▼
  [Store]   ingest/vectorstore.py ChromaDB upsert (ids=chunk_id) + data/status.json

QUERY (per question — NFR-1 p50 < 8 s)
  question
        │
        ▼
  [Guard]   app/guard.py          PII scan → intent classify (FACTUAL/ADVICE/PERFORMANCE/PII/OOC)
        │                       refusal ⇒ render fixed message + educational link, STOP
        ▼
  [Route]   app/retriever.py      optional metadata filter (doc_type/topic) from classification
        │
        ▼
  [Embed]   ingest/embedder.py    same model, same normaliser
        │
        ▼
  [Retrieve] app/retriever.py     hybrid: dense top-k  + exact rare-token boost + scheme affinity
        │                       min_score floor ⇒ OUT_OF_CORPUS, no LLM call
        ▼
  [Generate] app/generator.py     Groq call, temp≈0, ≤3 sentences, exactly one Source line
        │
        ▼
  [Validate] app/generator.py     citation ∈ retrieved ∩ allowlist → retry ×1
        │                       number/performance/PII filter → suppress on hit
        ▼
  [Present] app/ui.py             answer + Source + doc type + "Last updated from sources:"

OFFLINE  evaluate.py              golden set (10) + refusal set (8) → reports/eval_report.md
```

### 3.2 Layering

| Layer | Modules | May depend on | Must never |
|---|---|---|---|
| Config | `config.py` | — | do I/O |
| Ingest | `ingest/*` | config, models | import `app/*`, call the LLM, import Streamlit |
| App core | `app/guard.py`, `app/retriever.py`, `app/generator.py`, `app/prompts.py` | config, ingest/embedder, ingest/vectorstore | import Streamlit, read raw HTML |
| Presentation | `app/ui.py` | app core, `app/disclaimer.py` | contain retrieval/generation logic (NFR-8) |
| Evaluation | `evaluate.py` | app core only | bypass the guard or the validators |

The dependency graph is strictly acyclic: `config → ingest → app core → ui`, with
`evaluate.py` sitting outside as a peer of `ui`.

---

## 4. Repository Layout

```
Milestone4/
├── app/
│   ├── ui.py                 # Streamlit entry point (FR-8)
│   ├── guard.py              # pre-LLM intent rules + refusal messages (FR-7)
│   ├── retriever.py          # filter, hybrid search, min_score, dedupe (FR-5)
│   ├── generator.py          # Groq call + post-validation + fallback (FR-6)
│   ├── prompts.py            # system/user prompt assembly (FR-6.2, PRD §6.5)
│   ├── disclaimer.py         # single source-of-truth disclaimer (FR-9)
│   ├── models.py             # Chunk, RetrievalHit, Answer, Decision dataclasses
│   └── cache.py              # per-question response cache (FR-6.5)
├── ingest/
│   ├── loader.py             # allowlist fetch + raw cache + report (FR-1)
│   ├── cleaner.py            # block extraction + normalisation (FR-2)
│   ├── chunker.py            # section-aware chunking (FR-3, §6.3)
│   ├── embedder.py           # MiniLM wrapper, shared by both paths (FR-4.1)
│   ├── vectorstore.py        # ChromaDB persistent + status marker (FR-4)
│   └── topics.py             # heading → topic/doc_type classification (controlled vocab)
├── config/
│   ├── config.yaml           # allowlist, refusal keywords, thresholds, chunk params
│   └── sources.csv           # D2 deliverable: URL, publisher, doc type, retrieval date
├── data/
│   ├── raw/                  # cached HTML (git-ignored)
│   ├── chunks/chunks.jsonl   # D6: every chunk, full metadata
│   ├── chunks/chunks_preview.txt
│   ├── golden_questions.jsonl
│   └── chroma/               # PERSIST_DIR (git-ignored)
├── reports/eval_report.md    # D7
├── docs/
│   ├── sources.md, sample_qa.md
│   └── architecture.md, PRD.md
├── tests/                    # pytest: guard, chunker, validators, cleaner
├── run_ingest.py             # M1–M4 CLI
├── evaluate.py               # M9 CLI
├── requirements.txt          # pinned (R10)
├── .env.example              # committed; .env git-ignored (R11)
├── .gitignore
└── README.md                 # D3
```

`.gitignore` contains `.env`, `data/raw/`, `data/chroma/`, `__pycache__/`, `*.log`
(FR-10.2).

---

## 5. Component Design — Ingestion

### 5.1 `ingest/loader.py` — Load

**Responsibility.** Turn the version-controlled allowlist into cached raw HTML plus a
per-URL status report. No parsing beyond a sanity check.

| Contract point | Design |
|---|---|
| Input | `config.yaml → sources` (id, url, domain, doc_type, scheme_name, scheme_category, plan, option) |
| Output | `data/raw/<domain>/<source_id>.html`, `data/raw/<source_id>.meta.json`, list of `FetchResult` |
| Idempotency | Re-use cached file if present and `--force` not passed; record `fetched_at` + sha256 of body |
| Network policy | Only `groww.in, hdfcmf.com, sebi.gov.in, amfiindia.com, myamfi.in`; a page referencing other domains yields non-collectable links (FR-1.4) |
| Politeness | Descriptive UA, ≥2 s inter-request delay, per-host serialisation, 3 retries with exponential backoff + jitter, `Retry-After` honoured |
| Failure | A failed URL is reported and skipped; the run fails overall if **all** sources fail or if any page falls below the minimum extracted length (FR-2.5 fail-loud rule) |
| Status codes | 200 → ok; 3xx → follow to allowlisted host only; 403/429/5xx → `blocked`/`error` with reason, no bypass attempt |

`FetchResult` fields: `source_id, url, status, http_code, bytes, sha256, fetched_at, cache_hit, error`.

### 5.2 `ingest/cleaner.py` — Clean

**Responsibility.** Convert raw HTML into an ordered list of semantic **blocks** plus
normalised plain text, losslessly with respect to facts (FR-2.4).

| Contract point | Design |
|---|---|
| Extraction | `trafilatura` (preferred, main-content aware) with a BeautifulSoup + `lxml` structural fallback so block boundaries survive (FR-2.2) |
| Removed | `script, style, noscript, nav, header, footer, aside, form, iframe`, breadcrumb/cookie/consent/ad containers matched by role/aria/class heuristics, and any element whose text duplicates a nav label |
| Preserved | `h1–h6` (as block heading), `p`, `li` (one block per item), table → header row + row groups, definition lists |
| Normalisation | NFKC unicode, en/em dash + smart quotes → ASCII, NBSP → space, collapse runs of whitespace, strip zero-width chars; **numerals, units, `%`, `₹`, dates and "as on <date>" strings are protected** (FR-2.3) |
| Reject log | `data/rejected.jsonl`: `source_id, extracted_chars, reason`. Page under `MIN_PAGE_CHARS` → run fails loudly (FR-2.5) |
| Output | `CleanDocument(source_id, url, page_title, blocks[Block(kind, heading_path, text, row_group)], extracted_chars)` |

`heading_path` is the ancestor heading chain (e.g. `Fees → Exit load`) and is what makes
section-aware chunking and topic classification possible later.

### 5.3 `ingest/chunker.py` — Chunk

**Strategy (PRD §6.3), unchanged:** *section-aware, heading-anchored, token-bounded
windows with sentence-boundary snapping.*

| Parameter | Value | Rationale |
|---|---|---|
| Primary boundary | heading / labelled block (`heading_path`) | keeps a fact with its label |
| `chunk_size` | 900 chars (~200–230 tokens) | under MiniLM's 256-token limit → no silent truncation |
| `chunk_overlap` | 150 chars (~1 sentence, ~17%) | keeps the straddling sentence; avoids duplicate-vector dominance |
| Secondary split | fixed 900/150 window snapped to `.` `?` `;` `:` | never mid-number or mid-table |
| Protected spans | numerals+units, table rows, list items, "as on <date>" | a truncated `1.5%` corrupts the fact |
| Min chunk | 40 chars (dropped + logged) | avoids navigation-noise vectors |
| Tables | one row-group per chunk, header row repeated | a row without headers is meaningless |
| Max chunks/page | 400 (guard) | detects a whole-page cleaning failure |

**Fallback ordering:** heading block → sentence-aligned window → skip. A chunk is never
emitted without `source_url` and `doc_type`.

**Contract.** `chunk(clean_doc, params) -> list[Chunk]`

*Chunk record* — the metadata schema of PRD §6.4 is authoritative:

```json
{
  "chunk_id": "hdfc_large_cap_direct_growth::expense_ratio::a1b2c3d4",
  "source_id": "s1",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "publisher": "Groww (mirroring HDFC MF official data)",
  "source_domain": "groww.in",
  "page_title": "HDFC Large Cap Fund – Direct Growth",
  "doc_type": "scheme_page",
  "scheme_name": "HDFC Large Cap Fund – Direct Growth",
  "scheme_category": "large_cap",
  "plan": "direct",
  "option": "growth",
  "topic": "expense_ratio",
  "section_heading": "Expense ratio",
  "content_as_of": "2026-09-30",
  "chunk_index": 12,
  "char_len": 884,
  "token_count": 214,
  "text": "…verbatim cleaned text…",
  "content_hash": "sha256:…",
  "ingested_at": "2026-09-30T10:14:02Z"
}
```

`source_id` is an addition to the PRD schema, needed to re-derive `docs/sources.csv` (D2)
without re-fetching.

**ID derivation.** `chunk_id = f"{source_id}::{topic}::{sha256(normalised_text)[:8]}"`
— deterministic and content-addressed, so re-ingestion upserts instead of appending (FR-3.4,
FR-4.3). Normalisation before hashing: lowercase, collapse whitespace, strip punctuation
except `.`/`%`.

**Topic classification** (`ingest/topics.py`). Ordered rule list, first match wins, else
`general`. Rules are regex over `heading_path` + block text and live in `config.yaml`
(FR-10.3) so they are inspectable without reading code:

`exit load` → `exit_load`; `expense ratio|TER|charges` → `expense_ratio`; `minimum (sip|investment|lump)`
→ `min_sip`/`min_lump_sum`; `lock.?in|80C|tax` → `lockin_80c`; `riskometer` → `riskometer`;
`benchmark|index` → `benchmark`; `statement|tax document|how to download|capital gains` →
`statements_download`; `nav` → `nav`; `direct vs regular|plan` → `plan_direct_vs_regular`;
`objective|about the scheme|scheme profile` → `objective`.

**Artefacts (FR-3.3).** `chunks.jsonl` (one JSON object per line, full metadata) and
`chunks_preview.txt` (human-readable: `[n] source_url | doc_type | topic | section | char_len`
followed by the verbatim text and a rule).

**Assertions at ingest.** `token_count ≤ 250` for every chunk (R7); every chunk has
`source_url` + `doc_type`; `chunk_id` unique; chunk count ≤ 400/page.

### 5.4 `ingest/embedder.py` — Embed

Single module, single model instance, used by **both** paths (FR-4.1, NFR-2).

| Aspect | Design |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2`, pinned revision in `requirements.txt`, 384-dim, `normalize_embeddings=True` |
| Locality | Model cached under `data/models/`; load is offline; no telemetry |
| Doc API | `embed_documents(texts: list[str], batch_size=32) -> np.ndarray[384]` |
| Query API | `embed_query(text: str) -> np.ndarray[384]` |
| Token guard | `count_tokens` helper; ingest asserts ≤ 250; query side truncates defensively |
| Startup | Lazy singleton loaded once per process; Streamlit caches it via `@st.cache_resource` |

### 5.5 `ingest/vectorstore.py` — Store

| Aspect | Design |
|---|---|
| Client | `chromadb.PersistentClient(path=PERSIST_DIR)` |
| Collection | fixed name `mf_faqs`, `metadata={"hnsw:space": "cosine"}` (FR-4.2) |
| Write | `collection.upsert(ids, embeddings, documents, metadatas)` — never `add` |
| Idempotency | IDs are `chunk_id`, so re-ingest converges to the same rows (FR-4.3) |
| Staleness | `data/status.json` = `{corpus_hash, chunk_count, model, collection, last_ingest_at, source_hashes}`; app skips ingest unless `--force` or `corpus_hash` changed (FR-4.4) |
| Pruning | On ingest, chunk IDs not present in the current corpus are deleted, so a source edit cannot leave orphans |
| Query | `collection.query(query_embeddings=[q], n_results, where=...)`; Chroma distance is cosine distance, so `score = 1 - distance` |

Only scalar/str metadata is stored in Chroma (Chroma rejects nested values); the full record
including `text` is read from `chunks.jsonl` when a hit needs to be rendered. The `text` is
also stored as the Chroma `document` so retrieval alone can answer a "show the chunk" request.

### 5.6 `run_ingest.py` — Orchestration & report

```
load config → for each source: fetch (cache-aware) → clean → reject check
→ chunk → write jsonl + preview → embed → upsert → prune → write status.json
→ print per-URL report: source_id | http status | cache | chars | blocks | chunks | topics
→ exit code 0 (all good) / 1 (any source failed or rejected)
```

Flags: `--force` (ignore cache + status), `--sources s1,s2`, `--dry-run` (fetch+clean+chunk,
no embed/store), `--config path`. This is the FR-11.1 report and the M1–M4 exit gate.

---

## 6. Component Design — Query Path

### 6.1 `app/guard.py` — Guard (deterministic, pre-LLM)

Order of evaluation is significant: **PII first** (so a PII-bearing question is never passed
to later stages and never logged), then advice/performance, then scope.

| Stage | Rule source | Output |
|---|---|---|
| 1. PII scan | regex bank: PAN `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar `\b\d{4}\s?\d{4}\s?\d{4}\b`, 10-digit phone, email, OTP keywords + 4–8 digit codes, "account number", "IFSC", "CVV" | `PII` on any hit; message states identifiers must not be shared and to delete the message; the question body is dropped before any logging (FR-7.5, R8) |
| 2. Advice | intent phrases: *should I, which is best, better than, recommend, suggest, suitable for me, allocate, portfolio, my retirement, good time to (buy|sell|exit|enter), worth buying* | `ADVICE` |
| 3. Performance | *return, returns, % return, CAGR, XIRR, performance, how much did … earn/gain, compare … vs, best performing, ranking, NAV history* | `PERFORMANCE` |
| 4. Out-of-scope entities | known out-of-corpus AMC/scheme names registry (e.g. *Parag Parflex*) with no alias in the allowlist | `OUT_OF_CORPUS` |
| 5. Default | — | `FACTUAL` |

Precedence: `PII > ADVICE > PERFORMANCE > OUT_OF_CORPUS > FACTUAL`. For a question that trips
both advice and performance, the advice refusal is shown.

**Refusal messages** (`app/guard.py` constants, reused by README and `sample_qa.md`):
all contain the facts-only statement and exactly one **educational** link, selected by intent:

| Intent | Message shape | Educational link |
|---|---|---|
| `ADVICE` | "I answer only published facts about these schemes and can't recommend, compare or rate funds or suggest a portfolio." | SEBI mutual-fund basics / investor-education page |
| `PERFORMANCE` | "I don't state or compute returns. Performance figures for each scheme are in its official factsheet, linked below." | Official factsheet of the in-scope scheme if identified, else SEBI investor-education page |
| `PII` | "Please don't share personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). This message has not been stored. Delete it from your chat." | SEBI investor-education page |
| `OUT_OF_CORPUS` | "I don't have that in my sources." + what *is* in scope | none (per PRD §10.1.8 wording) |

**Contract.** `classify(question: str) -> Decision(intent, educational_url, reason_code)`
Rules live in `config.yaml` (`refusal.keywords`, `pii.patterns`, `out_of_corpus.aliases`) —
FR-7.2, FR-10.3. Unit tests assert 8/8 refusal on the PRD §10.4 set and 0 false accepts.

### 6.2 `app/retriever.py` — Retrieve

```
retrieve(question, decision) -> RetrievalResult
```

1. **Filter (FR-5.4).** Map the question to a preferred `topic` via `config.yaml` keyword
   rules. If a confident match exists, first pass uses `where={"$or": [{"topic": t}, {"topic": "general"}]}`
   **plus** a same-scheme constraint when the question names a scheme. If the filtered pass
   returns fewer than `TOP_K` hits, the filter is relaxed to the whole collection — the filter
   narrows, retrieval still ranks (FR-5.4), it never hides the answer.
2. **Dense search (FR-5.2).** `n_results = max(TOP_K * 4, 20)` candidates, cosine distance →
   `score = 1 - distance`.
3. **Hybrid boost (FR-5.5).** RRF-style fusion:
   `fused = 1/(60 + rank_dense) + 0.6 * (1/(60 + rank_lexical))`, where the lexical ranking
   is exact-token matching of rare tokens from the question (scheme-name aliases, `80C`,
   `SIP`, `exit load`, `riskometer`). This guarantees the right scheme's chunks surface when
   several schemes are in the corpus.
4. **Scheme affinity.** If the question resolves to a scheme, chunks whose `scheme_name`
   matches rank above chunks from other schemes; a chunk whose `scheme_name` conflicts with
   the question's scheme is dropped unless nothing else remains (guards the "0 scheme
   confusion" gate, §10.2).
5. **Dedupe (R2).** Collapse chunks with identical normalised text, keeping the one with the
   newest `content_as_of`.
6. **Threshold (FR-5.3).** Take `TOP_K` (default 5). If the best fused score `< MIN_SCORE`,
   or zero hits, return `OUT_OF_CORPUS` → the UI shows "I don't have that in my sources."
   **without calling the LLM.**
7. **Return** `RetrievalResult(hits: list[RetrievalHit], status, filtered_topic, total_candidates)`
   where `RetrievalHit = {chunk (full record), score, dense_rank, lexical_rank}`.

### 6.3 `app/generator.py` — Generate & Validate

```
answer(question, decision, retrieval) -> Answer
```

1. **Cache lookup (FR-6.5).** Key = sha256 of `model | question_normalised | chunk_id_set`.
   In-memory (per process) + optional `data/cache/answers.jsonl` (git-ignored) so evaluation
   re-runs do not re-bill Groq (R9).
2. **Prompt assembly (`app/prompts.py`).** System prompt = PRD §6.5 verbatim, plus runtime
   interpolation of: the retrieved chunks as `[i] source_url | doc_type | content_as_of` blocks
   of **verbatim** text only, the question, and the constraint that `content_as_of` of the cited
   chunk is the only permitted "as of" value. `temperature=0.0`, `max_tokens≈300`,
   `seed` set where supported.
3. **Call.** `groq.Groq(api_key=os.getenv("GROQ_API_KEY"))`. Missing key → clear message and
   exit, never a stack trace (FR-6.1, §10.1.7). Retry once on 429/5xx with backoff (R9).
4. **Citation validation (FR-6.3).** Extract the `Source: <url>` line. The URL must be present
   in **both** the retrieved chunks' metadata and `config.yaml` allowlist. Invalid → retry once
   with the failure reason injected. Still invalid → `SAFE_FALLBACK`.
5. **Output filter (FR-6.4, R5).** Regex scan of the final text:
   * performance/return patterns (`\d+(\.\d+)?\s*%` in a return context, `CAGR`, `XIRR`,
     `has returned`, `p.a.`, `YTD`, `1-year return`, ranking words);
   * PII patterns (the same bank as the guard);
   * any numeric literal that does not appear in the concatenated retrieved text
     (NFR-9, §10.2) — normalised comparison ignoring thousands separators;
   * sentence count > 3 (G5) and missing/incorrect "Last updated from sources:" line.
   A hit → `SAFE_FALLBACK` (the answer is suppressed, not shown). Retries are allowed only for
   the citation failure, so this path terminates in one step.
6. **Return** `Answer(text, source_url, doc_type, publisher, content_as_of, used_chunk_ids,
   status, was_cached)`. `status ∈ {ok, out_of_corpus, refused_advice, refused_performance,
   refused_pii, suppressed}`.

`SAFE_FALLBACK` is a single constant: facts-only apology + the educational link. It is the
terminal state for every failure mode, so the UI never renders a partially-validated answer.

### 6.4 `app/ui.py` — Present

| Element | Behaviour |
|---|---|
| Header | welcome line (FR-8.1) |
| Example questions | exactly 3, fixed, covering expense ratio, ELSS lock-in, capital-gains statement (FR-8.4) |
| Input | single `st.text_input`, no PII-soliciting fields, no login (FR-8.5) |
| Persistent note | "Facts-only. No investment advice." always visible (FR-8.1) |
| Answer block | answer text, exactly one **Source:** link, source document type, publisher, `Last updated from sources: <date>` (FR-8.2) |
| Retrieved context | `st.expander("Retrieved context", expanded=False)` → per chunk: rank, score, URL, section heading, `as of` date, text (FR-8.3) |
| Refusal block | message + educational link, no context panel |
| Footer | `DISCLAIMER` from `app/disclaimer.py` (FR-9.1) |
| Missing key | `st.error("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")` and stop |
| Missing corpus | `st.warning("Run: python run_ingest.py")` and stop |

`app/ui.py` contains no retrieval or generation logic — it calls one orchestrator
(`app/service.py`, or `answer_question()` in `app/pipeline.py`) that runs
guard → retrieve → generate → validate and returns an `Answer`. This keeps NFR-8 satisfiable
and makes the same path testable headlessly by `evaluate.py`.

### 6.5 `app/disclaimer.py`

Single constant `DISCLAIMER` (FR-9.2 wording) plus `SHORT_NOTE = "Facts-only. No investment
advice."`. Imported by `app/ui.py`, and textually mirrored in `README.md` and
`docs/sample_qa.md`. Refusal messages live in `app/guard.py`, not here, to keep one owner per
string.

### 6.6 `app/models.py`

Frozen dataclasses shared by ingest and app, so no layer re-parses another's output:
`Block`, `CleanDocument`, `Chunk`, `FetchResult`, `RetrievalHit`, `Decision`, `Answer`,
`IngestStatus`. `Answer.render()` produces the exact UI string, which is also what
`docs/sample_qa.md` and the eval report consume.

---

## 7. Configuration & Secrets

`config.yaml` is the inspectable control surface (FR-10.3):

```yaml
sources:                     # allowlist; no runtime URL is ever fetched
  - id: s1
    url: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
    domain: groww.in
    doc_type: scheme_page
    scheme_name: HDFC Large Cap Fund – Direct Growth
    scheme_category: large_cap
    plan: direct
    option: growth
allowed_domains: [groww.in, hdfcmf.com, sebi.gov.in, amfiindia.com, myamfi.in]
chunking: {size: 900, overlap: 150, min_chars: 40, max_chunks_per_page: 400}
retrieval: {top_k: 5, min_score: 0.28, candidate_multiplier: 4, lexical_weight: 0.6}
topics: {expense_ratio: ["expense ratio", "ter", ...], ...}
refusal:
  advice: ["should i", "which is best", ...]
  performance: ["return", "cagr", "xirr", ...]
  pii_patterns: ["[A-Z]{5}[0-9]{4}[A-Z]", ...]
  educational_links: {advice: "https://...", performance: "https://...", pii: "https://..."}
out_of_corpus: {known_foreign_schemes: ["parag parflex", ...]}
generation: {temperature: 0.0, max_tokens: 300, seed: 42}
```

`.env` (git-ignored, FR-10.2, R11): `GROQ_API_KEY`, `GROQ_MODEL=llama-3.1-8b-instant`,
`TOP_K=5`, `MIN_SCORE=0.28`, `CHUNK_SIZE=900`, `CHUNK_OVERLAP=150`, `PERSIST_DIR=data/chroma`,
`RAW_DIR=data/raw`. Precedence: real env var > `.env` > `config.yaml` default > code default.
`.env.example` is committed with empty values.

---

## 8. Cross-Cutting Concerns

### 8.1 Security & privacy

| Control | Implementation |
|---|---|
| Secrets | Only `os.getenv`; never logged, never written to `status.json`, `chunks.jsonl`, or the eval report; pre-commit scan for `gsk_` in the repo (R11) |
| PII in | Guard aborts before retrieval; the raw question is never passed to a logger and never persisted (FR-7.6) |
| PII out | Output filter → `SAFE_FALLBACK`; logs store intent + `reason_code`, not text |
| SSRF | `loader` resolves the URL, validates the scheme is `https` and the host is in `allowed_domains`, and rejects redirects to other hosts |
| Corpus integrity | Only allowlisted public pages; third-party links inside a page are recorded as `non-collectable` and never fetched |
| Telemetry | None (NFR-5) |

### 8.2 Observability

Structured per-stage logging to stdout with a `[stage] source_id` prefix: fetch status,
extracted chars, chunk count, top-k scores, decision intent, validation outcome, LLM latency,
token usage. Redaction filter applied before any handler. `evaluate.py` is the reporting
surface for quality (FR-11.2); no metrics server, no APM.

### 8.3 Error handling & degradation

| Failure | Behaviour | User sees |
|---|---|---|
| Source fetch fails at ingest | reported, run exits 1, other sources still written | — |
| Page extracts < `MIN_PAGE_CHARS` | rejected + logged, run exits 1 (fail loudly, never a silent empty corpus) | — |
| Chunk exceeds 250 tokens | assertion fails at ingest, chunk truncated at a sentence boundary + logged | — |
| ChromaDB dir deleted / model absent | `status.json` missing or corpus hash mismatch → app prompts re-ingest | "Run: python run_ingest.py" |
| `GROQ_API_KEY` missing | explicit error at startup | clear setup message |
| Groq 429 / 5xx | 1 retry with backoff, then `SAFE_FALLBACK` | facts-only apology |
| Retrieval empty or below `min_score` | no LLM call | "I don't have that in my sources." |
| Citation invalid | 1 retry, then `SAFE_FALLBACK` | facts-only apology |
| Number/performance/PII in output | `SAFE_FALLBACK` | facts-only apology |
| Streamlit reruns | `@st.cache_resource` for model + store; answer cache keyed by question + chunk set | — |

### 8.4 Performance budget (NFR-1: p50 < 8 s per answer)

| Segment | Budget |
|---|---|
| Guard + routing (regex only) | < 10 ms |
| Query embedding (MiniLM, CPU) | < 200 ms |
| Chroma search + fusion | < 100 ms |
| Groq generation | < 6 s |
| Validation + render | < 50 ms |

Ingestion: 5–10 pages, sequential polite fetch (≤ ~30 s network), embed ~600–1500 chunks in
batches of 32 (< 60 s on CPU) — well inside the 3-minute target.

### 8.5 Testing strategy

| Suite | Covers | Gate |
|---|---|---|
| `tests/test_guard.py` | 8-question refusal set, PII variants, 10 golden questions must classify `FACTUAL` | 0 false accepts (G4) |
| `tests/test_chunker.py` | token bound, ID determinism, table row-grouping, no-split of numerals/dates, fallback ordering | M3 exit |
| `tests/test_cleaner.py` | nav/script/banner removal, heading_path correctness, reject-log behaviour | M2 exit |
| `tests/test_validators.py` | citation in/out of allowlist, number-diff, sentence count, `as of` line | M6 exit |
| `tests/test_ingest_idempotency.py` | two runs → identical chunk count and IDs | M4 exit (§10.1.2) |
| `tests/test_retriever.py` | right-scheme top-k, out-of-corpus floor | M5 exit |
| `evaluate.py` | 10 golden + 8 refusal questions → `reports/eval_report.md` | §10.1 all gates |

Retrieval and guard tests run **offline** (no Groq key) so the suite is deterministic and
free; only `evaluate.py` and manual UI checks need the key (R9).

---

## 9. Requirement Traceability

| PRD ref | Architectural element |
|---|---|
| FR-1.1–FR-1.4 | `ingest/loader.py`, `allowed_domains` in `config.yaml`, `data/raw/` |
| FR-2.1–FR-2.5 | `ingest/cleaner.py`, `data/rejected.jsonl`, `MIN_PAGE_CHARS` fail-loud |
| FR-3.1–FR-3.4 | `ingest/chunker.py` (§5.3 parameters), `chunks.jsonl`, `chunks_preview.txt`, content-addressed IDs |
| FR-4.1–FR-4.5 | `ingest/embedder.py` (single local model), `ingest/vectorstore.py` (`mf_faqs`, cosine, upsert), `status.json` |
| FR-5.1–FR-5.5 | `app/retriever.py` filter → dense → hybrid fusion → dedupe → `min_score` floor |
| FR-6.1–FR-6.5 | `app/prompts.py`, `app/generator.py`, `app/cache.py`, Groq call, temp 0 |
| FR-7.1–FR-7.6 | `app/guard.py` (PII → advice → performance → scope), refusal messages, no persistence |
| FR-8.1–FR-8.5 | `app/ui.py` element table (§6.4) |
| FR-9.1–FR-9.2 | `app/disclaimer.py`, imported by UI, mirrored in README |
| FR-10.1–FR-10.3 | `config.py` precedence chain, `.gitignore`, `config.yaml` |
| FR-11.1–FR-11.3 | `run_ingest.py` report, `evaluate.py` report, redacted logging |
| G1–G8 | §5.3 chunking rationale, §6.2 retrieval, §6.3 validation, `chunks_preview.txt`, `content_as_of` rendering |
| NFR-1…NFR-9 | §8.4 budget, §5.4 local embedding, §8.3 caching/pinning, §5.3 assertions, `app/pipeline.py` boundary |
| D1–D8 | §4 repository layout maps 1:1 to the deliverable table |
| M0–M10 | §4 + §5 + §6; each milestone's exit criteria is the corresponding test gate in §8.5 |

---

## 10. Key Design Decisions (ADR summary)

| # | Decision | Alternatives rejected | Consequence |
|---|---|---|---|
| A1 | Deterministic rules guard runs **before** the LLM | Asking the LLM to self-refuse | Refusal correctness is testable offline and cannot be talked out of by prompt injection (G4) |
| A2 | Generation receives only retrieved chunk text + metadata | Passing raw page text | Forces grounding by construction; keeps the context inside the budget |
| A3 | Two independent post-LLM gates, terminal fallback | Rely on the prompt alone | A hallucinated number or a fake citation is suppressed, not displayed (R5, G2) |
| A4 | Content-addressed chunk IDs + upsert + prune | Random UUIDs, `add()` | Idempotent re-ingest is a one-line assertion (G6, §10.1.2) |
| A5 | `chunks.jsonl` is a first-class artefact, not a debug dump | keeping chunks only in Chroma | Reviewer can audit the corpus without running anything (G7, NFR-6) |
| A6 | Chunk text stored in Chroma *and* `chunks.jsonl` | one or the other | Retrieval is self-sufficient; the artefact stays canonical for humans |
| A7 | Hybrid dense + lexical fusion | dense-only | Guarantees the right scheme surfaces with 5 similar schemes in-corpus (FR-5.5) |
| A8 | Filter relaxes if it starves the result set | filter is a hard gate | A mis-tagged chunk cannot hide a correct answer (FR-5.4) |
| A9 | One `Answer` object is the single rendered artefact | separate rendering paths for UI/report/sample | UI, eval report and `sample_qa.md` cannot drift from each other |
| A10 | Answer cache keyed by question + chunk-id set | cache by question only | Cache stays correct when the corpus or `as of` dates change (FR-6.5, R2) |
| A11 | Streamlit (UI), no FastAPI | FastAPI + HTML, Gradio | Fastest path to the tiny UI; PRD §14 Q4 notes the choice is reversible behind `app/pipeline.py` |
| A12 | `data/status.json` corpus hash as the re-ingest gate | always re-ingest on boot | Meets G6 and NFR-1 startup cost; `--force` overrides |

---

## 11. Open Architecture Questions (from PRD §14, with a proposed default)

| # | Question | Proposed default pending owner confirmation | Impact if changed |
|---|---|---|---|
| Q1 | Which supplementary official pages beyond the 5 scheme pages? | Add HDFC AMC fee/charges page + one SEBI investor-education page; keep the allowlist at ≤ 12 URLs | Enlarges `config.yaml → sources`; no code change |
| Q2 | Direct Growth only, or also Direct IDCW / Regular? | Direct Growth only for M3; add variants as allowlist rows when the pipeline is stable | More chunks, more scheme-affinity testing (same code path) |
| Q3 | Freshness policy — hard-fail on old sources? | Show `content_as_of` and warn in the UI if > 90 days; do not hard-fail | If hard-fail is required, add a `freshness_max_days` check in `app/generator.py` step 5 |
| Q4 | UI framework | Streamlit, behind `app/pipeline.py` so the core is UI-agnostic | Gradio/FastAPI swap touches only `app/ui.py` |
| Q5 | Demo hosting | Local run + screen recording; Streamlit Community Cloud only if the key can be kept server-side | None architecturally |
| Q6 | Are Groww pages acceptable citations, or must they be hdfcmf.com PDFs? | Treat Groww as an accepted *mirror* per §4.1, but require `publisher` and `doc_type` on every citation so provenance is explicit; validation accepts any allowlisted URL | If PDFs-only, the allowlist and the fetch loader both change (PDF extraction path) |

---

## 12. Implementation Order (mapped to milestones)

| Milestone | Build order within the milestone | Exit signal |
|---|---|---|
| M0 Planning | this document + `config/sources.csv` | PRD + architecture reviewed; allowlist fixed |
| M1 Load | `config.yaml` → `loader.py` → `run_ingest.py` (fetch only) | 5 URLs fetched, per-URL report, raw cache on disk |
| M2 Clean | `cleaner.py` → block dump to a sample file | heading structure visible, no nav noise, no empty page passes |
| M3 Chunk | `chunker.py` → `topics.py` → `chunks.jsonl` + `chunks_preview.txt` | strategy validated on real text; metadata complete; token bound holds |
| M4 Embed+Store | `embedder.py` → `vectorstore.py` → `status.json` | 384-dim vectors, second run changes nothing, zero network |
| M5 Retrieval | `retriever.py` → threshold tuning against the golden set | right scheme in top-k; out-of-corpus filtered without the LLM |
| M6 Generation | `prompts.py` → `generator.py` → validators | ≤3 sentences, one valid citation, no unsourced numbers |
| M7 Guardrails | `guard.py` + `tests/test_guard.py` | 8/8 refusals, 0 false accepts |
| M8 UI | `app/pipeline.py` → `ui.py` → `disclaimer.py` | welcome, 3 examples, disclaimer, citation, as-of, context panel |
| M9 Evaluation | `evaluate.py` → `reports/eval_report.md` | all §10.1 gates pass |
| M10 Packaging | README, `sample_qa.md`, `.gitignore`, pinned `requirements.txt` | clone-and-run in < 15 min |

---

## 13. Appendix — Glossary

Terms as defined in PRD §15.2, plus: **block** (a semantic element from the cleaner),
**guard** (pre-LLM intent classifier), **fusion** (dense + lexical score combination),
**corpus hash** (sha256 over the sorted source hashes, used as the re-ingest gate),
**safe fallback** (the single terminal response used by every failure mode).
