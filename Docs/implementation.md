# Implementation Guide — Milestone 4: Mutual Fund FAQ RAG Chatbot

| Field | Value |
|---|---|
| Document | `Docs/implementation.md` |
| Version | 0.1 |
| Date | 2026-09-30 |
| Status | Ready to execute |
| Authority | `Docs/PRD.md` (what) → `Docs/architecture.md` (how) → **this file (in what order, to what exact contract)** |
| Audience | Cursor agent (phase-by-phase prompts in §2) + implementer verifying gates |

This file is the **plan of record for execution order**. It does not restate the PRD or the
architecture; it decomposes them into 11 build phases, each with an exact file list, frozen
contracts, ordered tasks, a test list, and a command-level exit gate that must pass before the
next phase starts.

---

## 1. Execution Protocol (applies to every phase)

### 1.1 Global rules for the coding agent

1. **One phase at a time.** Implement exactly the phase given. Do not start Phase N+1 files,
   do not add features "while you're in there", do not create placeholder TODOs.
2. **No new dependencies** beyond §1.2 without stopping and asking. If a phase genuinely needs one,
   say so and wait for approval.
3. **Layering is enforced by review, not by convention** (architecture §3.2):
   `config → ingest → app core → ui`, `evaluate.py` a peer of `ui`.
   - `ingest/*` must never import `app/*`, Streamlit, or the Groq SDK.
   - `app/*` (except `ui.py`) must never import Streamlit and must never read `data/raw/*.html`.
   - `app/ui.py` contains **zero** retrieval/generation logic — it calls one orchestrator.
   - `evaluate.py` imports app core only; it must not bypass guard or validators.
4. **No code comments.** Type hints on every public function. A one-line docstring only where the
   contract is non-obvious.
5. **Type hints + frozen dataclasses** for all cross-module data (§2.2). No bare `dict` across
   module boundaries, no `Any`.
6. **Secrets:** read only via `os.getenv`. Never log, never write a key into any artefact
   (including `status.json`, `chunks.jsonl`, `reports/eval_report.md`).
7. **Question text is never persisted or logged** (FR-7.6). Log `intent` + `reason_code` only.
   For PII-classified questions, the question body is dropped **before** any log record is built.
8. **Offline determinism:** all tests except `evaluate.py` run with no network and no API key.
   Ingest/store must work with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after the model is
   cached once.
9. **Case note:** the repo uses `Docs/` (capital D). Architecture §4 writes `docs/`. On this
   Windows checkout they are the same folder; use `Docs/` for `architecture.md`, `PRD.md`,
   `ProblemStatement.txt`, `implementation.md`, and put `sources.md` / `sample_qa.md` in `Docs/`
   too, so a Linux clone does not split into two folders.
10. **After every phase:** run the phase's gate command, then the full suite
    `python -m pytest -q`, then report: files created/changed, gate output, anything skipped.

### 1.2 Approved dependency set (PRD §6.2 — do not add to this list without approval)

```
python 3.10+ (3.11 recommended)
sentence-transformers   # all-MiniLM-L6-v2
torch                   # CPU wheel
chromadb
beautifulsoup4
lxml
trafilatura             # preferred main-content extractor + BS/lxml fallback
groq
streamlit
python-dotenv
pyyaml
requests
numpy
pytest
```

Version policy (R10, NFR-3): install unpinned once to generate a working set, then
`pip freeze` the subset above into `requirements.txt` with `==` pins. Keep an unpinned
`requirements.in` listing the same packages so the pins can be regenerated. Verify the pins on a
clean venv before M10.

### 1.3 Definition of done (a phase is done only when all are true)

- [ ] Every file in the phase's file table exists and does what the contract says.
- [ ] The phase's pytest file passes.
- [ ] `python -m pytest -q` (whole suite) is green.
- [ ] The gate command in the phase's §N.6 section produces the stated output.
- [ ] No new file outside the phase's table; no `.md` file other than this one and the ones named.
- [ ] No `TODO`, `FIXME`, `pass  # stub`, or `NotImplementedError` left anywhere.
- [ ] No secret, absolute machine path, or `data/raw`/`data/chroma` content committed.

### 1.4 Phase order and dependency graph

| Phase | Milestone | Produces | Consumes | Gate test file |
|---|---|---|---|---|
| 0 | — | scaffold, `config.py`, `config/config.yaml`, `requirements*`, `.gitignore`, `.env.example` | — | `tests/test_config.py` |
| 1 | M1 Load | `ingest/loader.py`, raw cache, fetch report | 0 | `tests/test_loader.py` |
| 2 | M2 Clean | `ingest/cleaner.py`, `data/rejected.jsonl` | 0,1 | `tests/test_cleaner.py` |
| 3 | M3 Chunk | `ingest/chunker.py`, `ingest/topics.py`, `chunks.jsonl`, `chunks_preview.txt`, `docs/sources.csv` | 2 | `tests/test_chunker.py` |
| 4 | M4 Embed+Store | `ingest/embedder.py`, `ingest/vectorstore.py`, `data/status.json` | 3 | `tests/test_ingest_idempotency.py` |
| 5 | M5 Retrieve | `app/retriever.py` | 4 | `tests/test_retriever.py` |
| 6 | M6 Generate | `app/prompts.py`, `app/generator.py`, `app/validators.py`, `app/cache.py` | 5 | `tests/test_validators.py` |
| 7 | M7 Guard | `app/guard.py` | 0 | `tests/test_guard.py` |
| 8 | M8 UI | `app/pipeline.py`, `app/ui.py`, `app/disclaimer.py` | 6,7 | `tests/test_pipeline.py` |
| 9 | M9 Evaluate | `evaluate.py`, `data/golden_questions.jsonl`, `reports/eval_report.md` | 8 | `tests/test_eval_harness.py` |
| 10 | M10 Package | `README.md`, `Docs/sources.md`, `Docs/sample_qa.md`, pinned pins | 9 | — |

Phase 7 (guard) is written **after** 6 because the guard's contract is only observable against the
full answer object; it has no dependency on Groq, so it can be pulled earlier if preferred —
but do not reorder without noting it here.

---

## 2. Shared Contracts (frozen in Phase 0 — do not change in later phases)

### 2.1 `config/config.yaml` (complete key set; no key is introduced after Phase 0)

```yaml
sources:
  - {id: s1, url: "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", domain: groww.in,
     doc_type: scheme_page, scheme_name: "HDFC Large Cap Fund - Direct Growth", scheme_category: large_cap,
     plan: direct, option: growth, publisher: "Groww (mirroring HDFC MF official data)"}
  - {id: s2, url: "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",      ...}
  - {id: s3, url: "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth", ...}
  - {id: s4, url: "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth", ...}
  - {id: s5, url: "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth", ...}
allowed_domains: [groww.in, hdfcmf.com, sebi.gov.in, amfiindia.com, myamfi.in]

paths: {raw_dir: data/raw, chunks_dir: data/chunks, persist_dir: data/chroma, cache_dir: data/cache,
        models_dir: data/models, status_file: data/status.json, rejected_file: data/rejected.jsonl}

chunking: {size: 900, overlap: 150, min_chars: 40, max_chunks_per_page: 400, max_tokens: 250,
           min_page_chars: 400}

retrieval: {top_k: 5, min_score: 0.28, candidate_multiplier: 4, lexical_weight: 0.6,
            rrf_k: 60, scheme_affinity: true}

embedding: {model: "sentence-transformers/all-MiniLM-L6-v2", dim: 384, batch_size: 32,
            normalize: true, offline_env: true}

generation: {temperature: 0.0, max_tokens: 300, seed: 42, max_sentences: 3, max_words: 60,
             retries_on_citation_failure: 1, retries_on_http_error: 1}

topics:                       # heading/block text -> controlled topic vocabulary (order matters)
  exit_load: ["exit load", "exit charges"]
  expense_ratio: ["expense ratio", "ter", "total expense", "charges"]
  min_sip: ["minimum sip", "min sip", "minimum instal", "minimum investment amount", "min lump sum"]
  lockin_80c: ["lock-in", "lock in", "lockin", "80c", "section 80c"]
  riskometer: ["riskometer", "risk o meter"]
  benchmark: ["benchmark", "index"]
  statements_download: ["statement", "tax document", "how to download", "capital gains", "tax statement"]
  nav: ["nav", "net asset value"]
  plan_direct_vs_regular: ["direct vs regular", "direct plan", "regular plan", "direct and regular"]
  objective: ["objective", "about the scheme", "scheme profile", "investment objective"]
  entry_load: ["entry load"]
  min_lump_sum: ["minimum lump sum", "lump sum"]

refusal:
  advice: ["should i", "which is best", "which of these", "best for me", "better than", "recommend",
           "suggest", "suitable for me", "is it good", "good for my", "my retirement", "portfolio",
           "allocate", "asset allocation", "worth buying", "good time to buy", "good time to sell",
           "good time to exit", "good time to enter", "can i buy", "can i sell", "hold or",
           "should we", "which fund should", "tell me if i should"]
  performance: ["return", "returns", "cagr", "xirr", "% return", "how much did", "how much has",
                "performance of", "best performing", "ranking", "ranked", "nav history",
                "compare", "vs ", "versus", "outperformed", "underperformed", "yield", "gain",
                "profit", "irr"]
  pii_patterns:
    - "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b"                 # PAN
    - "\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\b"             # Aadhaar
    - "\\b[6-9]\\d{9}\\b|\\b\\d{10}\\b"              # phone
    - "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"  # email
    - "\\b(otp|ifsc|cvv|pan number|aadhaar|account number|demat)\\b"
    - "\\b(otp|verification code)\\b[^\\n]{0,20}\\b\\d{4,8}\\b"
  educational_links:
    advice: "https://www.investor.gov.in/"
    performance: "https://www.investor.gov.in/"
    pii: "https://www.investor.gov.in/"

out_of_corpus: {known_foreign_schemes: ["parag parflex", "parag parflex", "axis small cap", "mirae asset",
                                        "sbi bluechip", "icici nifty fifty", "kotak flexicap", "dsp small cap",
                                        "franklin templeton", "tata nipun", "bandhan mutual"]}

freshness: {warn_after_days: 90}
```

Rules for the lists: keyword matching is **case-insensitive substring** on a normalised question
(lowercase, punctuation → space, whitespace collapsed). Regex patterns (`pii_patterns`) are
`re.search` case-insensitive unless the pattern says otherwise. If the pipeline cannot deliver
`answer_ratio` etc., drop the key rather than inventing one — the key set is frozen.

### 2.2 `app/models.py` — shared dataclasses (frozen)

```python
@dataclass(frozen=True) class SourceSpec:  id, url, domain, doc_type, scheme_name|None, scheme_category|None,
                                           plan|None, option|None, publisher
@dataclass(frozen=True) class FetchResult:  source_id, url, status, http_code, bytes, sha256, fetched_at, cache_hit, error
@dataclass(frozen=True) class Block:        kind, heading_path, text, row_group: int|None
@dataclass(frozen=True) class CleanDocument:source_id, url, page_title, blocks: tuple[Block,...], extracted_chars
@dataclass(frozen=True) class Chunk:        # fields == PRD 6.4 metadata schema + source_id + token_count
@dataclass(frozen=True) class RetrievalHit: chunk: Chunk, score: float, dense_rank: int, lexical_rank: int|None
@dataclass(frozen=True) class RetrievalResult: hits: tuple[RetrievalHit,...], status, filtered_topic, total_candidates
@dataclass(frozen=True) class Decision:     intent, educational_url: str|None, reason_code: str
@dataclass(frozen=True) class Answer:        text, source_url, doc_type, publisher, content_as_of,
                                           used_chunk_ids: tuple[str,...], status, was_cached: bool
@dataclass(frozen=True) class IngestStatus:  corpus_hash, chunk_count, model, collection, last_ingest_at, source_hashes
```

* `Chunk.to_metadata()` → the flat `dict[str, str|int|float]` stored in Chroma (scalars only —
  Chroma rejects nested values). `Chunk.from_metadata(md, text)` reconstructs a full `Chunk`.
* `Chunk.normalised_text` / `Chunk.content_hash` / `Chunk.chunk_id` are **properties** so the ID
  derivation (architecture §5.3) has exactly one implementation.
* `Answer.render()` returns the exact UI string, in this order:
  `"{text}\n\nSource: {url} ({doc_type} - {publisher})\nLast updated from sources: {content_as_of}"`.
  `docs/sample_qa.md` and `reports/eval_report.md` consume this same method (ADR A9).
* `Answer.status ∈ {ok, out_of_corpus, refused_advice, refused_performance, refused_pii, suppressed}`.
  `RetrievalResult.status ∈ {ok, out_of_corpus}`.
* `Decision.intent ∈ {FACTUAL, ADVICE, PERFORMANCE, PII, OUT_OF_CORPUS}`.

### 2.3 `app/disclaimer.py` — frozen strings (single owner per string)

```python
SHORT_NOTE = "Facts-only. No investment advice."
DISCLAIMER = (
    "Facts-only. No investment advice. This assistant answers only from official public sources "
    "(HDFC Mutual Fund / Groww scheme pages, SEBI, AMFI) for the five schemes listed in the README. "
    "It does not recommend, compare or rate funds, and does not compute returns. Mutual fund "
    "investments are subject to market risks; read the scheme's official factsheet, KIM and SID "
    "before investing. Verify every detail using the source link provided."
)
```

`app/ui.py` imports it; `README.md` and `Docs/sample_qa.md` mirror it **verbatim**. Refusal
messages live only in `app/guard.py`; `SAFE_FALLBACK` lives only in `app/generator.py`.

### 2.4 `config.py` — configuration loader

* `load_config(path: str|Path|None = None) -> AppConfig` (module-level `@lru_cache` on the resolved
  path so Streamlit reruns do not re-parse).
* `AppConfig` fields: `sources: tuple[SourceSpec,...]`, `allowed_domains: frozenset[str]`,
  `chunking: ChunkingParams`, `retrieval: RetrievalParams`, `embedding: EmbeddingParams`,
  `generation: GenerationParams`, `topics: dict[str, tuple[str,...]]`,
  `refusal: RefusalRules`, `out_of_corpus: frozenset[str]`, `paths: PathParams`,
  `freshness_warn_days: int`.
* `PathParams` resolves to absolute `Path` objects relative to the repo root (parent of `config.py`).
* Precedence chain, applied once in `load_config` (FR-10.1):
  **real environment variable > `.env` file > `config.yaml` value > code default.**
  `.env` is loaded with `python-dotenv` `load_dotenv(override=False)`, from repo root.
* Overridable by env: `GROQ_API_KEY`, `GROQ_MODEL`, `TOP_K`, `MIN_SCORE`, `CHUNK_SIZE`,
  `CHUNK_OVERLAP`, `PERSIST_DIR`, `RAW_DIR`, `EMBED_MODEL`.
* `get_groq_api_key() -> str|None` and `get_groq_model() -> str` are the only places secrets are
  read. `config.py` performs **no I/O other than reading `config.yaml`/`.env`**.

### 2.5 Path constants (no module may hardcode a path string)

All paths come from `AppConfig.paths`. `ingest/vectorstore.py` exposes
`get_client(cfg)`, `get_collection(cfg)` and `read_status(cfg)`; `app/*` uses those, never
`chromadb.PersistentClient(...)` directly.

---

## 3. Phase 0 — Scaffold, configuration, contracts

**Goal.** A runnable, importable skeleton with the frozen contracts above, so Phases 1–10 are pure
feature work with no design churn.

### 3.1 Files

| Path | Action | Notes |
|---|---|---|
| `app/__init__.py`, `ingest/__init__.py`, `tests/__init__.py` | create | empty |
| `app/models.py` | create | §2.2 verbatim |
| `app/disclaimer.py` | create | §2.3 verbatim |
| `config.py` | create | §2.4 |
| `config/config.yaml` | create | §2.1 verbatim |
| `requirements.in`, `requirements.txt` | create | §1.2 policy |
| `.env.example` | create | `GROQ_API_KEY=`, `GROQ_MODEL=llama-3.1-8b-instant`, `TOP_K=5`, `MIN_SCORE=0.28`, `CHUNK_SIZE=900`, `CHUNK_OVERLAP=150`, `PERSIST_DIR=data/chroma`, `RAW_DIR=data/raw` — all values empty or defaults |
| `.gitignore` | create | `.env`, `data/raw/`, `data/chroma/`, `data/cache/`, `data/models/`, `__pycache__/`, `*.log`, `.venv/`, `.pytest_cache/` |
| `tests/conftest.py` | create | fixtures: `tmp_config` (config written to tmp_path), `real_config` (repo `config/config.yaml`), `clean_doc_factory` (synthetic `CleanDocument`), `no_network` (monkeypatch `requests.Session.request` to raise) |
| `tests/test_config.py` | create | see §3.5 |

Directories `data/{raw,chunks,chroma,cache,models}` are created by the code that writes into them
(Phases 1, 3, 4) — do not commit empty dirs.

### 3.2 Ordered tasks

1. Create packages and the venv:
   `python -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `pip install -r requirements.in`.
2. Write `app/models.py` with the properties `content_hash`, `chunk_id`, `normalised_text`
   implemented **once** here (normalisation for hashing = lowercase, collapse whitespace, strip
   punctuation except `.` and `%`; `chunk_id = f"{source_id}::{topic}::{content_hash[:8]}"`).
3. Write `config.py`, then `config/config.yaml` (copy §2.1 exactly; the 5 sources come from
   PRD §4.1, IDs `s1`–`s5`).
4. Write `.env.example`, `.gitignore`, `requirements.in`; install; then pin into
   `requirements.txt` from the resolved working set.
5. Add `Doc/source_id` values `s1`–`s5` and `publisher` per source. Confirm every `url` host is in
   `allowed_domains`; `load_config` must **raise** if a source's host is not allowlisted.
6. `tests/conftest.py` fixtures.

### 3.3 Rules

* The five source URLs are exactly PRD §4.1. Supplementary URLs (PRD §14 Q1) are **not** added in
  this phase; adding a source is a config-only change and must be flagged to the user first.
* `config.py` must not import `requests`, `chromadb`, `streamlit`, or `groq`.

### 3.4 Gate

```
python -c "from config import load_config; c=load_config(); print(len(c.sources), c.allowed_domains, c.paths.persist_dir)"
```
Expected: `5 frozenset({...}) <abs path>\data\chroma`

### 3.5 Tests (`tests/test_config.py`)

| Case | Assertion |
|---|---|
| loads 5 sources with ids s1..s5 | ids sorted, all hosts allowlisted |
| source with foreign host | `load_config` raises `ConfigError` |
| env overrides yaml | `TOP_K=9` in env → `cfg.retrieval.top_k == 9` |
| `.env` does not override real env | set both, real env wins |
| missing optional key | code default used, no `KeyError` |
| paths are absolute and under repo root | `cfg.paths.raw_dir.is_absolute()` |

> **Cursor prompt — Phase 0**
> Implement Phase 0 of `Docs/implementation.md` in `C:\Users\Welcome\Milestone4`. Read §1 (rules),
> §2.2–§2.4 (frozen contracts) and §3 of that file first and copy the contracts verbatim — do not
> rename dataclass fields, do not change the `config.yaml` key set, do not add dependencies. Create
> the venv, write the files, run the §3.4 gate and `python -m pytest -q tests/test_config.py`, then
> report files created, gate output and anything you could not complete. No comments in code; type
> hints on every public function.

---

## 4. Phase 1 — M1 Load (`ingest/loader.py`)

**Goal.** The allowlist becomes cached raw HTML plus a per-URL status report. No parsing.

### 4.1 Files

| Path | Action | Notes |
|---|---|---|
| `ingest/loader.py` | create | fetch + cache + report |
| `run_ingest.py` | create | CLI shell; only `--config`/`--force`/`--sources`/`--dry-run` flags parsed, stage wiring added in later phases |
| `tests/test_loader.py` | create | offline |

### 4.2 Contract

```python
def fetch_source(spec: SourceSpec, cfg: AppConfig, *, force: bool = False) -> FetchResult
def fetch_all(specs: Sequence[SourceSpec], cfg: AppConfig, *, force: bool = False) -> list[FetchResult]
def render_report(results: Sequence[FetchResult]) -> str        # the FR-11.1 table
def cached_path(spec: SourceSpec, cfg: AppConfig) -> Path       # data/raw/<domain>/<source_id>.html
```

### 4.3 Ordered tasks

1. `User-Agent`: descriptive and static, e.g.
   `Milestone4-MF-FAQ-Research/0.1 (+contact: <team email placeholder>)`. Never a browser UA.
2. **SSRF guard:** reject any spec whose scheme is not `https` or whose host is not in
   `allowed_domains` **before** opening a connection.
3. Politeness: per-host serialisation, `>= 2 s` sleep between requests to the same host
   (`time.sleep`, not a busy loop), `requests.Session` with `timeout=30`.
4. Retries: 3 attempts, exponential backoff `2**attempt` seconds + `random.uniform(0, 0.5)` jitter;
   honour `Retry-After` when present and `<= 60`.
5. Redirects: follow manually, at most 3 hops, and only to allowlisted hosts; a cross-host redirect
   is `status="blocked"` with `reason=redirect_off_allowlist`.
6. Status mapping: `200` → `ok`; `403` → `blocked`; `429` → `blocked` (rate limited); other `4xx`
   → `error`; `5xx`/connection errors/timeouts → `error` with `error` text. **Never** try to
   bypass a block (no UA rotation, no proxy, no retry on 403).
7. Cache: write `data/raw/<domain>/<source_id>.html` and `<source_id>.meta.json`
   (`{url, http_code, fetched_at, sha256, bytes, content_type}`). Reuse the cached file when it
   exists and `--force` was not passed, returning `cache_hit=True`.
8. `sha256` over the **response body bytes**; `fetched_at` as ISO-8601 UTC (`...Z`).
9. Non-collectable links: while scanning the raw HTML for `<a href>`, record links whose host is
   not allowlisted into `data/raw/<source_id>.noncollectable.json` for the record. **Do not fetch
   them** (FR-1.4). This is a passive scan of already-downloaded bytes only.
10. `fetch_all` never raises for a single failure; it returns one `FetchResult` per requested spec.
11. `run_ingest.py --dry-run` in this phase = load + fetch + report. Print the report and
    `exit(0)` when every requested spec is `ok` or `cache_hit`, else `exit(1)`.
12. `render_report` columns:
    `source_id | http | cache | bytes | sha256[:8] | status | error` plus a totals line
    (n ok, n blocked, n error, elapsed seconds).

### 4.4 Rules

* No HTML parsing beyond the link scan in step 9.
* Never log the raw question or a key. Fetch logs use `[load] <source_id>` prefix only.
* If **all** sources fail, `run_ingest.py` exits 1 with a one-line summary — never an empty corpus.

### 4.5 Gate

```
python run_ingest.py
Get-ChildItem -Recurse data\raw -Filter *.html | Measure-Object
```
Expected: report table for 5 sources, 5 HTML files + 5 meta json, exit code 0. Re-run must show
`cache=True` for all 5.

### 4.6 Tests (`tests/test_loader.py`, all offline)

| Case | Assertion |
|---|---|
| disallowed host | `fetch_source` raises/returns `blocked` without any network call (assert the session was never opened) |
| cache reuse | two calls → one network call, second has `cache_hit=True` |
| `--force` | second call refetches |
| sha256 stability | cached `sha256` equals sha256 of the file bytes |
| 403 | `status == "blocked"`, no retry attempted |
| 429 with `Retry-After: 1` | retried at most twice, honouring the header (monkeypatch `time.sleep`) |
| report columns | header row contains `source_id` and `cache` |

> **Cursor prompt — Phase 1**
> Implement Phase 1 of `Docs/implementation.md` (§4 only). Build `ingest/loader.py` and the fetch
> portion of `run_ingest.py` to the exact contract and ordered tasks in §4.2–§4.4. Respect the
> layering rule: `ingest/` must not import `app/*`, Streamlit, or `groq`. Run the §4.5 gate and
> `python -m pytest -q tests/test_loader.py`, then report the per-URL report verbatim. If a URL is
> blocked or the page is JS-rendered, report it — do not add a headless-browser dependency or a
> bypass; stop and ask.

---

## 5. Phase 2 — M2 Clean (`ingest/cleaner.py`)

**Goal.** Raw HTML → ordered semantic **blocks** + normalised text, losslessly with respect to facts.

### 5.1 Files

| Path | Action | Notes |
|---|---|---|
| `ingest/cleaner.py` | create | extraction + normalisation + reject log |
| `tests/test_cleaner.py` | create | uses small inline HTML fixtures, no network |

### 5.2 Contract

```python
def clean_html(html: str, spec: SourceSpec) -> CleanDocument
def clean_file(path: Path, spec: SourceSpec, cfg: AppConfig) -> CleanDocument
def normalise_text(text: str) -> str
def write_reject_log(cfg: AppConfig, source_id: str, extracted_chars: int, reason: str) -> None
def dump_blocks(doc: CleanDocument) -> str        # debug artefact: data/chunks/blocks_<id>.txt
```

### 5.3 Ordered tasks

1. **Extraction:** `trafilatura.extract(html, output_format="lxml", include_tables=True,
   include_comments=False)` as the primary path; when it returns fewer than
   `min_page_chars` characters, fall back to a BeautifulSoup + `lxml` structural walk. Record which
   path was used in the `CleanDocument` debug dump (not in the chunk metadata).
2. **Removal set** (FR-2.1): `script, style, noscript, nav, header, footer, aside, form, iframe,
   svg`; plus heuristics — elements whose `role`/`aria-label`/`class`/`id` match
   `(cookie|consent|banner|advert|ad-|\bads?\b|breadcrumb|popup|modal|skip-link|sr-only)`, and any
   element whose stripped text is identical to a nav/header label.
3. **Block emission**, in document order:
   - `h1`–`h6` → `Block(kind="heading", heading_path=<ancestor chain>, text=<heading text>)`
   - `p`, and any block-level text container → `Block(kind="para", ...)`
   - `li` → one `Block(kind="list_item", ...)` per item
   - `table` → header row as a `heading` block, then **one `Block(kind="table_row", row_group=n)`
     per body row** with the header cells prefixed (`"Expense ratio | Direct Growth | 0.5%"`).
     `row_group` is the group index within the table so the chunker can keep one row-group together.
   - `dl` → `Block(kind="definition", text="<term>: <definition>")`
   - `heading_path` is the **ancestor heading chain** joined with `" > "`, e.g.
     `"Fees > Exit load"`. Track it with a stack while walking.
4. **Normalisation** (`normalise_text`, FR-2.3), in this order:
   `unicodedata.normalize("NFKC", t)` → map `—–−` to `-` → map smart quotes/ellipsis to ASCII →
   `NBSP`/`\u202f` → single space → drop zero-width (`\u200b-\u200d\ufeff`) → strip →
   collapse whitespace runs to one space.
   **Protected, never modified:** digits, decimal points, thousands separators, `%`, `₹`, `Rs.`,
   units (`bps`, `years`, `months`, `days`), ISO/Indian dates, and any `as on <date>` string.
   Verify protection in a test by asserting `"1.5%"`, `"₹500"`, `"1,00,000"`, `"as on 30 Sep 2026"`
   survive byte-for-byte.
5. **Drop** blocks whose normalised text is `< 3` characters (pure separators) — these are not the
   40-char chunk minimum, that is a chunker concern.
6. **Reject log** (FR-2.5): if `extracted_chars < cfg.chunking.min_page_chars` →
   `write_reject_log(...)` with `reason="below_min_page_chars"`, and `run_ingest.py` exits 1
   **after** processing all sources (fail loudly, still write the other sources).
   `data/rejected.jsonl` is append-only JSONL: `{source_id, extracted_chars, reason, ts}`.
7. `clean_file` sets `page_title` from `<h1>` or `<title>`, stripped and normalised.
8. `dump_blocks` writes `[{i}] {kind} | heading_path | chars` + the text, for human inspection in
   M2 exit review.

### 5.4 Rules

* Never paraphrase, reorder, summarise or de-duplicate sentences. "Lossless with respect to
  factual content" means the *characters of every number and label are unchanged* (FR-2.4).
* Do not introduce summarisation, keyword extraction, or an LLM at this stage.
* The cleaner must not read `config.yaml` sources or call the network.

### 5.5 Gate

```
python run_ingest.py
Get-Content data\chunks\blocks_s1.txt -TotalCount 60
Get-Content data\rejected.jsonl -ErrorAction SilentlyContinue
```
Expected: `blocks_s1.txt` shows heading paths, list items and table rows with real numbers; no nav
or cookie text; no page in the reject log; exit code 0.

### 5.6 Tests (`tests/test_cleaner.py`)

| Case | Assertion |
|---|---|
| nav/footer/script removal | fixture with `<nav>Menu</nav>` → `"Menu"` absent from all block text |
| cookie banner heuristics | element with `class="cookie-banner"` removed |
| heading path | `h2 Fees` > `h3 Exit load` > `p` → `heading_path == "Fees > Exit load"` |
| table rows | header repeated on each row block, `row_group` increments per group |
| protected spans | `1.5%`, `₹500`, `1,00,000`, `as on 30 Sep 2026` survive `normalise_text` |
| unicode normalisation | `—` → `-`, `’` → `'`, NBSP → space, zero-width removed |
| short page | `extracted_chars` below threshold → `write_reject_log` called, reason recorded |
| losslessness | concatenated block text contains every numeral of the fixture, in order |

> **Cursor prompt — Phase 2**
> Implement Phase 2 of `Docs/implementation.md` (§5 only): `ingest/cleaner.py` to the contract in
> §5.2, in the exact order of §5.3 (trafilatura first, BeautifulSoup/lxml fallback, removal set,
> block kinds, heading-path stack, protection of numerals/dates/units, reject log). Do not paraphrase
> or drop facts. Run the §5.5 gate for all 5 sources and paste the first 40 lines of
> `data/chunks/blocks_s1.txt`, then `python -m pytest -q tests/test_cleaner.py`. Report any page that
> produced under `min_page_chars`.

---

## 6. Phase 3 — M3 Chunk (`ingest/chunker.py`, `ingest/topics.py`)

**Goal.** Section-aware, heading-anchored, token-bounded chunks with the full PRD §6.4 metadata
schema, content-addressed IDs, and two human-readable artefacts.

### 6.1 Files

| Path | Action | Notes |
|---|---|---|
| `ingest/chunker.py` | create | chunking + `chunks.jsonl` + `chunks_preview.txt` |
| `ingest/topics.py` | create | heading/text → controlled topic vocabulary |
| `data/golden_questions.jsonl` | create | 10 rows, PRD §10.3 (needed by Phase 5; create now, keep stable) |
| `config/sources.csv` | create | D2 columns; regenerate from the corpus, not hand-written |
| `tests/test_chunker.py` | create | |

### 6.2 Contract

```python
def classify_topic(heading_path: str, text: str, cfg: AppConfig) -> str
def chunk_document(doc: CleanDocument, spec: SourceSpec, cfg: AppConfig) -> list[Chunk]
def count_tokens(text: str) -> int                     # simple whitespace+punct heuristic, no model load
def write_chunks(chunks: Sequence[Chunk], cfg: AppConfig) -> None   # jsonl + preview, atomic
def write_sources_csv(chunks: Sequence[Chunk], path: Path) -> None
```

`count_tokens` must not require the embedding model (chunker runs before the model exists). Use
`max(1, ceil(len(text) / 4))` as the estimate and let Phase 4's real tokenizer assert the real
value. Chunker asserts the **estimate** `<= max_tokens`; Phase 4 asserts the **real** count.

### 6.3 Chunking algorithm (implement exactly this order)

1. Group blocks into **sections** by `heading_path`: consecutive blocks sharing a heading path form
   one section, in document order.
2. **Tables:** one chunk per `row_group` (all rows of a group together), header repeated. If a group
   exceeds `size`, split by rows, never mid-row.
3. **Heading / labelled blocks:** if a section's combined text `<= size`, emit **one** chunk whose
   `section_heading` is the last element of the heading path. This is the primary case for these
   pages ("Expense ratio: 0.5%").
4. **Secondary split** when a section exceeds `size`: sliding 900/150 windows over the section's
   text, each window **snapped forward** to the nearest sentence boundary from `".", "?", ";", ":"`.
   If no boundary exists within the window, snap **backward** to the previous boundary; if the
   first 40 characters contain no boundary, use the raw window.
5. **Never split** (assert in tests): a numeral with its unit/`%`/`₹`, a table row, a list item, or an
   `as on <date>` stamp. Implement as: if the split point falls inside a matched protected span,
   move the split point past the span.
6. **Overlap:** the next window starts `overlap` characters before the previous window ended, then
   snaps forward to a sentence boundary. Overlap text is prefixed to the next chunk only when the
   snapped start lands on a sentence boundary.
7. **Min chunk:** drop and log any chunk `< min_chars` (40). Count drops in the preview footer.
8. **Max chunks:** if a document yields `> max_chunks_per_page` (400), raise
   `ChunkingError` — that is a cleaning failure signal, not a truncation.
9. **Fallback ordering** (architecture §5.3): heading block → sentence-aligned window → skip.
   A chunk is never emitted without `source_url` and `doc_type`; enforce in code, not by comment.
10. `chunk_index` is the 0-based index **within the document**. `topic` from `classify_topic`.
    `content_as_of` = the most recent `as on <date>` / `Last updated` date found in the section
    text; if none, `spec` has no date → use the ingest date **only if** `config.yaml` sets
    `content_as_of_fallback: ingest_date`, else `""`. Default: `""` — never invent a date.
11. `content_hash` = `sha256(normalised_text)`, `chunk_id` = `f"{source_id}::{topic}::{hash[:8]}"`.
    Both come from the `Chunk` properties in `app/models.py` (single implementation).

### 6.4 Topic classification

* `config.yaml → topics` is an **ordered** mapping; the first key whose any phrase occurs in
  `heading_path + " " + text` (case-insensitive) wins; otherwise `general`.
* Phrase matching must be word-boundary aware for short tokens (`ter`, `nav`, `80c`) so `ter` does
  not match `"after"` and `nav` does not match `"navigation"`.
* `ingest/topics.py` owns the vocabulary constant `TOPIC_VOCABULARY` and raises if `config.yaml`
  contains a key outside PRD §6.4's controlled list.

### 6.5 Artefacts

* `data/chunks/chunks.jsonl` — one JSON object per line, **full** record including `text`, keys in
  the PRD §6.4 order plus `source_id` and `token_count`. Write via temp file + `os.replace`
  (atomic). Write it from scratch each run (never append).
* `data/chunks/chunks_preview.txt` — for each chunk:
  `[n] source_url | doc_type | topic | section | chars=<n> tokens=<n>` then the verbatim text,
  then a `---` rule. Footer: totals, per-topic counts, dropped-chunk count, per-source chunk count.
* `config/sources.csv` — one row per **source**: `source_id, url, publisher, doc_type, scheme_name,
  scheme_category, plan, option, allowed_domain, first_fetched_at, last_fetched_at, sha256,
  chunk_count, retrieved_at`. Derive it from the loader meta + chunk counts (D2).

### 6.6 Rules

* No model load, no Chroma, no network in this phase.
* Chunk text is **verbatim** normalised block text — no prefix like "Expense ratio: " that is not
  in the source, and no truncation of the first line.

### 6.7 Gate

```
python run_ingest.py
(Get-Content data\chunks\chunks.jsonl | Measure-Object).Count
Get-Content data\chunks\chunks_preview.txt -TotalCount 40
Get-Content config\sources.csv
```
Expected: 100–400 chunks, every line of `chunks.jsonl` parses and has all PRD §6.4 keys, no chunk
longer than 900 chars, preview shows real section headings and intact numbers, `sources.csv` has 5
rows.

### 6.8 Tests (`tests/test_chunker.py`)

| Case | Assertion |
|---|---|
| token bound | `count_tokens(c.text) <= cfg.chunking.max_tokens` for every chunk |
| id determinism | chunking the same doc twice gives identical `chunk_id` list |
| content addressing | changing one character in a block changes that chunk's `chunk_id` and no other |
| table row-group | a 3-row table yields one chunk with all 3 rows and the header repeated |
| no split of numerals | no chunk ends mid-number: regex `/\d+(\.\d+)?%?$/`-style scan finds no dangling number; `"as on"` never spans a boundary |
| heading anchoring | a labelled block smaller than `size` yields exactly one chunk with `section_heading` set |
| overlap | consecutive windows from the same section share `>= 1` sentence and total length `<= size` |
| min chunk | a 20-char block is dropped and counted |
| max chunks guard | a synthetic 50 000-block doc raises `ChunkingError` |
| metadata completeness | every chunk has `source_url`, `doc_type`, `topic`, `publisher`, `chunk_index`, `char_len`, `content_hash` |
| artefact round-trip | each `chunks.jsonl` line → `Chunk` → `to_metadata()` → no exception, no nested value |

> **Cursor prompt — Phase 3**
> Implement Phase 3 of `Docs/implementation.md` (§6 only): `ingest/chunker.py` + `ingest/topics.py`
> to §6.2–§6.5, in the exact algorithm order of §6.3 (sections by heading path → table row-groups →
> heading block → 900/150 sentence-snapped windows → overlap → min/max guards). Use the `Chunk`
> properties in `app/models.py` for `content_hash`/`chunk_id` — do not reimplement hashing. Also
> create `data/golden_questions.jsonl` (10 rows, exact questions and expectations from PRD §10.3,
> columns: `id, category, question, expect_topic, expect_refusal`) and generate `config/sources.csv`
> from the loader metadata. Run the §6.7 gate and `python -m pytest -q tests/test_chunker.py`, then
> report the chunk count, the per-topic counts and any chunk you had to drop.

---

## 7. Phase 4 — M4 Embed + Store (`ingest/embedder.py`, `ingest/vectorstore.py`)

**Goal.** 384-dim local vectors in a persistent, idempotent ChromaDB collection plus a
`status.json` re-ingest gate. Zero network after the model is cached.

### 7.1 Files

| Path | Action | Notes |
|---|---|---|
| `ingest/embedder.py` | create | single model, used by both paths |
| `ingest/vectorstore.py` | create | client, collection, upsert, prune, status |
| `run_ingest.py` | modify | add embed → store → prune → status → report stages |
| `tests/test_ingest_idempotency.py` | create | uses a temp `persist_dir` |

### 7.2 Contract

```python
# embedder
def get_model(cfg: AppConfig) -> SentenceTransformer          # lazy process singleton
def embed_documents(texts: Sequence[str], cfg: AppConfig) -> np.ndarray   # (n, 384) float32, L2-normalised
def embed_query(text: str, cfg: AppConfig) -> np.ndarray                  # (384,)
def real_token_count(text: str, cfg: AppConfig) -> int                    # model tokenizer, no truncation

# vectorstore
def get_client(cfg: AppConfig) -> chromadb.ClientAPI
def get_collection(cfg: AppConfig, *, create: bool) -> Collection
def upsert_chunks(chunks: Sequence[Chunk], vectors: np.ndarray, cfg: AppConfig) -> int
def prune(chunks: Sequence[Chunk], cfg: AppConfig) -> int
def query_chunks(vector: np.ndarray, n: int, cfg: AppConfig, where: dict|None) -> QueryResult
def write_status(cfg: AppConfig, chunks: Sequence[Chunk], source_hashes: Mapping[str,str]) -> IngestStatus
def read_status(cfg: AppConfig) -> IngestStatus|None
def corpus_hash(source_hashes: Mapping[str,str]) -> str
```

### 7.3 Ordered tasks

1. `get_model` returns a module-level singleton (`functools.lru_cache`); set
   `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME` to `paths.models_dir` **before** importing/loading
   sentence-transformers so the model caches inside the repo, and set `HF_HUB_OFFLINE=1` +
   `TRANSFORMERS_OFFLINE=1` after the first successful load.
2. `SentenceTransformer(cfg.embedding.model, device="cpu")` with
   `model_kwargs={"torch_dtype": torch.float32}`; verify `get_sentence_embedding_dimension() == 384`
   at load and raise `ConfigError` otherwise.
3. `embed_documents`: `model.encode(list(texts), batch_size=32, normalize_embeddings=True,
   show_progress_bar=False, convert_to_numpy=True)`, dtype `float32`, assert shape `(n, 384)`.
4. `real_token_count` uses `model.tokenizer` with `truncation=False, add_special_tokens=True`.
   Ingest **asserts** `real_token_count(text) <= cfg.chunking.max_tokens` (250) for every chunk and
   fails the run listing the offending `chunk_id`s (architecture §5.3, R7). No silent truncation.
5. `get_collection`: `create=True` creates `mf_faqs` with `metadata={"hnsw:space": "cosine"}`;
   `create=False` fetches and raises `CorpusMissingError` when absent or empty.
6. `upsert_chunks` — always `upsert`, never `add`. `ids=[c.chunk_id]`, `documents=[c.text]`,
   `metadatas=[c.to_metadata() for c in chunks]`, `embeddings=vectors.tolist()`.
   `to_metadata()` must be **scalars only** (str/int/float/bool) — Chroma rejects nested values;
   `text` is not in the metadata (it is the `document`).
7. `prune`: fetch `collection.get(include=[])` ids, delete ids not in the current corpus, return the
   count. Prevents orphans when a source page is edited.
8. `corpus_hash` = `sha256` over `"\n".join(f"{sid}:{h}" for sid, h in sorted(source_hashes.items()))`.
9. `write_status` writes `data/status.json` atomically:
   `{corpus_hash, chunk_count, model, collection, last_ingest_at, source_hashes}`. Never include
   absolute machine paths or secrets.
10. `query_chunks` wraps `collection.query(..., include=["documents","metadatas","distances"])` and
    converts `score = 1 - distance`.
11. `run_ingest.py` stage wiring, in this order:
    `load config → status check → fetch (cache-aware) → clean → reject check → chunk → write
    artefacts → embed → upsert → prune → write status → print report`.
    Status gate: if `status.json` exists **and** `corpus_hash` matches **and** `--force` not passed,
    print "corpus unchanged, skipping embed+store" and exit 0 (A12).
    `--sources s1,s2` restricts the run; with a subset, skip the status gate and skip pruning.
    `--dry-run` stops after chunk artefacts.
12. Report gains columns: `chunks`, `topics`, `vectors`, `pruned`, `corpus_hash[:8]`.

### 7.4 Rules

* No network at embed/store time. Verify with
  `python -c "import os; os.environ['HF_HUB_OFFLINE']='1'; import run_ingest"` style checks and by
  running the second ingest with the machine offline-capable.
* Never send chunk text to any service. `chromadb` is local; no cloud embedding provider may be
  introduced (FR-4.5).
* Pin `chromadb` exactly; do not use a client API that changed signature between versions — code
  against the pinned version and note the version in `README.md` (R10).

### 7.5 Gate

```
python run_ingest.py
python run_ingest.py
python -c "import json,chromadb,pathlib; c=chromadb.PersistentClient(path='data/chroma'); print(c.get_collection('mf_faqs').count())"
Get-Content data\status.json
```
Expected: first run embeds and upserts; second run prints "corpus unchanged, skipping embed+store";
`count()` is identical before and after the second run; `status.json` has exactly the 6 keys.

### 7.6 Tests (`tests/test_ingest_idempotency.py`)

| Case | Assertion |
|---|---|
| two runs → same state | chunk count, set of `chunk_id`s and `count()` identical |
| upsert not duplicate | a changed chunk's old `chunk_id` is deleted by `prune` |
| status gate | matching `corpus_hash` + no `--force` → no embed call (monkeypatched counter) |
| force | `--force` re-embeds |
| metadata shape | every value in `to_metadata()` is `str|int|float|bool` |
| real token bound | `real_token_count(chunk.text) <= 250` for all chunks (slow marker) |
| vector shape | `embed_documents` returns `(n, 384)`, `float32`, unit-norm rows |
| no network | `requests` and `httpx` are blocked by the fixture for the whole run |

> **Cursor prompt — Phase 4**
> Implement Phase 4 of `Docs/implementation.md` (§7 only): `ingest/embedder.py`,
> `ingest/vectorstore.py`, and the embed/store/status stages in `run_ingest.py`, exactly per
> §7.2–§7.4. Reuse the single embedder for both paths; never `add()` — always `upsert` by
> `chunk_id`; always `prune`; write `data/status.json` atomically with only the 6 documented keys.
> Run the §7.5 gate twice, then `python -m pytest -q tests/test_ingest_idempotency.py`. Report the
> chunk count, vector count, and confirm the second run made no writes.

---

## 8. Phase 5 — M5 Retrieval (`app/retriever.py`)

**Goal.** Filter → dense → hybrid fusion → scheme affinity → dedupe → `min_score` floor, with no
LLM anywhere in this phase.

### 8.1 Files

| Path | Action | Notes |
|---|---|---|
| `app/retriever.py` | create | |
| `tests/test_retriever.py` | create | offline; uses the real corpus if present, else skips |

### 8.2 Contract

```python
def resolve_scheme(question: str, cfg: AppConfig) -> str|None       # scheme_name or None
def resolve_topic(question: str, cfg: AppConfig) -> str|None        # topic or None
def retrieve(question: str, decision: Decision, cfg: AppConfig) -> RetrievalResult
def build_where(topic: str|None, scheme: str|None) -> dict|None    # Chroma where clause
def fuse(dense: Sequence[RetrievalHit], lexical_ranks: Mapping[str,int], cfg: AppConfig) -> list[RetrievalHit]
```

### 8.3 Ordered tasks

1. `resolve_scheme`: match the question against the aliases of each `SourceSpec.scheme_name`
   (lowercase; also match a shortened form such as `"hdfc flexi cap"`, `"flexi cap"`, `"hdfc large
   cap"` derived by stripping the plan/option words). Longest alias wins. `"parag parflex"` matches
   no spec → `None`.
2. `resolve_topic`: reuse `config.yaml → topics` with the same word-boundary matcher as
   `ingest/topics.py` (import it; do not duplicate the logic).
3. `build_where`: when a topic is resolved, first pass uses
   `{"$and": [{"$or": [{"topic": t}, {"topic": "general"}]}, {"scheme_name": scheme}]}` when a
   scheme is known, dropping the scheme clause when it is not. `where=None` when neither resolves.
4. Dense pass: `n_results = max(top_k * candidate_multiplier, 20)`, `score = 1 - distance`.
5. Lexical pass: tokenise the question (lowercase, `[a-z0-9]+`), keep rare tokens = those with
   length `>= 3` that are not in a small stoplist (`the, and, for, what, is, of, my, in, a, to, how,
   do, i, my, fund, funds`), plus every token that appears in a scheme alias. Score each candidate
   by exact token overlap; rank by score. `lexical_ranks[chunk_id] = rank` (1-based) for any chunk
   with `score > 0`; `None` otherwise.
6. `fuse`: `fused = 1/(rrf_k + dense_rank) + lexical_weight * (1/(rrf_k + lexical_rank))` for chunks
   that appear in either ranking; chunks in neither are dropped. `rrf_k = 60`, `lexical_weight = 0.6`.
7. Scheme affinity: if a scheme resolved, chunks with a **conflicting** `scheme_name` are dropped;
   if that would empty the set, keep them and record it in the `RetrievalResult` (log line only).
   For general (non-scheme) sources, no affinity filtering.
8. Dedupe (R2): collapse hits whose `normalised_text` is identical, keeping the one with the newest
   `content_as_of` (empty dates sort oldest), then the highest fused score as tie-break.
9. Threshold (FR-5.3): take `top_k`. If there are no hits, or `best.fused_score < min_score`, or the
   resolved scheme has no hits at all → `RetrievalResult(status="out_of_corpus", hits=())`.
10. **Filter relaxation (A8):** if the filtered pass returned `< top_k` hits, re-run the dense
    query with `where=None` and re-fuse, keeping the filter only if it is the better result set
    (more hits above `min_score`). Log `filtered_topic` actually used, and `"relaxed"` when the
    fallback was used.
11. **No LLM import** in this module, and no `groq` anywhere under `app/` until Phase 6.

### 8.4 Rules

* Retrieval must be fully offline and deterministic. Same corpus + same question → identical hit
  list, ordering included.
* `min_score` applies to the **fused** score, and the RRF formula produces scores well below 1 —
  so `min_score = 0.28` in `config.yaml` is a floor on the *fused* value and must be re-tuned
  against the golden set in Phase 5's exit review. Record the chosen value and the top-5 fused
  scores for all 10 golden questions in your report.
* Do not lower `min_score` below the point where Q9 ("Parag Parflex expense ratio") returns hits.
  Out-of-corpus detection must survive: at minimum, a `resolved_scheme is None and
  lexical_ranks empty` case combined with a low best score returns `out_of_corpus`.

### 8.5 Gate

```
python -c "from config import load_config; from app.retriever import retrieve; from app.guard_stub import FACTUAL; ..."
```
Use this throwaway driver instead (no `guard.py` exists yet — Phase 7):

```python
# scratch/try_retrieval.py
from config import load_config
from app.retriever import retrieve
from app.models import Decision
cfg = load_config()
for q in ["What is the expense ratio of HDFC Large Cap Fund - Direct Growth?",
          "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
          "What is the expense ratio of Parag Parflex Fund?",
          "Who won the 2024 cricket world cup?"]:
    r = retrieve(q, Decision("FACTUAL", None, "manual"), cfg)
    print(q, "->", r.status, [(h.chunk.scheme_name, h.chunk.topic, round(h.score, 4)) for h in r.hits])
```

Expected: the two in-scope questions return hits from the **correct** scheme with a single `topic`
each; the last two return `out_of_corpus` with zero hits.

### 8.6 Tests (`tests/test_retriever.py`)

| Case | Assertion |
|---|---|
| right scheme in top-k | for each of the 5 schemes, a question naming it returns that scheme in the top-k |
| no scheme confusion | zero hits from a different scheme when a scheme resolved |
| out-of-corpus floor | "Parag Parflex expense ratio" and an off-topic question → `status == "out_of_corpus"`, `hits == ()` |
| topic filter | an expense-ratio question's top hits include at least one `topic == "expense_ratio"` chunk |
| filter relaxation | a question whose topic is mis-tagged still returns the correct chunk (A8) |
| dedupe | two identical-text chunks with different `content_as_of` → one hit, the newer date |
| determinism | two `retrieve` calls return identical `[(chunk_id, score)]` |
| lexical boost | a question containing `"80C"` and a scheme name ranks that scheme's `lockin_80c` chunk first |

> **Cursor prompt — Phase 5**
> Implement Phase 5 of `Docs/implementation.md` (§8 only): `app/retriever.py` exactly per §8.2–§8.4,
> in the order of §8.3. Do not import `groq` or Streamlit, and do not call any LLM. Tune
> `retrieval.min_score` against the 10 golden questions in `data/golden_questions.jsonl` and report
> the value you chose plus the top-5 fused scores for each of the 10 questions, then confirm that
> Parag Parflex and an off-topic question both return `out_of_corpus`. Run
> `python -m pytest -q tests/test_retriever.py` and paste the scratch driver output.

---

## 9. Phase 6 — M6 Generation + Validation (`app/prompts.py`, `app/validators.py`, `app/generator.py`, `app/cache.py`)

**Goal.** ≤3-sentence answers grounded in retrieved chunks, with two independent post-LLM gates
and one terminal fallback.

### 9.1 Files

| Path | Action | Notes |
|---|---|---|
| `app/prompts.py` | create | system prompt + context blocks |
| `app/validators.py` | create | **pure functions, no LLM** — citation, numbers, sentences, as-of, PII |
| `app/cache.py` | create | per-question + chunk-set cache |
| `app/generator.py` | create | Groq call, retry, gate, fallback |
| `tests/test_validators.py` | create | fully offline, no key |

### 9.2 `app/prompts.py` contract

```python
SYSTEM_PROMPT: str                     # frozen, see §9.3
def build_context_blocks(hits: Sequence[RetrievalHit]) -> str
def build_user_prompt(question: str, hits: Sequence[RetrievalHit]) -> str
def citation_allowlist(hits: Sequence[RetrievalHit], cfg: AppConfig) -> set[str]
```

Context block format (verbatim text only, no paraphrase, no extra facts):

```
[i] source_url: <url>
    doc_type: <doc_type> | publisher: <publisher> | content_as_of: <YYYY-MM-DD or "unknown">
    <verbatim chunk text>
```

If every hit has `content_as_of == ""`, render `content_as_of: unknown` and instruct the model to
omit the "Last updated" line rather than guess a date.

### 9.3 Frozen system prompt (PRD §6.5)

```
You are a facts-only mutual fund FAQ assistant for a fixed set of HDFC Mutual Fund schemes.
Answer using only the context provided. If the context does not contain the answer, reply
exactly: "I don't have that in my sources." Never give advice, recommendations, opinions,
comparisons or suitability judgements. Never state or compute returns, performance, rankings or
NAV history - if asked, refuse and point to the official factsheet. Never request or repeat
personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). Keep the answer to at
most 3 sentences. End with exactly one line "Source: <url>". End with
"Last updated from sources: <content_as_of>".
```

Appended at runtime (also in `prompts.py`, not hand-edited per call site):
"Use only these source URLs: `<sorted allowlist>`. The `content_as_of` value you print must be
copied from the cited chunk. If the cited chunk has no date, print
`Last updated from sources: unknown`."

### 9.4 `app/validators.py` contract (all pure; return `list[Violation]`)

```python
@dataclass(frozen=True) class Violation: code, detail, value

def split_answer_lines(text: str) -> tuple[str, str, str]   # (body, source_line, asof_line)
def check_citation(text: str, allowed: set[str]) -> list[Violation]
def check_sentence_count(body: str, max_sentences: int) -> list[Violation]
def check_word_count(body: str, max_words: int) -> list[Violation]
def check_numbers(body: str, retrieved_text: str, question: str) -> list[Violation]
def check_pii(text: str, patterns: Sequence[str]) -> list[Violation]
def check_performance(text: str) -> list[Violation]
def check_asof_line(asof_line: str, hits: Sequence[RetrievalHit]) -> list[Violation]
def validate_answer(text: str, hits: Sequence[RetrievalHit], question: str, cfg: AppConfig) -> list[Violation]
```

### 9.5 Validator rules (implement precisely)

1. `split_answer_lines`: a line matching `^\s*Source:\s*(\S+)` is the source line; a line matching
   `^\s*Last updated from sources:\s*(.+)$` is the as-of line; everything else is the body. A body
   line that starts with `Source:` inside a sentence is **not** extracted — the anchor is required.
2. `check_citation`: normalise the URL (strip markdown `[]()`, trailing `.`/`,`/`)`, lowercase the
   host, drop `www.`, drop the trailing `/`). It must be in `allowed` (retrieved chunk URLs
   ∩ `config.yaml` source URLs). Violation codes: `citation_missing`, `citation_not_retrieved`,
   `citation_not_in_allowlist`, `citation_multiple`.
3. `check_sentence_count`: on the **body only**; first mask decimals (`\d\.\d` → `0.0`) and
   common abbreviations (`e.g.`, `i.e.`, `Rs.`, `vs.`, `p.a.`), then split on `(?<=[.!?])\s+` and
   count non-empty segments. `> max_sentences` → violation `too_many_sentences`.
4. `check_word_count`: whitespace tokens in the body. `> max_words` → violation `too_long`
   (reported, and treated as **suppressing** only when combined with `too_many_sentences`; see §9.6).
5. `check_numbers`: extract every numeric literal from the body with
   `re.findall(r"\d[\d,]*(?:\.\d+)?%?", body)`. Allowed set = the same regex applied to
   `retrieved_text + " " + question`, plus the spelled-out numbers `one..twelve`, plus `80C`, plus
   any single digit `0-9`. Compare numerically after stripping thousands separators (`1,00,000` and
   `100000` are equal) and treating a trailing `%` as decoration. Unmatched literal → violation
   `unsourced_number` with the literal in `detail` (NFR-9).
6. `check_pii`: the `config.yaml → refusal.pii_patterns` regexes, `re.search`, case-insensitive.
7. `check_performance`: the `refusal.performance` keyword list plus
   `\b\d+(\.\d+)?\s*%\s*(return|cagr|growth|gain)` and `\b(cagr|xirr|ytd|p\.?a\.?|irr)\b` and
   `1[- ]?year return`. Note: expense ratio / exit load / min SIP answers legitimately contain
   percentages, so **only return-context percentages** are violations.
8. `check_asof_line`: the as-of value must equal one of the cited chunk's `content_as_of` values, or
   be `unknown` when all cited chunks have no date. Any other value → `asof_mismatch`.
9. `validate_answer` runs 2–8 and returns the concatenated violations in that order.

### 9.6 Generator behaviour

```python
def answer(question: str, decision: Decision, retrieval: RetrievalResult, cfg: AppConfig) -> Answer
```

1. If `decision.intent != "FACTUAL"` → return the refusal `Answer` built from `app/guard.py`
   constants (Phase 7 supplies them; until then, raise a clear `NotImplementedError`-free stub
   message — no, better: Phase 6 ships `REFUSAL_MESSAGES` in `app/guard.py` as *constants only*;
   Phase 7 adds `classify()`).
2. If `retrieval.status == "out_of_corpus"` → `Answer(text="I don't have that in my sources.", status="out_of_corpus")`,
   **no LLM call**.
3. Cache lookup (`app/cache.py`): key = `sha256(f"{model}|{normalised_question}|{sorted_chunk_ids}")`.
   In-memory dict + optional JSONL at `paths.cache_dir/answers.jsonl` (git-ignored). Cache stores
   only validated answers; a suppressed result is **not** cached (so a fix is picked up).
4. Call `groq.Groq(api_key=get_groq_api_key())` with
   `model=get_groq_model(), temperature=0.0, max_tokens=300, seed=42` where the pinned SDK accepts
   `seed`; on `TypeError`/`BadRequest` for `seed`, retry once without it and log once.
5. Missing key → raise `MissingApiKeyError` with the message
   `"GROQ_API_KEY is not set. Copy .env.example to .env and add your key."` — no stack trace to the
   user (the UI catches this; `evaluate.py` prints it and exits 2).
6. HTTP `429`/`5xx` → one retry after `2 s * attempt` + jitter, then `SAFE_FALLBACK`.
7. Citation failure (`citation_missing` / `citation_not_retrieved` / `citation_not_in_allowlist` /
   `citation_multiple`) → **one** retry with the failure reason appended to the user message
   ("Your previous answer cited <x>, which is not in the provided context. Cite one of these URLs:
   ..."). Still invalid → `SAFE_FALLBACK`.
8. Any other violation (`unsourced_number`, `pii`, `performance`, `too_many_sentences`, `asof_mismatch`)
   → `SAFE_FALLBACK` immediately, **no retry** (terminates in one step, ADR A3). A violation that is
   both `too_long` and `too_many_sentences` suppresses; `too_long` alone is reported to the eval
   report but does not suppress (G5 is a 95% gate).
9. `SAFE_FALLBACK` is one module constant:
   `"I can't give a verified answer from my sources. Please refer to the scheme's official
   factsheet and the investor-education resources linked below."` + the `advice` educational link.
   `status="suppressed"`, `source_url=""`, `content_as_of=""`.
10. On success build `Answer(text=<body>, source_url, doc_type, publisher, content_as_of=<cited
    chunk's>, used_chunk_ids=tuple(h.chunk.chunk_id for h in hits[:top_k]), status="ok",
    was_cached=<flag>)`. `text` is the **full model text including the Source and as-of lines**;
    `Answer.render()` re-emits them in canonical order so the UI never double-prints.
11. Log one structured line per call: `intent`, `n_hits`, `latency_ms`, `prompt_tokens`,
    `completion_tokens`, `violations` (codes only), `status`. Never log the question or the answer
    text.

### 9.7 Gate

Needs `GROQ_API_KEY` in `.env` (if absent, run the offline half of the gate and report the blocker):

```
python -c "import json;from config import load_config;from app.guard import classify;from app.retriever import retrieve;from app.generator import answer;q=json.loads(open('data/golden_questions.jsonl',encoding='utf-8').readline());d=classify(q['question']);r=retrieve(q['question'],d,load_config());a=answer(q['question'],d,r,load_config());print(a.render())"
```
Expected: a ≤3-sentence body, exactly one `Source:` line whose URL is in the allowlist, and
`Last updated from sources:` equal to the cited chunk's date.

### 9.8 Tests (`tests/test_validators.py`, no key)

| Case | Assertion |
|---|---|
| citation in allowlist + retrieved | no violation |
| citation from a non-retrieved allowlist URL | `citation_not_retrieved` |
| citation not in allowlist at all | `citation_not_in_allowlist` |
| two `Source:` lines | `citation_multiple` |
| missing `Source:` line | `citation_missing` |
| number present in retrieved text | no violation (test `1.5%`, `₹500`, `1,00,000`) |
| number absent from retrieved text | `unsourced_number` naming the literal |
| expense ratio answer with `%` | **not** flagged (return-context only) |
| return claim "returned 32% last year" | flagged `performance` |
| PAN / email / phone in output | flagged `pii` |
| sentence count | 4-sentence body → `too_many_sentences`; 3 → clean |
| as-of line mismatch | `asof_mismatch`; matching date → clean |

> **Cursor prompt — Phase 6**
> Implement Phase 6 of `Docs/implementation.md` (§9 only). Write `app/validators.py` first as pure
> functions and get `tests/test_validators.py` green **before** writing the Groq call — the
> validators are the hard gate and must be testable with no API key. Then `app/prompts.py`
> (system prompt frozen exactly as in §9.3), `app/cache.py`, `app/generator.py` per §9.6, plus the
> `REFUSAL_MESSAGES` constants in `app/guard.py` (constants only; no `classify()` yet). Do not
> implement the UI. Paste the rendered answer for golden Q1 and the structured log line.

---

## 10. Phase 7 — M7 Guardrails (`app/guard.py`)

**Goal.** Deterministic pre-LLM classification with 8/8 refusals and 0 false accepts, offline.

### 10.1 Files

| Path | Action | Notes |
|---|---|---|
| `app/guard.py` | create | `classify()` + refusal constants |
| `tests/test_guard.py` | create | the G4 gate |

### 10.2 Contract

```python
@dataclass(frozen=True) class Decision: intent, educational_url, reason_code     # already in models.py

REFUSAL_MESSAGES: dict[str, str]          # ADVICE / PERFORMANCE / PII / OUT_OF_CORPUS
SAFE_ADVICE_TEXT / SAFE_PERFORMANCE_TEXT / SAFE_PII_TEXT / SAFE_OUT_OF_CORPUS_TEXT

def classify(question: str, cfg: AppConfig) -> Decision
def scan_pii(question: str, patterns: Sequence[str]) -> bool
def normalise_question(question: str) -> str        # lowercase, punctuation->space, collapse ws
def refusal_answer(decision: Decision) -> Answer    # status=refused_<intent_lower>
```

### 10.3 Ordered evaluation (order is significant — architecture §6.1)

1. `scan_pii` → `PII`. On a hit, the **question body is dropped immediately**; no other rule runs
   and no log record contains the text. `reason_code = "pii:<matched_pattern_index>"` (the index,
   not the match) so logs stay non-identifying.
2. Advice phrases → `ADVICE`, `reason_code="advice:<phrase>"`.
3. Performance phrases → `PERFORMANCE`, `reason_code="performance:<phrase>"`.
4. `out_of_corpus.known_foreign_schemes` phrase present **and** no allowlisted scheme alias present →
   `OUT_OF_CORPUS`, `reason_code="out_of_corpus:<phrase>"`.
5. Otherwise → `FACTUAL`, `reason_code="factual"`, `educational_url=None`.

Precedence `PII > ADVICE > PERFORMANCE > OUT_OF_CORPUS > FACTUAL`. Phrase matching uses
`normalise_question` + word-boundary-aware substring match for short phrases.

### 10.4 Refusal messages (frozen; also used by README and `Docs/sample_qa.md`)

| Intent | Message | Link |
|---|---|---|
| `ADVICE` | "I answer only published facts about these schemes and can't recommend, compare or rate funds or suggest a portfolio." | `refusal.educational_links.advice` |
| `PERFORMANCE` | "I don't state or compute returns. Performance figures for each scheme are in its official factsheet, linked below." | `refusal.educational_links.performance` |
| `PII` | "Please don't share personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). This message has not been stored. Delete it from your chat." | `refusal.educational_links.pii` |
| `OUT_OF_CORPUS` | "I don't have that in my sources." + "I answer only about: HDFC Large Cap, HDFC Equity (Flexi Cap), HDFC ELSS Tax Saver, HDFC Small Cap and HDFC Balanced Advantage funds, and general official SEBI/AMFI information." | none |

Every refusal `Answer` sets `status` to `refused_advice` / `refused_performance` / `refused_pii` /
`out_of_corpus`, `source_url=""`, `content_as_of=""`, `used_chunk_ids=()`, and renders
`"{message}\n\n{link}"` when a link exists.

### 10.5 Rules

* The LLM is never asked to adjudicate refusal (A1). `generator.answer` consumes
  `decision.intent` and never re-classifies.
* No logging of the question, ever — including for non-PII intents. Log `intent` + `reason_code`.
* Keyword lists live in `config.yaml`, not in code (FR-7.2, FR-10.3). `guard.py` may add *regex*
  rules in code only for PII patterns that are structurally impossible to express as keywords, and
  must list them in the module docstring.

### 10.6 Gate

```
python -m pytest -q tests/test_guard.py
```
Expected: 8/8 refusals on the PRD §10.4 set, 10/10 `FACTUAL` on the golden set, 0 false accepts.

### 10.7 Tests (`tests/test_guard.py`)

| Case | Assertion |
|---|---|
| the 8 PRD §10.4 questions | every one is non-`FACTUAL` (parametrised, one assert per question so failures name the question) |
| the 10 golden questions | every one is `FACTUAL` |
| PII variants (12+) | PAN, spaced Aadhaar, 10-digit phone, `+91` phone, email, "my IFSC is HDFC0001234", "OTP 482913", "account number 0012345678", "CVV", "Demat" |
| precedence | "Should I buy X? My PAN is ABCDE1234F" → `PII` |
| advice+performance | "Which is best and how much did it return?" → `ADVICE` |
| out-of-corpus | "What is the expense ratio of Parag Parflex Fund?" → `OUT_OF_CORPUS` |
| in-scheme not flagged | "What is the expense ratio of HDFC Large Cap Fund?" → `FACTUAL` |
| short-phrase boundary | "after the merger" must not match the `ter` rule; "navigator" must not match `nav` |
| refusal payload | each refusal `Answer` has a non-empty message, the right `status`, and a link for ADVICE/PERFORMANCE/PII |
| no logging | assert the `logging` module records for a PII question contain no substring of the question |

> **Cursor prompt — Phase 7**
> Implement Phase 7 of `Docs/implementation.md` (§10 only): `app/guard.py` per §10.2–§10.5, with the
> exact evaluation order PII → ADVICE → PERFORMANCE → OUT_OF_CORPUS → FACTUAL and the frozen
> messages in §10.4. Keep the keyword/pattern data in `config.yaml`. Do not touch the UI. Then run
> `python -m pytest -q tests/test_guard.py` and paste the result, naming any test that fails and the
> exact reason — do not weaken a keyword to make a test pass; add the case to the report instead.

---

## 11. Phase 8 — M8 UI (`app/pipeline.py`, `app/disclaimer.py` already frozen, `app/ui.py`)

**Goal.** The tiny Streamlit UI, with all logic in one orchestrator so `evaluate.py` and the UI
share one code path (A9, NFR-8).

### 11.1 Files

| Path | Action | Notes |
|---|---|---|
| `app/pipeline.py` | create | the single orchestrator |
| `app/ui.py` | create | presentation only |
| `tests/test_pipeline.py` | create | headless, LLM stubbed |

### 11.2 `app/pipeline.py` contract

```python
@dataclass(frozen=True) class PipelineDeps:   # injection seam for tests/evaluate
    retriever: Callable[[str, Decision, AppConfig], RetrievalResult]
    generator: Callable[[str, Decision, RetrievalResult, AppConfig], Answer]
    classifier: Callable[[str, AppConfig], Decision]

DEFAULT_DEPS: PipelineDeps

def answer_question(question: str, cfg: AppConfig, deps: PipelineDeps = DEFAULT_DEPS) -> Answer
def corpus_is_ready(cfg: AppConfig) -> bool          # status.json present + collection non-empty
```

`answer_question` order: `classify → (refuse if not FACTUAL) → retrieve → (out_of_corpus if so) →
generate`. It catches `MissingApiKeyError` and returns an `Answer` with `status="suppressed"` and
that message so both the UI and `evaluate.py` render it identically.

### 11.3 `app/ui.py` element contract (architecture §6.4)

| Element | Requirement |
|---|---|
| `@st.cache_resource` | `load_config()`, embedder model, Chroma collection — one load per process, not per rerun |
| Startup checks | missing corpus → `st.warning("Run: python run_ingest.py")` + `st.stop()`; missing key → `st.error(...)` + `st.stop()` |
| Header | welcome line naming the AMC, the 5 schemes, and that answers are facts-only |
| Examples | exactly 3 fixed buttons: (1) "What is the expense ratio of HDFC Large Cap Fund – Direct Growth?" (2) "What is the lock-in period in HDFC ELSS Tax Saver Fund?" (3) "How do I download my capital gains statement?" — clicking sets the input value (FR-8.4) |
| Persistent note | `SHORT_NOTE` visible above the input, on every rerun (FR-8.1) |
| Input | a single `st.text_input(label="Ask a factual question about these schemes", max_chars=300)`, placeholder, and a submit button. No other fields, no login (FR-8.5) |
| Answer block | `answer.render()` split so the `Source:` line renders as a markdown link `[Source: url](url)`, plus `f"Source document type: {doc_type}"` and `f"Last updated from sources: {content_as_of}"` |
| Retrieved context | `st.expander("Retrieved context", expanded=False)`; per hit: rank, `f"score {score:.3f}"`, URL, `section_heading`, `content_as_of`, and the chunk text in a fenced block |
| Refusal block | message + link, **no** context panel |
| Freshness | if `content_as_of` is older than `freshness.warn_after_days` (90), show a one-line `st.info` warning — do not block (PRD §14 Q3 default) |
| Footer | `DISCLAIMER` in `st.caption`, verbatim from `app/disclaimer.py` |
| Logging | the UI logs `intent`/`reason_code`/`latency_ms` only |

Also: use `st.session_state` for the last `Answer` + `RetrievalResult` so the context panel
survives a rerun; never store the raw question in session state beyond the current run (it is
in-memory only, never written to disk).

### 11.4 Rules

* `app/ui.py` must not import `chromadb`, `groq`, `requests`, or any `ingest/` module other than
  through `app/pipeline.py`/`app/retriever.py`. It must contain no regex, no threshold and no
  prompt text.
* No `st.write` of raw HTML from fetched pages.

### 11.5 Gate

```
streamlit run app/ui.py
```
Manual checklist: welcome line; exactly 3 examples; `SHORT_NOTE` visible; a factual question returns
a ≤3-sentence answer with one working `Source:` link, document type and date; the "Retrieved context"
expander lists 5 chunks with scores; an advice question returns the refusal with the educational
link and **no** context panel; the disclaimer is in the footer; a PII question returns the PII
refusal; deleting `data/chroma` produces the "Run: python run_ingest.py" warning.

### 11.6 Tests (`tests/test_pipeline.py`)

| Case | Assertion |
|---|---|
| refusal short-circuits | with a stubbed generator that raises if called, an advice question still returns a refusal `Answer` |
| out-of-corpus short-circuits | stub generator raises if called; `status == "out_of_corpus"` |
| happy path order | stub call log is exactly `["classify", "retrieve", "generate"]` |
| missing key | `MissingApiKeyError` from the stub → `Answer` with the setup message and `status="suppressed"` |
| corpus readiness | `corpus_is_ready` false when `status.json` is absent |

> **Cursor prompt — Phase 8**
> Implement Phase 8 of `Docs/implementation.md` (§11 only). Write `app/pipeline.py` first with the
> dependency-injection seam, then `app/ui.py` as pure presentation per the §11.3 table. Enforce the
> import ban in §11.4 — a grep for `groq|chromadb|requests` in `app/ui.py` must return nothing.
> Run `python -m pytest -q tests/test_pipeline.py`, then launch `streamlit run app/ui.py` and walk
> the §11.5 checklist, reporting the result of each item. If `GROQ_API_KEY` is missing, report that
> instead of inventing a key.

---

## 12. Phase 9 — M9 Evaluation (`evaluate.py`)

**Goal.** A single command that scores the golden set and the refusal set and writes
`reports/eval_report.md` against the PRD §10.1/§10.2 gates.

### 12.1 Files

| Path | Action | Notes |
|---|---|---|
| `evaluate.py` | create | CLI |
| `data/golden_questions.jsonl` | verify | created in Phase 3; must stay byte-stable |
| `reports/eval_report.md` | generated | D7 |
| `tests/test_eval_harness.py` | create | runs the report writer with a stubbed pipeline |

### 12.2 Contract

```python
def load_questions(path: Path) -> list[dict]
def score_answer(q: dict, answer: Answer, hits: Sequence[RetrievalHit], cfg: AppConfig) -> dict
def run_suite(cfg: AppConfig, *, use_cache: bool = True) -> SuiteResult
def render_report(result: SuiteResult) -> str
```

CLI: `python evaluate.py [--no-cache] [--questions data/golden_questions.jsonl] [--out
reports/eval_report.md] [--json]`. Exit code `0` when every hard gate passes, `1` when a gate
fails, `2` when the run could not start (missing key, missing corpus).

### 12.3 Per-question checks (one row per question, pass/fail for each)

| Check | Rule |
|---|---|
| `answer_present` | `status == "ok"` and non-empty body (for refusal rows: the expected refusal status) |
| `sentences_le_3` | body sentence count `<= 3` |
| `words_le_60` | body word count `<= 60` |
| `citation_valid` | exactly one `Source:` line, URL in retrieved ∩ allowlist |
| `citation_matches_chunk` | the cited URL equals the `source_url` of at least one used chunk (G3) |
| `no_advice` | guard did not classify `ADVICE`; body has no recommendation verbs |
| `no_performance` | `check_performance` clean; body has no `return/CAGR/XIRR` in return context |
| `no_pii` | `check_pii` clean |
| `numbers_traceable` | `check_numbers` clean against the retrieved text (NFR-9) |
| `asof_line` | `Last updated from sources:` present and equal to the cited chunk's date (or `unknown`) |
| `topic_match` | at least one used chunk's `topic == q["expect_topic"]` (where `expect_topic` is set) |
| `expected_refusal` | for refusal rows, the observed `status` equals the expected `refused_*`/`out_of_corpus` |
| `no_scheme_confusion` | every used chunk's `scheme_name` is either the question's scheme or empty (G: 0 confusion) |

### 12.4 Gates written into the report header (all hard)

```
G1  correctness            >= 8/10 golden answered, judged by rule (answer present + topic match + citation valid + numbers traceable)
G2  citation present       100% of non-refusal answers have a valid citation URL
G3  citation matches chunk 100% (audited, not sampled)
G4  refusal correctness    8/8 refusals, 0 false accepts
G5  brevity                >= 95% of answers <= 3 sentences AND <= 60 words
G6  reproducibility        re-run with --no-cache gives the same pass/fail matrix
G7  inspectability         chunks_preview.txt exists and chunk count > 0
G8  transparency           100% of answers carry doc_type + content_as_of
NFR-9                      0 unsourced numbers
```

### 12.5 Report format (`reports/eval_report.md`)

1. Title, timestamp, model, `corpus_hash`, `chunk_count`, `min_score`, cache on/off.
2. Gate table: `gate | threshold | observed | PASS/FAIL`.
3. Golden-set table: one row per question with all 13 checks as PASS/FAIL cells.
4. Refusal-set table: question (truncated) | expected status | observed status | PASS/FAIL.
5. Tuning notes: per-question `top_hit_scheme | topic | fused score`, so Phase 5's `min_score` can
   be re-tuned with evidence.
6. Known limits and any suppressed answers with their violation codes (codes only, no text).
7. A copy of `SHORT_NOTE` + `DISCLAIMER`.

Never write the question text of a **PII-classified** row into the report (FR-7.6): render those
rows as `"<redacted: PII>"`.

### 12.6 Gate

```
python evaluate.py
Get-Content reports\eval_report.md -TotalCount 40
```
Expected: all gates PASS, exit code 0. Then `python evaluate.py --no-cache` and confirm G6.

### 12.7 Tests (`tests/test_eval_harness.py`)

| Case | Assertion |
|---|---|
| report writer | a synthetic `SuiteResult` with one failing gate renders `FAIL` and the gate table |
| redaction | a PII row's question is never present in the rendered report |
| refusal accounting | a 7/8 refusal result fails G4 |
| brevity math | 4/5 answers over-long fails G5 (95% threshold) |
| missing key | `evaluate.py` exits 2 with the setup message and writes no report |

> **Cursor prompt — Phase 9**
> Implement Phase 9 of `Docs/implementation.md` (§12 only). `data/golden_questions.jsonl` was
> created in Phase 3 — do not edit the 10 questions or the 8 PRD §10.4 refusal questions. Build the
> 13 per-question checks, the 9 hard gates, and the report format in §12.5 including the PII-row
> redaction. Run `python evaluate.py`, then `--no-cache`, and paste the gate table. Report any
> failing gate with the per-question evidence rather than relaxing a threshold.

---

## 13. Phase 10 — M10 Packaging

**Goal.** A reviewer can clone and run in under 15 minutes.

### 13.1 Files

| Path | Action | Notes |
|---|---|---|
| `README.md` | create | D3 |
| `Docs/sources.md` | create | human-readable twin of `config/sources.csv` (D2) |
| `Docs/sample_qa.md` | create | D4 — generated, not hand-written |
| `requirements.txt` | finalise | exact `==` pins |
| `.gitignore` | verify | per §3.1 |
| `docs/` → `Docs/` | verify | single capital-D folder |

### 13.2 `README.md` required sections (in order)

1. Title + one-paragraph what/who, and the `SHORT_NOTE` line.
2. Scope: AMC (HDFC MF), the 5 schemes in a table with links, and the out-of-scope list
   (advice, returns, live NAV, accounts, non-official sources) — PRD §2.1/§9.
3. Architecture at a glance: the two pipeline diagrams from architecture §3.1 (ASCII, as-is) and a
   module table.
4. `Docs/architecture.md`, `Docs/implementation.md`, `Docs/PRD.md` links.
5. **Setup:** `python -m venv .venv` (Windows **and** macOS/Linux commands), `pip install -r
   requirements.txt`, exact pinned Python and package versions, `Copy-Item .env.example .env`
   (`cp` on Unix) and where to get the Groq key.
6. **Run:** `python run_ingest.py` (first run ~2–3 min) → `streamlit run app/ui.py` →
   `python evaluate.py`, with the expected output of each.
7. `.env` table (name, meaning, default) and `.env.example`.
8. Configuration: which knobs live in `config/config.yaml` (allowlist, chunking, thresholds,
   refusal keywords) and the env-override precedence chain.
9. **Deliverables map:** D1–D8 with links to the artefacts in this repo.
10. **Evaluation results:** the gate table pasted from the latest `reports/eval_report.md`.
11. **Known limits:** non-JS-rendered corpus only; corpus is 5 pages; no live NAV; answers are
    as-of the `content_as_of` date; `min_score` tuned on a 10-question set.
12. `DISCLAIMER` verbatim.
13. Security note: no key in git, `.env` ignored, PII never logged, no telemetry.

### 13.3 `Docs/sample_qa.md` (generated by a small `export_sample_qa` function in `evaluate.py`)

5–10 entries. Each: the question, the assistant's answer via `Answer.render()`, the source link,
`doc_type`, `publisher`, `Last updated from sources:`. Include at least 2 refusals and 1
out-of-corpus, so the guard behaviour is documented. State at the top that it was generated by
`python evaluate.py` on `<date>` from `corpus_hash` `<hash>`. Mirror `SHORT_NOTE` and `DISCLAIMER`
at the bottom.

### 13.4 `Docs/sources.md`

One row per source: URL, publisher, document type, `content_as_of`, chunk count, retrieval date,
plus a short note on why Groww is an accepted mirror (PRD §14 Q6 default) and the
`non-collectable` links observed. Keep `config/sources.csv` as the machine-readable twin.

### 13.5 Final gate (the whole milestone, in order)

```
git init
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env       # then add the key
python -m pytest -q
python run_ingest.py
python run_ingest.py               # expect "corpus unchanged"
streamlit run app/ui.py           # §11.5 checklist
python evaluate.py                # all gates PASS, exit 0
python evaluate.py --no-cache     # G6 PASS
git status --short                # no .env, no data/raw, no data/chroma, no data/models, no data/cache
```

Additionally, a secret scan must find no `gsk_` string anywhere in the working tree, and
`data/chunks/chunks.jsonl` + `data/chunks/chunks_preview.txt` must be committed (they are the
inspectability deliverable D6) while `data/raw/` is not.

### 13.6 Milestone gate check

| Milestone | Exit signal (PRD §12) | Verified in |
|---|---|---|
| M0 | PRD + architecture reviewed; allowlist fixed | Phase 0 |
| M1 | all 5 URLs fetched, statuses logged | §4.5 |
| M2 | block structure preserved, no nav noise, no empty page | §5.5 |
| M3 | strategy validated on real text, full metadata, token bound | §6.7 |
| M4 | 384-dim vectors, idempotent re-run, zero API calls | §7.5 |
| M5 | right scheme in top-k, out-of-corpus filtered | §8.5 |
| M6 | ≤3 sentences, one valid citation, no unsourced numbers | §9.7, §9.8 |
| M7 | 0/8 false accepts | §10.6 |
| M8 | welcome, 3 examples, disclaimer, citation, as-of, context panel | §11.5 |
| M9 | all §10.1 gates pass | §12.6 |
| M10 | clone-and-run in < 15 min | §13.5 |

---

## 14. Traceability (phase → requirement → evidence)

| Phase | Requirements satisfied | Architecture § | Evidence artefact |
|---|---|---|---|
| 0 | FR-10.1–FR-10.3, NFR-4, NFR-8 | §2, §4, §7 | `config.py`, `config/config.yaml` |
| 1 | FR-1.1–FR-1.4, FR-11.1, R11, R12 | §5.1, §8.1, §8.3 | `data/raw/**`, fetch report |
| 2 | FR-2.1–FR-2.5, R1 | §5.2 | `data/chunks/blocks_*.txt`, `data/rejected.jsonl` |
| 3 | FR-3.1–FR-3.4, G7, R7, D2, D6, R2 | §5.3, §8.3 | `chunks.jsonl`, `chunks_preview.txt`, `config/sources.csv` |
| 4 | FR-4.1–FR-4.5, G6, NFR-2, NFR-3, R10 | §5.4, §5.5, A4, A12 | `data/status.json`, idempotency test |
| 5 | FR-5.1–FR-5.5, G4-adjacent, NFR-1, NFR-9 | §6.2, A7, A8 | retrieval report + test |
| 6 | FR-6.1–FR-6.5, G2, G3, G5, G8, R5, R9, NFR-3 | §6.3, A2, A3, A10 | validator tests, log line |
| 7 | FR-7.1–FR-7.6, G4, NFR-5, R6, R8, A1 | §6.1 | `tests/test_guard.py` |
| 8 | FR-8.1–FR-8.5, FR-9.1, A9, A11, NFR-8 | §6.4, §6.5 | UI checklist, `tests/test_pipeline.py` |
| 9 | FR-11.2, G1–G8, NFR-9, D7 | §8.5 | `reports/eval_report.md` |
| 10 | FR-10.2, NFR-4, NFR-6, D3, D4, D5 | §4 | `README.md`, `Docs/sample_qa.md` |

---

## 15. Open decisions to confirm with the owner before the phase that needs them

| # | Question (PRD §14) | Default assumed by this guide | Blocks |
|---|---|---|---|
| Q1 | Supplementary URLs beyond the 5 scheme pages | none; allowlist is exactly 5. Adding one is a config-only change | Phase 0/1 |
| Q2 | Direct IDCW / Regular variants too | no; Direct Growth only | Phase 0 |
| Q3 | Freshness hard-fail on old sources | no; UI `st.info` warning at 90 days | Phase 8 |
| Q4 | UI framework | Streamlit behind `app/pipeline.py` | Phase 8 |
| Q5 | Demo hosting | local run + screen recording | Phase 10 |
| Q6 | Groww acceptable as a citation target | yes, as an accepted mirror; `publisher` + `doc_type` always shown | Phase 6 |

If the owner changes any of these, update §2.1 (or the phase section that owns the value) **first**,
then re-run that phase's gate — do not patch the code in a later phase.
