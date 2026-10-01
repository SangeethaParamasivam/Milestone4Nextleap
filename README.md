# Milestone 4 — Mutual Fund FAQ RAG Chatbot (Facts-Only Q&A)

A retrieval-augmented chatbot that answers questions about **five HDFC Mutual Fund schemes** in
2–3 sentences, each answer carrying the source URL, the document type and the date the source
states. A deterministic rules layer runs *before* the language model and refuses advice,
performance, comparison and PII questions outright, so the model is never asked to adjudicate
whether it should answer. Built for retail investors and support/content staff who need a
verifiable factual answer, not an opinion.

> **Facts-only. No investment advice.**

**Build status: Phases 0–8 of 11 complete** (see [Build status](#build-status)). The pipeline
runs end to end: fetch → clean → chunk → embed → store → guard → retrieve → generate → validate,
and it is now reachable through the Streamlit chat UI in `app/ui.py` (or the terminal with
`ask.py`). Still missing: the evaluation harness (Phase 9) and the final packaging (Phase 10).

---

## Scope

**AMC in scope: HDFC Mutual Fund.** Schemes are referenced through their public Groww scheme
pages, which mirror AMC-published data and link back to the official factsheet / KIM / SID.

| # | Scheme | Category | Plan / option | Source URL |
|---|---|---|---|---|
| S1 | HDFC Large Cap Fund – Direct Growth | Large Cap | Direct / Growth | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> |
| S2 | HDFC Equity (Flexi Cap) Fund – Direct Growth | Flexi Cap | Direct / Growth | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> |
| S3 | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | ELSS | Direct / Growth | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> |
| S4 | HDFC Small Cap Fund – Direct Growth | Small Cap | Direct / Growth | <https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth> |
| S5 | HDFC Balanced Advantage Fund – Direct Growth | Hybrid / Balanced Advantage | Direct / Growth | <https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth> |

> **Deviation from `Docs/PRD.md` §4.1:** the S3 URL published in the PRD
> (`.../hdfc-elss-tax-saver-fund-direct-growth`) returns **HTTP 404** on groww.in. The working
> slug is `.../hdfc-elss-tax-saver-fund-direct-plan-growth`, verified 200 / 452 KB on
> 2026-09-30 and now used in `config/config.yaml`. The PRD table should be corrected.

Allowed domains for fetching: `groww.in`, `hdfcmf.com`, `sebi.gov.in`, `amfiindia.com`,
`myamfi.in`. Any other domain found in a page is recorded as a non-collectable link and never
fetched (FR-1.4).

### Explicitly out of scope (hard constraints, PRD §2.1 / §9)

- Personalised investment advice, buy/sell/timing calls, portfolio or asset-allocation guidance.
- Return, performance, CAGR/XIRR, ranking or fund-comparison computation of any kind.
- Live or historical NAV, market data, sector calls.
- Multi-user accounts, authentication, payments, mobile apps.
- Non-official public sources (blogs, YouTube, forums, news) and any user PII collection.
- Fine-tuning or training a model — RAG only.

---

## Architecture at a glance

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
        │
        ▼
  [Chunk]   ingest/chunker.py     section-aware 900/150 char windows → Block objects
        │                       → data/chunks/chunks.jsonl + chunks_preview.txt
        │
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
        │
        ▼
  [Route]   app/retriever.py      optional metadata filter (doc_type/topic) from classification
        │
        ▼
  [Embed]   ingest/embedder.py    same model, same normaliser
        │
        ▼
  [Retrieve] app/retriever.py     hybrid: dense top-k  + exact rare-token boost + scheme affinity
        │                       min_score floor ⇒ OUT_OF_CORPUS, no LLM call
        │
        ▼
  [Generate] app/generator.py     Groq call, temp≈0, ≤3 sentences, exactly one Source line
        │
        ▼
  [Validate] app/generator.py     citation ∈ retrieved ∩ allowlist → retry ×1
        │                       number/performance/PII filter → suppress on hit
        │
        ▼
  [Present] app/ui.py             answer + Source + doc type + "Last updated from sources:"

OFFLINE  evaluate.py              golden set (10) + refusal set (8) → reports/eval_report.md
```

| Module | Layer | May depend on | Must never |
|---|---|---|---|
| `config.py` | Config | — | do I/O beyond reading `config.yaml` / `.env` |
| `ingest/*` | Ingest | `config`, `app/models.py` | import app behaviour, call the LLM, import Streamlit |
| `app/guard.py`, `app/retriever.py`, `app/generator.py`, `app/prompts.py` | App core | `config`, `ingest/embedder.py`, `ingest/vectorstore.py` | import Streamlit, read raw HTML |
| `app/pipeline.py`, `app/memory.py` | App core | app core, `config` | import Streamlit, re-classify a decision |
| `app/ui.py` | Presentation | `app/pipeline.py`, `app/disclaimer.py` | contain retrieval or generation logic |
| `evaluate.py` | Evaluation | app core only | bypass the guard or the validators |

Design documents: [`Docs/architecture.md`](Docs/architecture.md) ·
[`Docs/implementation.md`](Docs/implementation.md) · [`Docs/PRD.md`](Docs/PRD.md) ·
[`Docs/ProblemStatement.txt`](Docs/ProblemStatement.txt)

---

## Setup

**Interpreter: CPython 3.14.7 is what this repository was built and tested on.** The guide
(`Docs/implementation.md` §1.2) recommends 3.11; see [Known limits](#known-limits). Phases 0–8
all run on 3.14, including embedding, ChromaDB storage and the Streamlit UI, with the HuggingFace
cache falling back to non-symlink mode (a warning, not a failure).

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` now pins every approved dependency; `requirements.in` holds the same list
unpinned, per the PRD §6.2 policy.

| Package | Version (pinned) |
|---|---|
| Python | 3.14.7 |
| beautifulsoup4 | 4.15.0 |
| chromadb | 1.5.9 |
| groq | 1.7.0 |
| lxml | 6.1.3 |
| numpy | 2.5.3 |
| pytest | 9.1.1 |
| python-dotenv | 1.2.3 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| sentence-transformers | 6.1.0 |
| streamlit | 1.64.0 |
| torch | 2.14.0 |
| trafilatura | 2.2.0 |

### Groq API key

Only one outbound call needs a key: the generation call in Phase 6. Create a key at
<https://console.groq.com/keys> (free tier), then put it in `.env`:

```
GROQ_API_KEY=gsk_...
```

`.env` is git-ignored and `.env.example` is committed with empty values. The key is read only by
`config.get_groq_api_key()` via `os.getenv` and is never logged or written into any artefact.
Ingestion, cleaning, chunking, embedding, storage and retrieval all work with **no key and no
network**.

---

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 284 tests, all offline except the stubbed Groq client
streamlit run app/ui.py                      # the chat UI (needs GROQ_API_KEY and a built corpus)
.\.venv\Scripts\python.exe ask.py            # the same Q&A in the terminal
.\.venv\Scripts\python.exe run_ingest.py     # fetch the 5 allowlisted URLs, print the per-URL report
.\.venv\Scripts\python.exe run_ingest.py     # re-run: every row shows cache=True, exit 0
.\.venv\Scripts\python.exe evaluate.py       # NOT IMPLEMENTED (Phase 9)
```

### `app/ui.py` — the Streamlit chat UI

```
.\.venv\Scripts\python.exe -m streamlit run app/ui.py
```

(`.venv\Scripts\streamlit.exe run app/ui.py` is the same thing.) The page opens on
<http://localhost:8501>. It needs `GROQ_API_KEY` in `.env` and a built corpus; with neither, the
page shows `Run: python run_ingest.py` or the key-setup message and stops instead of failing.

- **Chat history.** Every answered question stays on the page as a user/assistant pair, newest at
  the bottom, until **Clear chat** in the sidebar empties both the transcript and the memory window.
- **Sources.** Each answer carries a `Sources` expander listing the retrieved chunks it was built
  from — rank, fused score, `chunk_id`, URL, section heading and the verbatim chunk text. A refusal
  has no sources panel, because nothing was retrieved.
- **Citation line.** The `Source:` URL renders as a working markdown link, with
  `Source document type:` and `Last updated from sources:` underneath (`unknown` when the cited
  chunk carries no date), and a one-line freshness notice for a source older than 90 days.
- **Conversation memory.** The session keeps the last `memory.max_messages` messages (default 10,
  i.e. five question/answer pairs) in `app/memory.py`, shown in the sidebar as
  `retrieval context window: last 10 messages`. A follow-up is rewritten to a standalone question
  before retrieval — the same mechanism as `ask.py` — and the guard always classifies the raw
  question first, so a refusal is never reached through a rewrite.
- **3 example buttons** fill the input; `Facts-only. No investment advice.` sits above the input on
  every rerun and the full disclaimer is in the footer. No login, no other fields.

The UI holds no retrieval or generation logic: it calls `app.pipeline.run_turn` and renders what
comes back. `app/ui.py` imports no `chromadb`, `groq`, `requests` or `re`, and that ban is asserted
by a test.

### `ask.py`

```
.\.venv\Scripts\python.exe ask.py "What is the expense ratio of HDFC Large Cap Fund?"
.\.venv\Scripts\python.exe ask.py --show-chunks --no-cache "What is the exit load on HDFC Small Cap Fund?"
.\.venv\Scripts\python.exe ask.py --quiet "What is the minimum SIP for HDFC Equity (Flexi Cap) Fund?"
```

Flags: `--config PATH`, `--show-chunks` (print the retrieved chunks and the answer provenance),
`--no-cache` (bypass the answer cache), `--no-memory` (forget the conversation, retrieve every
question as typed), `--show-rewrite` (print the standalone question a follow-up became), `--quiet`
(hide the per-call generator log line). With no
positional question it starts an interactive loop; `exit`, `quit` or `:q` leaves it. Exit code
`0` normally, `2` on a config error or a missing `GROQ_API_KEY`. Retrieval works with no key, so
refusals and out-of-corpus answers still work offline.

A refusal or an out-of-corpus answer is produced **without any LLM call**; only a question that
survives the guard and clears the scope check spends a Groq token.

#### Conversation memory

Both the CLI loop and the UI session keep the last `memory.max_messages` messages (default 10, so
five question/answer pairs) in `app/memory.py`, so a follow-up is retrieved as a standalone
question:

```
question> What is the expense ratio of the HDFC Large Cap Fund?
The expense ratio for the HDFC Large Cap Fund - Direct Growth is 1.03%.
question> what about its fund size?
[rewrite] what about its fund size? -> What is the fund size of the HDFC Large Cap Fund?
The fund size (AUM) of the HDFC Large Cap Fund - Direct Growth is ₹39,933.36 Cr.
```

`history` prints the kept window in the CLI, `reset` forgets it, and **Clear chat** in the UI
empties it. Three rules keep it honest: the guard
always sees the raw question, so a refusal is never reached through a rewrite; a rewrite is only
requested when the question carries a deictic trigger (`memory.deictic_triggers`) or is shorter
than `memory.min_rewrite_words` words, so an ordinary question costs no extra call; and any
failure or unusable reply falls back to the question as typed. The rewritten question is
re-classified, so a follow-up that resolves to an out-of-scope scheme is still refused.

`run_ingest.py` flags: `--config PATH`, `--force` (ignore the raw cache), `--sources s1,s2`
(restrict the run), `--dry-run`. Exit code `0` when every requested source is `ok`, `1` when any
source is blocked or errored, `2` on a config error.

Expected output of a completed `python run_ingest.py`:

```
source_id | http | cache | bytes  | sha256   | status | error
----------+------+-------+--------+----------+--------+------
s1        | 200  | True  | 454248 | 33f61711 | ok     | -
s2        | 200  | True  | 494560 | 607e14ee | ok     | -
s3        | 200  | True  | 452913 | c34e4182 | ok     | -
s4        | 200  | True  | 497240 | 1086a21d | ok     | -
s5        | 200  | True  | 815998 | d3c3c665 | ok     | -
totals: 5 requested, 5 ok, 0 blocked, 0 error, 0.0s elapsed
```

Artefacts written under `data/raw/` (git-ignored): `<domain>/<source_id>.html` (cached bytes),
`<domain>/<source_id>.meta.json` (`url, http_code, fetched_at, sha256, bytes, content_type`),
`<source_id>.noncollectable.json` (off-allowlist links seen on the page, recorded only).

A cold first run takes roughly 90 s against groww.in: the loader is deliberately polite
(≥ 2 s between requests to the same host, descriptive User-Agent, 3 attempts with exponential
backoff, max 3 redirects, all restricted to allowlisted hosts). It never attempts to bypass a
block — a 403 is reported, not retried.

---

## Configuration

`config/config.yaml` is the single source of truth and its key set is frozen: the source
allowlist, allowed domains, paths, chunking parameters, retrieval thresholds, embedding and
generation parameters, the ordered topic vocabulary, refusal keyword lists, the PII regex set,
out-of-corpus scheme names, and the freshness warning window. The key set is not extended
implicitly; a new source is a config-only change that must be flagged before it is made.

Precedence chain, applied once in `config.load_config()`:

```
real environment variable  >  .env file  >  config/config.yaml  >  code default
```

Overridable by environment variable:

| Name | Meaning | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for the generation call (Phase 6) | unset |
| `GROQ_MODEL` | Generation model id | `qwen/qwen3.8-27b` |
| `TOP_K` | Retrieved chunks per question | `5` |
| `MIN_SCORE` | Dense-cosine floor below which the answer is out-of-corpus | `0.28` |
| `CHUNK_SIZE` | Chunk character size | `900` |
| `CHUNK_OVERLAP` | Overlap between consecutive windows | `150` |
| `PERSIST_DIR` | ChromaDB persist directory | `data/chroma` |
| `RAW_DIR` | Raw HTML cache directory | `data/raw` |
| `EMBED_MODEL` | Sentence-transformers model id | `sentence-transformers/all-MiniLM-L6-v2` |

`config.load_config()` **raises** `ConfigError` if a source URL is not `https` or its host is not
in `allowed_domains`, so a typo cannot silently add an out-of-allowlist fetch to the corpus.

---

## Deliverables map (PRD §11)

| ID | Deliverable | Artefact in this repo | Status |
|---|---|---|---|
| D1 | Working prototype | `app/ui.py` (Streamlit chat UI), `ask.py` | **done** |
| D2 | Source list | `config/sources.csv` (Phase 3), `Docs/sources.md` (Phase 10) | not implemented |
| D3 | README | this file | **done** (pulled forward from Phase 10) |
| D4 | Sample Q&A | `Docs/sample_qa.md` (Phase 10) | not implemented |
| D5 | Disclaimer snippet | `app/disclaimer.py`, mirrored below | **done** |
| D6 | Chunk inspection artefact | `data/chunks/chunks_preview.txt` (Phase 3) | not implemented |
| D7 | Evaluation report | `reports/eval_report.md` (Phase 9) | not implemented |
| D8 | PRD | `Docs/PRD.md` | **done** |

## Evaluation results

Not available yet. `python evaluate.py` (Phase 9) scores the 10 golden questions plus the
8-question refusal set and writes the gate table G1–G8 and NFR-9 to `reports/eval_report.md`.
That table will be pasted here when Phase 9 lands.

---

## Build status

| Phase | Milestone | State |
|---|---|---|
| 0 | Scaffold, config, frozen contracts | done |
| 1 | M1 Load — `ingest/loader.py`, raw cache, fetch report | done |
| 2 | M2 Clean — `ingest/cleaner.py` | done |
| 3 | M3 Chunk — `ingest/chunker.py`, `ingest/topics.py` | done |
| 4 | M4 Embed + Store — `ingest/embedder.py`, `ingest/vectorstore.py` | done |
| 5 | M5 Retrieve — `app/retriever.py` | done |
| 6 | M6 Generate — `app/prompts.py`, `app/validators.py`, `app/cache.py`, `app/generator.py` | done |
| 7 | M7 Guard — `app/guard.py` | done (built early, see below) |
| 8 | M8 UI — `app/pipeline.py`, `app/ui.py`, `app/memory.py` | done |
| 9 | M9 Evaluate — `evaluate.py`, `reports/eval_report.md` | not started |
| 10 | M10 Package — `Docs/sources.md`, `Docs/sample_qa.md`, final pins | partial (this README) |

**Phase order note.** Phase 7 (guard) was implemented before Phases 5 and 6, which
`Docs/implementation.md` §1 permits because `app/guard.py` has no Groq dependency — it is
deterministic and fully offline. `generator.answer` consumes `Decision.intent` rather than
re-classifying the question.

**Test suite:** 284 tests, all offline except the tests that stub the Groq client. Eleven of them
drive `app/ui.py` through `streamlit.testing.AppTest`, so the chat history, the sources expander,
the clear-chat button and the two startup checks are asserted rather than eyeballed.

---

## Known limits

**Corpus**

- The corpus is **five web pages** from one AMC, all `doc_type: scheme_page`, all
  Direct/Growth. There is no factsheet, KIM/SID, fee-charges page or statement guide in the
  allowlist yet, so questions outside those five pages must be refused.
- **Non-JS-rendered HTML only.** All five pages are server-rendered and were verified to contain
  the expected terms in the raw bytes. A JS-only page would extract to almost nothing, fail the
  `min_page_chars` check and be written to `data/rejected.jsonl` with the run exiting 1 — it is
  never silently dropped, and no headless browser is used to work around it.
- Answers are **as of the `content_as_of` date printed by the source**, not live. There is no
  live NAV and no market data. `freshness.warn_after_days: 90` drives a one-line `st.info`
  notice under a stale answer in the UI; it never blocks the answer (PRD §14 Q3).
- Groww pages carry their own date stamps, which may lag the AMC's official factsheet. Citations
  point at the Groww mirror; `Docs/PRD.md` §14 Q6 (Groww as an acceptable "official public
  page") is still an open question for the reviewer.

**Retrieval and answering**

- **`min_score` is a floor on the dense cosine score, not on the fused RRF score.** This is a
  documented deviation from `Docs/implementation.md` §8.3 step 9, explained in the module docstring
  of `app/retriever.py`. RRF is a rank score with no absolute scale: every golden question's top
  hit lands in 0.0245–0.0263 regardless of relevance, and the out-of-corpus Q9 ("Parag Parflex
  expense ratio") scores *above* several valid answers, so a fused floor either accepts everything
  or rejects everything. `min_score = 0.28` is unchanged from the PRD's starting value and is
  applied to the cosine inside each pass; the fused score only decides ranking.
- Cosine alone is not a scope test either — off-topic questions reach 0.377 ("What is the price of
  Bitcoin today?"), above valid in-corpus questions as low as 0.358, because MiniLM maps every
  finance-shaped question near the HDFC pages. A question that resolves no in-scope scheme must
  therefore also show lexical evidence against a retrieved chunk, or name a scheme from
  `out_of_corpus`. All 10 golden questions and the off-topic probes behave correctly.
- A 384-dim MiniLM embedder over a 5-page corpus is a lexical/statistical approximation; exact
  rare-token matching and scheme-affinity filtering carry much of the accuracy, not the vectors.
- **The corpus cannot answer five of the ten golden questions.** `lockin_80c`, `riskometer`,
  `statements_download`, `objective` and `plan_direct_vs_regular` have no chunks, because the five
  Groww pages do not carry that content. The model correctly replies "I don't have that in my
  sources." and the generator reports `status=insufficient_context` rather than pretending to
  answer. Widening the allowlist (factsheet / KIM / SID pages) is the fix, and it is a config-only
  change that must be flagged.
- **`GROQ_MODEL` was changed from `llama-3.1-8b-instant` to `qwen/qwen3.8-27b`.** Groq retired the
  Llama model and now returns `404 model_not_found`. Of the models the key can reach, the
  OpenAI-hosted `gpt-oss` variants are excluded because the project rules out the OpenAI provider;
  `qwen/qwen3.8-27b` is the remaining instruction-following chat model.

**Engineering**

- **Python version:** built and tested on 3.14.7, including `torch` and `chromadb`. The
  HuggingFace cache falls back to non-symlink mode, which warns and uses more disk. ChromaDB
  persistence is version-sensitive — install the pinned version, and if the store misbehaves after
  an upgrade, delete `data/chroma/` and re-run `.\.venv\Scripts\python.exe run_ingest.py`.
- Ingestion depends on groww.in staying up and keeping its URL slugs. A slug change is a 404 in
  the report and a non-zero exit; the allowlist in `config/config.yaml` must be updated by hand.
- A blocked (403) source is reported, never bypassed: no User-Agent rotation, no proxy, no
  retry-on-403. If groww.in starts blocking us, the fix is a different approved source, not a
  workaround.
- Groq free-tier rate limits bite on a full evaluation run: the live 10-question check hit 429s
  and the SDK's own retry took up to 16 s. `app/generator.py` adds one backoff retry for 429/5xx
  on top of that and then falls back safely, and validated answers are cached in
  `data/cache/answers.jsonl`. Rate limits remain a real constraint on Phase 9.
- No authentication, no multi-user support, no telemetry, no metrics server. Run locally.

---

## Disclaimer

`app/disclaimer.py` is the single source of truth for this string; `Docs/sample_qa.md` and
`reports/eval_report.md` mirror it verbatim.

```
Facts-only. No investment advice. This assistant answers only from official public sources (HDFC Mutual Fund / Groww scheme pages, SEBI, AMFI) for the five schemes listed in the README. It does not recommend, compare or rate funds, and does not compute returns. Mutual fund investments are subject to market risks; read the scheme's official factsheet, KIM and SID before investing. Verify every detail using the source link provided.
```

## Security note

- The Groq key lives only in `.env`, which is git-ignored; `.env.example` is committed with empty
  values. No secret is ever logged or written into `data/status.json`, `data/chunks/chunks.jsonl`
  or `reports/eval_report.md`. Before sharing the repo, scan the working tree for `gsk_`.
- User question text is never persisted or logged (FR-7.6). For a PII-classified question the
  question body is dropped before any log record is built; only `intent` and `reason_code` are
  recorded.
- `.gitignore` excludes `.env`, `data/raw/`, `data/chroma/`, `data/cache/`, `data/models/`,
  `__pycache__/`, `*.log`, `.venv/` and `.pytest_cache/`.
- Raw HTML is only ever read from allowlisted hosts, over HTTPS, with redirects restricted to
  allowlisted hosts.
