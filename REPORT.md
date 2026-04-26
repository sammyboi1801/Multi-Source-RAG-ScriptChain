# Multi-Source RAG for Technical Support
## Approach Report

**Candidate:** Sam Selvaraj  
**Submission Date:** April 25, 2026  
**Product:** Supabase (fictional product used as knowledge base)  
**Stack:** Python, ChromaDB, sentence-transformers, Groq (llama-3.3-70b-versatile)

---

## Overview

This system answers developer questions about Supabase by pulling from three distinct
knowledge sources - official documentation, technical blog posts, and community forum
threads - then combining them through a hybrid retrieval and reranking pipeline before
generating a grounded answer via an LLM.

The main design goal was making sure the source mix reflects the question type: a
how-to question should lean on docs, a "why is this broken" question should surface
forum threads. Attribution is always shown so developers know how authoritative each
piece of information is.

---

## Requirement 1: Three Data Sources

**File:** `scraper.py`

Three structurally distinct data sources were scraped from Supabase's public GitHub
repository:

| Source | Origin | Volume | Format |
|---|---|---|---|
| Documentation | `supabase/supabase` MDX guides | 25 pages | Structured markdown with headings and code blocks |
| Blog posts | `supabase/supabase` MDX blog | 25 posts | Narrative MDX with section headings |
| Forum threads | GitHub Issues API | 49 threads | Q&A with multiple replies, labels, and state |

The three sources cover different kinds of knowledge: docs tell you how things are
supposed to work, blogs show real usage patterns and tutorials, and forum threads are
where you find out what actually breaks and how people fixed it.

The scraper handles MDX frontmatter parsing, JSX component stripping, and GitHub API
pagination. Forum data is fetched via the GitHub Issues search API using queries targeting
`bug`, `question`, and error-related labels to ensure the forum corpus is representative
of actual support traffic.

---

## Requirement 2: Source-Appropriate Chunking

**File:** `chunker.py`

One chunking strategy was applied per source type rather than a generic fixed-size split.
The data was inspected before any strategy was chosen - this revealed that blogs have
between 5 and 20 section headings each, making sliding windows an actively bad choice
for that source.

### Documentation - Heading-Based Hierarchical

Splits on `##` and `###` markdown headers while tracking code fence state so headings
inside code blocks are never treated as split boundaries. Each chunk represents one
complete concept or procedure. Sections under 60 characters are discarded as
heading-only lines with no body content.

**Result:** 265 chunks, average 1,023 characters each.

### Blog Posts - Heading-Based with JSX Cleanup

All 25 scraped blog posts use the same MDX format as documentation and have explicit
section headings. A two-pass HTML stripper removes JSX render components (`<iframe>`,
`<video>`, `<source>`, `<div className>`) left behind by the scraper before splitting.
The first pass uses a DOTALL regex to handle multi-line tags spanning 4-6 lines; the
second pass handles remaining single-line tags while respecting code fences.

Sections under 100 characters are discarded (a higher threshold than docs because
blog sections should have narrative substance; short blobs indicate noise).

**Result:** 267 chunks, average 750 characters each.

### Forum Threads - Thread-Level + Post-Level

Each thread produces a primary chunk (original question + first reply) and one secondary
chunk per additional reply. The original question is prefixed to every secondary chunk
so retrieval always has context for what problem was being discussed.

GitHub issue templates prefix every question with `# Bug report` or `# Feature request`
headers. These are stripped before chunking because they would otherwise dominate
embeddings for 39 of 49 threads. Bold template section labels (`**Describe the bug**`)
are also removed.

No accepted-answer field exists in GitHub Issues, so the first comment is used as the
top reply (the scraper fetches issues sorted by comment count, meaning threads with
substantive discussions appear first).

**Result:** 194 chunks, average 1,420 characters each.

**Total corpus: 726 chunks across three ChromaDB collections.**

---

## Requirement 3: Multi-Source Retrieval with Intelligent Weighting

**File:** `retriever.py`

A hybrid retrieval approach is used: dense retrieval (vector similarity) and sparse
retrieval (BM25) are run independently per source, merged with Reciprocal Rank Fusion,
then scaled by source-specific weights.

### Dense Retrieval

Chunks are embedded with `all-MiniLM-L6-v2` (384 dimensions, cosine similarity space)
and stored in ChromaDB. Three separate collections are maintained - one per source type -
so the retriever can query and weight them independently rather than filtering post-hoc
on a unified index.

### Sparse Retrieval

A BM25Okapi index is built per source at initialisation time from the same ChromaDB
content. BM25 complements dense retrieval for keyword-heavy queries (error codes,
function names, version numbers) where semantic similarity alone misses exact matches.

### Reciprocal Rank Fusion

Dense and sparse results are merged per source using standard RRF:

```
score(d) = sum over lists of 1 / (k + rank)   where k = 60
```

RRF is robust to score scale differences between dense and sparse signals because it
operates on ranks rather than raw scores.

### Source Weighting

After RRF fusion, each source's scores are multiplied by a static weight reflecting
source authority:

```python
SOURCE_WEIGHTS = {
    "documentation": 1.0,
    "blog":          0.75,
    "forum":         0.6,
}
```

Forum weight is dynamically boosted to 0.9 (capped at the docs weight) when the query
contains error/debugging terms such as `error`, `bug`, `not working`, `workaround`, or
`crash`. If someone hits a storage permissions error or a PGRST116, a GitHub issue thread
is going to be more useful than the official docs.

The retriever returns the top 20 candidates across all three sources, sorted by weighted
RRF score, for the reranker to process.

---

## Requirement 4: Reranking Mechanism

**File:** `reranker.py`

The top 20 retrieval candidates are reranked using a three-stage pipeline:

### Stage 1: Cross-Encoder Scoring

`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair jointly. Unlike
bi-encoder cosine similarity - where query and document are embedded independently - a
cross-encoder attends to both texts simultaneously, producing substantially better
relevance scores at the cost of higher latency. Running it only on the top 20 candidates
(rather than the full corpus) keeps this cost acceptable.

Cross-encoder scores are logit values that can range from approximately -10 to +10.
Positive scores indicate relevance; negative scores indicate the model found the chunk
not useful for the query.

### Stage 2: Metadata Boosting

Small multipliers are applied on top of cross-encoder scores based on structural signals:

| Signal | Multiplier | Rationale |
|---|---|---|
| Forum primary chunk (`is_primary=True`) | 1.10x | Q+top-answer pairing carries more signal than isolated replies |
| Forum thread with >5 replies | 1.05x | Community engagement suggests real-world relevance |
| Blog post from 2022 or later | 1.05x | More recent posts better reflect current platform state |

Boosts are kept small (5-10%) so cross-encoder scores remain the dominant ranking signal.

### Stage 3: Maximal Marginal Relevance

After sorting by boosted score, MMR is applied over a pool of `top_n * 2` candidates
to select the final top-5 context chunks:

```
MMR(d) = λ * relevance(d) - (1 - λ) * max_similarity(d, already_selected)
```

`λ = 0.7` favours relevance over diversity. Similarity between candidates is computed
using sentence embeddings from the same `all-MiniLM-L6-v2` model. This prevents the
context window from being filled with near-identical chunks from the same document
section while still allowing the most relevant source to contribute multiple chunks
when it genuinely has the best answers.

---

## Requirement 5: Contradiction Handling

**File:** `rag_pipeline.py`

Contradiction handling uses a two-stage approach: detection then resolution.

### Detection

After reranking, the top 5 chunks are sent to the LLM with a dedicated prompt asking
it to identify factual conflicts and return JSON:

```json
{"contradictions": [{"indices": [0, 2], "description": "..."}]}
```

The prompt is temperature-0 for deterministic output. JSON is parsed with a markdown
code-fence stripper to handle models that wrap output in triple backticks. Detection
is best-effort - if the LLM returns malformed JSON or the call fails, the answer
generation proceeds without a contradiction note rather than blocking the response.

### Resolution

Detected conflicts are resolved in priority order:

1. **Authority wins** - Documentation outranks Blog outranks Forum. If a doc and a forum
   thread disagree, the documentation answer is used and the conflict is noted.

2. **Recency wins** - When sources have equal authority, the more recent post (by `date`
   for blogs, `created_at` for forums) is preferred. Blog dates in `MM-DD-YYYY` format
   are normalised to `YYYY-MM-DD` before comparison.

3. **Transparency fallback** - If neither signal resolves the conflict, both answers are
   surfaced to the user with a note to verify against the official docs. The pipeline
   never invents agreement between sources.

### Observed Rate

Across the 10 demo queries, contradiction detection found 0 conflicts. Supabase writes
and maintains all three source types, so the corpus is tightly controlled. The detection
still runs on every query and would catch conflicts in a messier corpus - multiple
third-party guides covering the same API, for example, or docs spread across versions.

---

## Requirement 6: Source Logging

**File:** `logger.py`

Every query appends one JSON line to `logs/rag_audit.jsonl`. Each entry records:

```json
{
  "query_id": "uuid",
  "timestamp": "2026-04-25T...",
  "model": "llama-3.3-70b-versatile",
  "query": "...",
  "answer": "...",
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "source_type": "documentation",
      "source_id": "auth_overview",
      "title": "...",
      "score_before_rerank": 0.02841,
      "used_in_prompt": true
    }
  ],
  "final_chunks": [
    {
      "chunk_id": "...",
      "source_type": "documentation",
      "ce_score": 5.285,
      "boost": 1.0,
      "final_score": 5.285
    }
  ],
  "contradiction_detected": false,
  "contradiction_note": "",
  "final_source_mix": {"documentation": 4, "blog": 1}
}
```

JSONL (one record per line) was chosen over a structured database because it requires
no schema migration, can be read with a single `open()` call, and appends atomically.
`logger.py` also exposes `read_logs()` and `print_stats()` for computing aggregate
metrics directly from the log file.

---

## Architecture Summary

```
scraper.py       ->  data/{docs,blogs,forums}/*.json
chunker.py       ->  726 Chunk objects (in memory)
ingest.py        ->  ChromaDB collections + BM25 indices built at runtime
retriever.py     ->  top-20 SearchResult candidates per query
reranker.py      ->  top-5 RankedResult with CE scores
rag_pipeline.py  ->  RAGResponse (answer + contradiction note + source mix)
logger.py        ->  logs/rag_audit.jsonl
main.py          ->  CLI: demo mode / chat mode / stats-only
```

Each module is independently runnable (`python chunker.py`, `python ingest.py`, etc.)
for verification and debugging without running the full pipeline.
