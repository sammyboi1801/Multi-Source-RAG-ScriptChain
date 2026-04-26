# Multi-Source RAG - Supabase Technical Support

A RAG system that answers developer questions about Supabase by pulling from three
knowledge sources: official documentation, blog posts, and GitHub issue threads.


**Stack:** Python, ChromaDB, sentence-transformers, Groq (or Ollama)

## Video Presentation

**Link** - https://youtu.be/L5HU1faF20E

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

To use a local Ollama model instead (no API key needed):

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

---

## Running

The pipeline has three steps. Run them in order on first use.

### Step 1 - Scrape

Downloads docs, blog posts, and GitHub issues from Supabase's public repository.

```bash
python scraper.py
```

Saves JSON files to `data/docs/`, `data/blogs/`, `data/forums/`.

> Scraping forums hits the GitHub API (unauthenticated = 10 req/min). Set
> `GITHUB_TOKEN=ghp_...` in `.env` for a higher rate limit if it slows down.

### Step 2 - Ingest

Generates embeddings and loads everything into ChromaDB.

```bash
python ingest.py
```

Only needs to run once. To force a full rebuild:

```bash
python ingest.py --force
```

### Step 3 - Run

**Demo mode** - runs 10 pre-set queries and prints a performance summary:

```bash
python main.py
```

**Chat mode** - interactive question/answer:

```bash
python main.py --chat
```

**Stats only** - print aggregate stats from the audit log without running any queries:

```bash
python main.py --stats-only
```

---

## Project Structure

```
scraper.py          Downloads docs, blogs, and forum threads from GitHub
chunker.py          Splits raw content into chunks (different strategy per source)
ingest.py           Embeds chunks and stores them in ChromaDB
retriever.py        Hybrid dense + BM25 retrieval with per-source weighting
reranker.py         Cross-encoder reranking + MMR diversity selection
rag_pipeline.py     Orchestrates the full pipeline + contradiction handling
logger.py           Appends a JSONL audit record after each query
main.py             CLI entry point (demo / chat / stats)

data/               Scraped JSON files (created by scraper.py)
chroma_db/          Vector store (created by ingest.py)
logs/               Query audit log (created on first query)

REPORT.md           Written report covering each requirement
PERFORMANCE_ANALYSIS.md  Results from the 10-query demo run
```

---

## Individual Module Testing

Each module can be run standalone to verify it works before running the full pipeline:

```bash
python chunker.py       # prints chunk counts and spot-checks one chunk per source
python ingest.py        # embeds and indexes, then runs sanity queries
python retriever.py     # runs 5 test queries and shows ranked results
python reranker.py      # shows before/after reranking comparison
python rag_pipeline.py  # runs 3 sample queries end-to-end (needs API key)
python logger.py        # writes a test log entry and prints current stats
```
