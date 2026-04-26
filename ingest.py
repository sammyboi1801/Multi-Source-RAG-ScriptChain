"""
ingest.py - embed chunks and load into ChromaDB

Reads all three source types from the scraped JSON files, generates sentence
embeddings, and stores them in three separate ChromaDB collections so the
retriever can apply per-source weights independently.

Run once after scraping:
    python ingest.py

Force a full re-index (drops and rebuilds existing collections):
    python ingest.py --force

Index a single source:
    python ingest.py --only docs
    python ingest.py --only blogs
    python ingest.py --only forums
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import Chunk, chunk_blogs, chunk_docs, chunk_forums

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# One collection per source so retriever.py can query and weight them separately
COLLECTIONS = {
    "documentation": "supabase_docs",
    "blog":          "supabase_blogs",
    "forum":         "supabase_forums",
}


def load_chunks() -> dict[str, list[Chunk]]:
    root = Path("data")
    docs   = json.loads((root / "docs/supabase_docs.json").read_text(encoding="utf-8"))
    blogs  = json.loads((root / "blogs/supabase_blogs.json").read_text(encoding="utf-8"))
    forums = json.loads((root / "forums/supabase_forums.json").read_text(encoding="utf-8"))
    return {
        "documentation": chunk_docs(docs),
        "blog":          chunk_blogs(blogs),
        "forum":         chunk_forums(forums),
    }


def _to_chroma_meta(chunk: Chunk) -> dict:
    """
    ChromaDB metadata values must be str, int, float, or bool.
    Lists (e.g. forum labels) are joined to a comma-separated string.
    """
    meta: dict = {
        "chunk_id":   chunk.chunk_id,
        "source_type": chunk.source_type,
        "source_id":  chunk.source_id,
        "source_url": chunk.source_url,
        "title":      chunk.title,
    }
    for k, v in chunk.metadata.items():
        if isinstance(v, list):
            meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            meta[k] = v
        else:
            meta[k] = str(v)
    return meta


def _embed_batched(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs = model.encode(batch, show_progress_bar=False)
        embeddings.extend(vecs.tolist())
        done = min(start + BATCH_SIZE, total)
        print(f"    {done}/{total} embedded", end="\r", flush=True)
    print()
    return embeddings


def ingest_source(
    source_type: str,
    chunks: list[Chunk],
    model: SentenceTransformer,
    client: chromadb.ClientAPI,
    force: bool = False,
) -> int:
    col_name = COLLECTIONS[source_type]

    if force:
        try:
            client.delete_collection(col_name)
            print(f"  Dropped existing collection: {col_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if not force and existing > 0:
        print(f"  {col_name}: already has {existing} chunks, skipping (use --force to rebuild)")
        return existing

    print(f"  Embedding {len(chunks)} {source_type} chunks...")

    ids        = [c.chunk_id for c in chunks]
    documents  = [c.content  for c in chunks]
    metadatas  = [_to_chroma_meta(c) for c in chunks]
    embeddings = _embed_batched(documents, model)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    count = collection.count()
    print(f"  {col_name}: {count} chunks stored")
    return count


def _verify_collection(col_name: str, client: chromadb.ClientAPI, model: SentenceTransformer) -> bool:
    """
    ChromaDB 1.5.x occasionally fails to flush the HNSW index on first creation,
    producing 'Nothing found on disk' on the next query. A quick probe catches this
    before the caller hits it at query time.
    """
    try:
        col = client.get_collection(col_name)
        vec = model.encode(["test"])[0].tolist()
        col.query(query_embeddings=[vec], n_results=1)
        return True
    except Exception:
        return False


def run(only: Optional[str] = None, force: bool = False) -> None:
    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Chunking source files...")
    all_chunks = load_chunks()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    source_filter = {
        "docs":   "documentation",
        "blogs":  "blog",
        "forums": "forum",
    }
    targets = [source_filter[only]] if only else list(all_chunks.keys())

    total = 0
    for source_type in targets:
        print(f"\n[{source_type}]")
        total += ingest_source(source_type, all_chunks[source_type], model, client, force=force)

    # Verify each collection is actually queryable; rebuild silently if not
    print("\nVerifying collections...")
    for source_type in targets:
        col_name = COLLECTIONS[source_type]
        if not _verify_collection(col_name, client, model):
            print(f"  {col_name}: query probe failed, rebuilding...")
            ingest_source(source_type, all_chunks[source_type], model, client, force=True)
        else:
            print(f"  {col_name}: ok")

    print(f"\nIndex ready: {total} chunks across {len(targets)} collection(s)")


# Standalone test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and index chunks into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild existing collections")
    parser.add_argument("--only", choices=["docs", "blogs", "forums"], help="Index one source type only")
    args = parser.parse_args()

    run(only=args.only, force=args.force)

    # Sanity check: query each collection and confirm results come back
    print("\n-- Sanity queries ------------------------------------------")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    probes = [
        ("supabase_docs",   "How do I enable row level security?"),
        ("supabase_blogs",  "What is new in Supabase storage?"),
        ("supabase_forums", "Authentication not working after upgrade"),
    ]
    for col_name, query in probes:
        try:
            col = client.get_collection(col_name)
            vec = model.encode([query])[0].tolist()
            res = col.query(query_embeddings=[vec], n_results=1)
            meta = res["metadatas"][0][0]
            snippet = res["documents"][0][0][:120].replace("\n", " ")
            print(f"\n  [{col_name}] '{query}'")
            print(f"  -> {meta['title']!r}")
            print(f"     {snippet!r}")
        except Exception as e:
            print(f"  [{col_name}] query failed: {e}")
