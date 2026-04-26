"""
retriever.py - hybrid retrieval with per-source weighting

Per query:
  1. Dense search  - ChromaDB cosine similarity, queried per source collection
  2. Sparse search - BM25Okapi, one index per source built at init time
  3. RRF merge     - Reciprocal Rank Fusion combines dense + sparse per source
  4. Source weight - documentation > blog > forum, with dynamic forum boost
                     for queries that look like error/workaround requests

Returns up to top_k candidates for the reranker to score.

Run standalone to test retrieval on sample queries:
    python retriever.py
"""

import re
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

SOURCE_WEIGHTS: dict[str, float] = {
    "documentation": 1.0,
    "blog":          0.75,
    "forum":         0.6,
}

COLLECTIONS: dict[str, str] = {
    "documentation": "supabase_docs",
    "blog":          "supabase_blogs",
    "forum":         "supabase_forums",
}

# When these appear in the query the user likely wants community fixes -> lift forum weight
_FORUM_BOOST_TERMS = frozenset({
    "error", "bug", "broken", "not working", "workaround", "fix", "fail",
    "crash", "exception", "problem", "issue", "help", "stuck", "cannot", "cant",
})


@dataclass
class SearchResult:
    chunk_id:    str
    source_type: str
    source_id:   str
    source_url:  str
    title:       str
    content:     str
    metadata:    dict = field(default_factory=dict)
    initial_score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class Retriever:
    def __init__(self, chroma_dir: str = CHROMA_DIR, embed_model: str = EMBED_MODEL):
        self.model  = SentenceTransformer(embed_model)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self._bm25:        dict[str, BM25Okapi]   = {}
        self._bm25_chunks: dict[str, list[dict]]  = {}
        self._load_bm25_indices()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_bm25_indices(self) -> None:
        """Pull every chunk from ChromaDB and build one BM25 index per source."""
        for source_type, col_name in COLLECTIONS.items():
            try:
                col    = self.client.get_collection(col_name)
                result = col.get(include=["documents", "metadatas"])
                docs   = result["documents"]
                metas  = result["metadatas"]

                self._bm25[source_type]        = BM25Okapi([_tokenize(d) for d in docs])
                self._bm25_chunks[source_type] = [
                    {"content": doc, "metadata": meta}
                    for doc, meta in zip(docs, metas)
                ]
            except Exception as exc:
                print(f"Warning: BM25 index skipped for {source_type}: {exc}")

    # ------------------------------------------------------------------
    # Per-query helpers
    # ------------------------------------------------------------------

    def _effective_weight(self, source_type: str, query: str) -> float:
        weight = SOURCE_WEIGHTS.get(source_type, 0.5)
        if source_type == "forum":
            q = query.lower()
            if any(term in q for term in _FORUM_BOOST_TERMS):
                weight = min(weight * 1.5, SOURCE_WEIGHTS["documentation"])
        return weight

    def _dense_search(self, query: str, source_type: str, n: int) -> list[SearchResult]:
        try:
            col = self.client.get_collection(COLLECTIONS[source_type])
            vec = self.model.encode([query])[0].tolist()
            raw = col.query(
                query_embeddings=[vec],
                n_results=min(n, col.count()),
                include=["documents", "metadatas", "distances"],
            )
            results = []
            for doc, meta, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ):
                results.append(SearchResult(
                    chunk_id=    meta.get("chunk_id", ""),
                    source_type= source_type,
                    source_id=   meta.get("source_id", ""),
                    source_url=  meta.get("source_url", ""),
                    title=       meta.get("title", ""),
                    content=     doc,
                    metadata=    meta,
                    initial_score= 1.0 - float(dist),   # cosine distance -> similarity
                ))
            return results
        except Exception:
            return []

    def _sparse_search(self, query: str, source_type: str, n: int) -> list[SearchResult]:
        if source_type not in self._bm25:
            return []

        tokens = _tokenize(query)
        scores = self._bm25[source_type].get_scores(tokens)
        chunks = self._bm25_chunks[source_type]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        results = []
        for idx in ranked:
            if scores[idx] == 0:
                break
            meta = chunks[idx]["metadata"]
            results.append(SearchResult(
                chunk_id=    meta.get("chunk_id", ""),
                source_type= source_type,
                source_id=   meta.get("source_id", ""),
                source_url=  meta.get("source_url", ""),
                title=       meta.get("title", ""),
                content=     chunks[idx]["content"],
                metadata=    meta,
                initial_score= float(scores[idx]),
            ))
        return results

    def _rrf_merge(
        self,
        dense:         list[SearchResult],
        sparse:        list[SearchResult],
        source_weight: float,
        k:             int = 60,
    ) -> list[SearchResult]:
        """
        Standard RRF: score(d) = sum_over_lists( 1 / (k + rank) ).
        Rank is 1-indexed. Source weight is applied after fusion so it
        scales the entire source uniformly rather than individual signals.
        """
        rrf_scores: dict[str, float]        = {}
        chunk_map:  dict[str, SearchResult] = {}

        for rank, result in enumerate(dense, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map.setdefault(result.chunk_id, result)

        for rank, result in enumerate(sparse, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map.setdefault(result.chunk_id, result)

        merged = []
        for chunk_id, rrf in rrf_scores.items():
            result = chunk_map[chunk_id]
            result.initial_score = rrf * source_weight
            merged.append(result)

        merged.sort(key=lambda r: r.initial_score, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """
        Returns up to top_k candidates sorted by weighted RRF score.
        Pass these directly to reranker.rerank().
        """
        all_results: list[SearchResult] = []

        for source_type in COLLECTIONS:
            weight = self._effective_weight(source_type, query)
            dense  = self._dense_search(query, source_type, n=top_k)
            sparse = self._sparse_search(query, source_type, n=top_k)
            merged = self._rrf_merge(dense, sparse, weight)
            all_results.extend(merged)

        all_results.sort(key=lambda r: r.initial_score, reverse=True)
        return all_results[:top_k]


# Standalone test

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        ("How do I set up row level security?",        "expects docs"),
        ("Storage upload not working error",           "expects forum boost"),
        ("How do edge functions handle authentication?","expects docs + blogs"),
        ("realtime presence tracking users online",    "expects docs + blogs"),
        ("JWT token expired after migration bug",      "expects forum boost"),
    ]

    print("Building retriever (loading model + BM25 indices)...")
    retriever = Retriever()
    print("Ready.\n")

    for query, note in test_queries:
        results = retriever.search(query, top_k=5)
        print(f"Query : {query!r}  [{note}]")
        print(f"{'':>4} {'source':<15} {'score':>7}  title")
        print(f"{'':>4} {'-'*14}  {'-'*6}  {'-'*40}")
        for i, r in enumerate(results, 1):
            snippet = r.content[:80].replace("\n", " ")
            print(f"  {i}.  {r.source_type:<15} {r.initial_score:>7.4f}  {r.title!r}")
            print(f"       {snippet!r}")
        print()

    # Show the forum weight boost in action
    print("=" * 60)
    print("Forum weight boost check")
    print("=" * 60)
    baseline = "How does realtime work"
    boosted  = "realtime not working error help"
    for label, q in [("no boost", baseline), ("boosted ", boosted)]:
        w = retriever._effective_weight("forum", q)
        print(f"  {label}: forum weight = {w:.2f}  (query={q!r})")
