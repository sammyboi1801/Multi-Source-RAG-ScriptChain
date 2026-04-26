"""
reranker.py - cross-encoder reranking with metadata boosting and MMR diversity

Takes the top-20 candidates from retriever.py and applies three passes:
  1. Cross-encoder   - scores each (query, chunk) pair jointly; much more
                       accurate than cosine similarity because it attends to
                       both texts simultaneously
  2. Metadata boost  - small multipliers for forum primary chunks, high-reply
                       threads, and recent blog posts
  3. MMR             - Maximal Marginal Relevance selects the final top-n by
                       balancing relevance against similarity to already-selected
                       chunks, so the context window isn't full of near-duplicates

Run standalone to test reranking on live retrieval results:
    python reranker.py
"""

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from retriever import SearchResult

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL         = "all-MiniLM-L6-v2"


@dataclass
class RankedResult:
    chunk_id:      str
    source_type:   str
    source_id:     str
    source_url:    str
    title:         str
    content:       str
    metadata:      dict  = field(default_factory=dict)
    initial_score: float = 0.0   # RRF score from retriever
    ce_score:      float = 0.0   # raw cross-encoder logit
    boost:         float = 1.0   # metadata multiplier
    final_score:   float = 0.0   # ce_score * boost, used for MMR


# Metadata boosting

def _metadata_boost(result: SearchResult) -> float:
    """
    Returns a score multiplier based on structural signals in the metadata.
    Intentionally small so cross-encoder scores remain the dominant signal.
    """
    boost = 1.0
    meta  = result.metadata

    if result.source_type == "forum":
        if meta.get("is_primary"):
            boost *= 1.10   # Q+top-answer pair carries more signal than a reply
        reply_count = meta.get("reply_count", 0)
        if isinstance(reply_count, int) and reply_count > 5:
            boost *= 1.05   # high reply count = community confirmed the issue

    elif result.source_type == "blog":
        date = str(meta.get("date", ""))
        if date[:4] >= "2022":
            boost *= 1.05   # more recent posts better reflect current platform state

    return boost


# MMR

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _mmr_select(
    pool:       list[tuple[float, RankedResult, np.ndarray]],
    top_n:      int,
    lam:        float,
) -> list[RankedResult]:
    """
    Standard MMR selection loop.
    pool   : (relevance_score, result, embedding) tuples, pre-sorted by relevance
    lam    : lambda in [0, 1] - higher = more relevance, lower = more diversity
    """
    selected:    list[RankedResult] = []
    selected_emb: list[np.ndarray]  = []
    remaining = list(pool)

    while len(selected) < top_n and remaining:
        best_idx   = -1
        best_score = float("-inf")

        for i, (rel, _, emb) in enumerate(remaining):
            if not selected_emb:
                mmr = rel
            else:
                max_sim = max(_cosine(emb, s) for s in selected_emb)
                mmr     = lam * rel - (1.0 - lam) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx   = i

        _, result, emb = remaining.pop(best_idx)
        selected.append(result)
        selected_emb.append(emb)

    return selected


# Reranker

class Reranker:
    def __init__(
        self,
        ce_model:    str = CROSS_ENCODER_MODEL,
        embed_model: str = EMBED_MODEL,
    ):
        self.cross_encoder = CrossEncoder(ce_model)
        self.embed_model   = SentenceTransformer(embed_model)

    def rerank(
        self,
        query:      str,
        candidates: list[SearchResult],
        top_n:      int   = 5,
        mmr_lambda: float = 0.7,
    ) -> list[RankedResult]:
        """
        Full reranking pipeline: cross-encoder -> metadata boost -> MMR.
        Returns top_n RankedResult objects in final relevance order.
        """
        if not candidates:
            return []

        # 1. Score every (query, chunk) pair with the cross-encoder
        pairs     = [(query, c.content) for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        # 2. Apply metadata boosts and collect results
        scored: list[RankedResult] = []
        for candidate, raw_ce in zip(candidates, ce_scores):
            boost = _metadata_boost(candidate)
            scored.append(RankedResult(
                chunk_id=      candidate.chunk_id,
                source_type=   candidate.source_type,
                source_id=     candidate.source_id,
                source_url=    candidate.source_url,
                title=         candidate.title,
                content=       candidate.content,
                metadata=      candidate.metadata,
                initial_score= candidate.initial_score,
                ce_score=      float(raw_ce),
                boost=         boost,
                final_score=   float(raw_ce) * boost,
            ))

        scored.sort(key=lambda r: r.final_score, reverse=True)

        # 3. MMR over a 2x pool so a single great result isn't displaced
        #    by diversity pressure from marginal candidates
        pool_size = min(top_n * 2, len(scored))
        pool      = scored[:pool_size]

        embeddings = self.embed_model.encode(
            [r.content for r in pool],
            show_progress_bar=False,
        )

        mmr_input = [
            (r.final_score, r, embeddings[i])
            for i, r in enumerate(pool)
        ]

        return _mmr_select(mmr_input, top_n, mmr_lambda)


# Standalone test

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from retriever import Retriever

    print("Loading retriever...")
    retriever = Retriever()

    print("Loading reranker (will download cross-encoder on first run)...")
    reranker = Reranker()
    print("Ready.\n")

    test_queries = [
        "How do I enable row level security on a table?",
        "Storage upload fails with permissions error",
        "How to track user presence with Supabase Realtime?",
    ]

    for query in test_queries:
        candidates = retriever.search(query, top_k=20)
        results    = reranker.rerank(query, candidates, top_n=5)

        print(f"Query: {query!r}")
        print(f"  {'#':<3} {'source':<15} {'init':>7} {'ce':>8} {'boost':>6} {'final':>8}  title")
        print(f"  {'-'*3}  {'-'*14}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*35}")

        for i, r in enumerate(results, 1):
            print(
                f"  {i:<3} {r.source_type:<15} {r.initial_score:>7.4f} "
                f"{r.ce_score:>8.3f} {r.boost:>6.2f} {r.final_score:>8.3f}  {r.title!r}"
            )

        # Show what the retriever had at position 1 vs what reranker promoted
        top_before = candidates[0]
        top_after  = results[0]
        if top_before.chunk_id != top_after.chunk_id:
            print(f"\n  Reranker promoted: {top_after.title!r}")
            print(f"  Over retriever #1: {top_before.title!r}")
        print()

    # MMR diversity check - show that two near-identical chunks don't both appear
    print("=" * 65)
    print("MMR diversity check")
    print("=" * 65)
    q         = "row level security policies"
    cands     = retriever.search(q, top_k=20)
    reranked  = reranker.rerank(q, cands, top_n=5, mmr_lambda=0.7)
    source_ids = [r.source_id for r in reranked]
    duplicates = len(source_ids) - len(set(source_ids))
    print(f"  Source IDs in top-5: {source_ids}")
    print(f"  Duplicate source docs: {duplicates}  (0 = full diversity)")
