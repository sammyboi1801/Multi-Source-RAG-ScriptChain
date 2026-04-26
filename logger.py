"""
logger.py - JSONL audit logging for RAG queries

Appends one record per query to logs/rag_audit.jsonl.

Each entry captures:
  - All retrieved candidates with pre-rerank scores
  - Final chunks that made it into the context, with CE scores and boosts
  - Contradiction flag and resolution note
  - Source mix summary

The log feeds directly into the performance analysis in main.py:
  - Source utilization rates
  - Average rerank score delta (before vs after cross-encoder)
  - Contradiction frequency across queries

Run standalone to write a test entry and print current stats:
    python logger.py
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag_pipeline import RAGResponse

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "rag_audit.jsonl"


# Write

def log_query(response: RAGResponse, model: str = "") -> str:
    """
    Append one JSONL record for a completed query.
    Returns the generated query_id for cross-referencing with the caller.
    """
    LOG_DIR.mkdir(exist_ok=True)

    final_ids = {r.chunk_id for r in response.final_results}

    entry = {
        "query_id":   str(uuid4()),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "model":      model,
        "query":      response.query,
        "answer":     response.answer,
        "retrieved_chunks": [
            {
                "chunk_id":            c.chunk_id,
                "source_type":         c.source_type,
                "source_id":           c.source_id,
                "title":               c.title,
                "score_before_rerank": round(c.initial_score, 6),
                "used_in_prompt":      c.chunk_id in final_ids,
            }
            for c in response.all_candidates
        ],
        "final_chunks": [
            {
                "chunk_id":    r.chunk_id,
                "source_type": r.source_type,
                "source_id":   r.source_id,
                "title":       r.title,
                "ce_score":    round(r.ce_score, 4),
                "boost":       round(r.boost, 4),
                "final_score": round(r.final_score, 4),
            }
            for r in response.final_results
        ],
        "contradiction_detected": response.contradiction_detected,
        "contradiction_note":     response.contradiction_note,
        "final_source_mix":       response.source_mix,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry["query_id"]


# Read + analysis

def read_logs(log_file: Path = LOG_FILE) -> list[dict]:
    if not log_file.exists():
        return []
    entries = []
    with open(log_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def print_stats(log_file: Path = LOG_FILE) -> None:
    entries = read_logs(log_file)
    if not entries:
        print("No log entries found.")
        return

    n = len(entries)
    print(f"Queries logged: {n}")
    print()

    # --- Source utilization ---
    source_counts: Counter = Counter()
    for e in entries:
        for chunk in e.get("final_chunks", []):
            source_counts[chunk["source_type"]] += 1

    total_chunks = sum(source_counts.values())
    print("Source utilization in final contexts:")
    for source, count in source_counts.most_common():
        pct = 100 * count / total_chunks if total_chunks else 0
        print(f"  {source:<16} {count:>4} chunks  ({pct:.1f}%)")
    print()

    # --- Rerank delta ---
    # Compare each final chunk's initial retrieval score vs CE final score.
    # CE scores are logits, not directly comparable to RRF scores, so we
    # report them separately rather than computing a misleading delta.
    rrf_scores, ce_scores = [], []
    for e in entries:
        cand_map = {
            c["chunk_id"]: c["score_before_rerank"]
            for c in e.get("retrieved_chunks", [])
        }
        for fc in e.get("final_chunks", []):
            before = cand_map.get(fc["chunk_id"])
            if before is not None:
                rrf_scores.append(before)
                ce_scores.append(fc["final_score"])

    if rrf_scores:
        avg_rrf = sum(rrf_scores) / len(rrf_scores)
        avg_ce  = sum(ce_scores)  / len(ce_scores)
        print(f"Avg retriever score (RRF * source weight): {avg_rrf:.5f}")
        print(f"Avg reranker score  (CE * boost):          {avg_ce:.3f}")
        print()

    # --- Contradiction rate ---
    n_contradictions = sum(1 for e in entries if e.get("contradiction_detected"))
    print(f"Contradiction rate: {n_contradictions}/{n} ({100 * n_contradictions / n:.1f}%)")
    print()

    # --- Per-query source mix ---
    print("Per-query source mix:")
    for e in entries:
        mix     = e.get("final_source_mix", {})
        mix_str = ", ".join(f"{k}:{v}" for k, v in sorted(mix.items()))
        q_short = e["query"][:58]
        flag    = " [contradiction]" if e.get("contradiction_detected") else ""
        print(f"  {q_short:<58}  [{mix_str}]{flag}")


# Standalone test

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from reranker import Reranker
    from retriever import Retriever

    # Build a real RAGResponse using retriever + reranker but no LLM,
    # so this test works without any API key configured.
    print("Building retriever and reranker...")
    retriever = Retriever()
    reranker  = Reranker()

    test_query = "How do I set up row level security in Supabase?"
    print(f"Running retrieval + reranking for: {test_query!r}\n")

    candidates = retriever.search(test_query, top_k=20)
    results    = reranker.rerank(test_query, candidates, top_n=5)

    mock_response = RAGResponse(
        query=test_query,
        answer="[Mock answer - LLM not called in logger test]",
        final_results=results,
        all_candidates=candidates,
        contradiction_detected=False,
        contradiction_note="",
        source_mix={r.source_type: 0 for r in results},
    )
    # Fix source mix count
    for r in results:
        mock_response.source_mix[r.source_type] = (
            mock_response.source_mix.get(r.source_type, 0) + 1
        )

    query_id = log_query(mock_response, model="test-no-llm")
    print(f"Logged entry: {query_id}")
    print(f"Log file    : {LOG_FILE}\n")

    print("Final chunks logged:")
    for r in results:
        print(f"  [{r.source_type:<15}] ce={r.ce_score:>7.3f}  {r.title!r}")

    print()
    print("=" * 60)
    print("Current log stats")
    print("=" * 60)
    print_stats()
