"""
main.py - 10 example queries with full performance analysis + interactive chat

Modes:
    python main.py               - run all 10 demo queries then show stats
    python main.py --chat        - interactive chat interface
    python main.py --stats-only  - print stats from existing log, no queries

Provider setup:
    $env:GROQ_API_KEY = "gsk_..."       (Groq, default)
    $env:LLM_PROVIDER = "ollama"        (local Ollama, no key needed)
"""

import argparse
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from logger import log_query, print_stats
from rag_pipeline import RAGPipeline

# Query set

# Chosen to exercise the full retrieval surface:
#   - Docs-heavy (setup/reference questions)
#   - Forum-boosted (error/debugging queries)
#   - Blog-likely (conceptual + best-practice questions)
#   - Cross-source (questions where all three sources should contribute)
#   - Potential contradictions (topics that evolved across versions)

QUERIES = [
    # Auth
    "How do I enable email and password authentication in Supabase?",
    "JWT token keeps expiring unexpectedly after login - how do I fix this?",

    # Row Level Security
    "How do I write a row level security policy that lets users only see their own data?",
    "RLS policy not working for service role key - why?",

    # Storage
    "How do I upload files to Supabase Storage from a React app?",
    "Storage uploads failing with permissions error after enabling RLS",

    # Realtime
    "How does Supabase Realtime presence work and how do I track online users?",
    "Realtime subscription stops receiving events after a few minutes",

    # Database / Edge Functions
    "What is the recommended way to run scheduled jobs in Supabase?",
    "How do I call a Supabase Edge Function from the client and pass auth headers?",
]


# Runner

def run_all(pipeline: RAGPipeline) -> None:
    total_queries = len(QUERIES)

    for i, question in enumerate(QUERIES, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{total_queries}] {question}")
        print("=" * 70)

        t0       = time.perf_counter()
        response = pipeline.query(question)
        elapsed  = time.perf_counter() - t0

        query_id = log_query(response, model=pipeline.model)

        # Answer
        print(f"\n{response.answer}\n")

        # Sources
        print("Sources:")
        for r in response.final_results:
            print(
                f"  [{r.source_type:<15}]  ce={r.ce_score:>7.3f}  "
                f"boost={r.boost:.2f}  {r.title!r}"
            )

        # Metadata
        mix_str = ", ".join(f"{k}:{v}" for k, v in sorted(response.source_mix.items()))
        print(f"\nSource mix : {mix_str}")
        print(f"Time       : {elapsed:.2f}s")
        print(f"Query ID   : {query_id}")

        if response.contradiction_detected:
            print(f"\nContradiction: {response.contradiction_note}")


# Chat interface

def run_chat(pipeline: RAGPipeline) -> None:
    print(f"\nSupabase Technical Support  (model: {pipeline.model})")
    print("Ask anything about Supabase. Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        t0       = time.perf_counter()
        response = pipeline.query(question)
        elapsed  = time.perf_counter() - t0

        log_query(response, model=pipeline.model)

        print(f"\nAssistant: {response.answer}\n")

        mix_str = ", ".join(f"{k}:{v}" for k, v in sorted(response.source_mix.items()))
        print(f"Sources: {mix_str}  |  {elapsed:.2f}s")

        if response.contradiction_detected:
            print(f"Note: {response.contradiction_note}")

        print()


# Entry point

def _build_pipeline() -> RAGPipeline:
    print("\nInitialising pipeline (loading models + BM25 indices)...")
    try:
        pipeline = RAGPipeline()
        print(f"Provider : {pipeline.model}\n")
        return pipeline
    except EnvironmentError as exc:
        print(f"\nConfiguration error: {exc}")
        print(
            "\nSet your API key and try again:\n"
            "  $env:GROQ_API_KEY = 'gsk_...'\n"
            "  python main.py\n"
            "\nOr use a local Ollama model:\n"
            "  $env:LLM_PROVIDER = 'ollama'\n"
            "  python main.py"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat",       action="store_true", help="Start interactive chat")
    parser.add_argument("--stats-only", action="store_true", help="Print log stats and exit")
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Source RAG - Supabase Technical Support")
    print("=" * 70)

    if args.stats_only:
        print_stats()
        return

    pipeline = _build_pipeline()

    if args.chat:
        run_chat(pipeline)
        return

    # Default: run all 10 demo queries
    print(f"Queries  : {len(QUERIES)}\n")
    run_all(pipeline)

    print(f"\n\n{'=' * 70}")
    print("Performance Analysis (all queries)")
    print("=" * 70)
    print_stats()


if __name__ == "__main__":
    main()
