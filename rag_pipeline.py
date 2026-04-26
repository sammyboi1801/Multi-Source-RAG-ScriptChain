"""
rag_pipeline.py - full RAG pipeline orchestration

Ties together retriever, reranker, contradiction detection, and LLM answer generation.
Two LLM calls per query:
  1. Contradiction check - cheap call that inspects the top chunks for conflicts
  2. Answer generation   - full answer with source attribution

LLM provider is set via environment variables:

    # Groq (default - fast, free tier)
    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama-3.3-70b-versatile

    # Ollama (local, no API key needed)
    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434/v1
    OLLAMA_MODEL=llama3.2

Run standalone:
    python rag_pipeline.py
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from reranker import RankedResult, Reranker
from retriever import SearchResult, Retriever

# LLM client setup

def _build_llm_client() -> tuple[OpenAI, str]:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Export it or set LLM_PROVIDER=ollama to use a local model."
            )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model    = os.getenv("OLLAMA_MODEL", "llama3.2")
        client   = OpenAI(api_key="ollama", base_url=base_url)

    else:
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER={provider!r}. Valid options: 'groq', 'ollama'."
        )

    return client, model


# Response dataclass

@dataclass
class RAGResponse:
    query:                  str
    answer:                 str
    final_results:          list[RankedResult]    # top-n used to build the context
    all_candidates:         list[SearchResult]    # top-20 from retriever (for logging)
    contradiction_detected: bool
    contradiction_note:     str
    source_mix:             dict[str, int]        # {"documentation": 2, "forum": 1, ...}


# Contradiction detection + resolution

_AUTHORITY = {"documentation": 3, "blog": 2, "forum": 1}

def _parse_date(result: RankedResult) -> str:
    """Pull the most reliable date string from a result's metadata."""
    raw = str(
        result.metadata.get("date")
        or result.metadata.get("created_at")
        or ""
    )
    # Normalise MM-DD-YYYY -> YYYY-MM-DD for clean string comparison
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", raw)
    if m:
        raw = f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return raw[:10]


def _resolve(
    a: RankedResult,
    b: RankedResult,
    description: str,
) -> str:
    """
    Apply priority rules to pick a side and return a human-readable note.
    Priority: source authority > recency > transparency fallback.
    """
    auth_a = _AUTHORITY.get(a.source_type, 0)
    auth_b = _AUTHORITY.get(b.source_type, 0)

    if auth_a != auth_b:
        winner = a if auth_a > auth_b else b
        loser  = b if auth_a > auth_b else a
        return (
            f"Note: Sources disagree here. Prioritising the {winner.source_type} "
            f"(\"{winner.title}\") over the {loser.source_type} (\"{loser.title}\"). "
            f"{description}"
        )

    date_a = _parse_date(a)
    date_b = _parse_date(b)
    if date_a and date_b and date_a != date_b:
        winner = a if date_a > date_b else b
        return (
            f"Note: Sources disagree. Using the more recent source "
            f"(\"{winner.title}\", {max(date_a, date_b)}). {description}"
        )

    # Can't automatically resolve - surface both
    return (
        f"Note: Sources provide conflicting information on this point. "
        f"\"{a.title}\" ({a.source_type}) and \"{b.title}\" ({b.source_type}) "
        f"disagree: {description}. Verify against the official documentation."
    )


def _check_contradictions(
    query:   str,
    results: list[RankedResult],
    client:  OpenAI,
    model:   str,
) -> tuple[bool, str]:
    """
    Asks the LLM to scan the top chunks for factual conflicts.
    Returns (contradiction_found, resolution_note).
    """
    passages = "\n\n".join(
        f"[{i}] ({r.source_type}) \"{r.title}\"\n{r.content[:400]}"
        for i, r in enumerate(results)
    )

    prompt = (
        "You are checking retrieved passages for factual contradictions "
        "before answering a user question.\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{passages}\n\n"
        "Do any passages state contradictory facts? "
        "Reply with JSON only, no other text:\n"
        '{"contradictions": [{"indices": [i, j], "description": "..."}]}\n'
        'If none, reply: {"contradictions": []}'
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if the model wrapped the JSON
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        conflicts = data.get("contradictions", [])
        if not conflicts:
            return False, ""

        notes = []
        for conflict in conflicts:
            idxs = conflict.get("indices", [])
            desc = conflict.get("description", "")
            if len(idxs) >= 2 and idxs[0] < len(results) and idxs[1] < len(results):
                notes.append(_resolve(results[idxs[0]], results[idxs[1]], desc))

        return True, " | ".join(notes)

    except Exception:
        # Contradiction check is best-effort; never block the answer
        return False, ""


# Answer generation

_SYSTEM_PROMPT = """You are a helpful technical support assistant for Supabase, \
a Postgres-based backend-as-a-service platform used by developers to build \
applications with a database, authentication, storage, and realtime features.

Your job is to answer developer questions clearly and accurately using only \
the context passages provided. Follow these guidelines:

- Be direct and practical. Developers want answers, not preamble.
- If the answer involves steps, use a numbered list.
- If the answer involves code, format it in a code block with the correct language tag.
- Always mention which source type the key information came from \
(Documentation, Blog, or Forum) so the developer knows how authoritative it is.
- If a forum thread or blog post is the source, note that official docs may \
have more up-to-date details.
- If the context passages do not contain enough information to answer \
confidently, say so clearly rather than guessing. It is better to say \
"I don't have enough information on that" than to give a wrong answer.
- If a conflict note is included in the context, surface it to the user \
and recommend they verify against the official Supabase documentation.
- Keep responses focused. Do not add unsolicited advice or tangents."""


def _build_context(results: list[RankedResult], contradiction_note: str) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        source_label = r.source_type.capitalize()
        parts.append(
            f"[{i}] {source_label} | {r.title}\n"
            f"{r.content[:600]}"
        )
    context = "\n\n---\n\n".join(parts)
    if contradiction_note:
        context += f"\n\n⚠ Conflict detected: {contradiction_note}"
    return context


def _generate_answer(
    query:             str,
    results:           list[RankedResult],
    contradiction_note: str,
    client:            OpenAI,
    model:             str,
) -> str:
    context = _build_context(results, contradiction_note)
    user_msg = f"Context:\n\n{context}\n\nQuestion: {query}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[Answer generation failed: {exc}]"


# Pipeline

class RAGPipeline:
    def __init__(self, retriever_top_k: int = 20, reranker_top_n: int = 5):
        self.retriever       = Retriever()
        self.reranker        = Reranker()
        self.client, self.model = _build_llm_client()
        self._retriever_top_k   = retriever_top_k
        self._reranker_top_n    = reranker_top_n

    def query(self, question: str) -> RAGResponse:
        # Stage 1: hybrid retrieval
        candidates = self.retriever.search(question, top_k=self._retriever_top_k)

        if not candidates:
            return RAGResponse(
                query=question,
                answer="No relevant information found in the knowledge base.",
                final_results=[],
                all_candidates=[],
                contradiction_detected=False,
                contradiction_note="",
                source_mix={},
            )

        # Stage 2: cross-encoder reranking + MMR
        results = self.reranker.rerank(
            question, candidates, top_n=self._reranker_top_n
        )

        # Stage 3: contradiction check (best-effort, never blocks answer)
        contradicted, contradiction_note = _check_contradictions(
            question, results, self.client, self.model
        )

        # Stage 4: answer generation
        answer = _generate_answer(
            question, results, contradiction_note, self.client, self.model
        )

        source_mix: dict[str, int] = {}
        for r in results:
            source_mix[r.source_type] = source_mix.get(r.source_type, 0) + 1

        return RAGResponse(
            query=question,
            answer=answer,
            final_results=results,
            all_candidates=candidates,
            contradiction_detected=contradicted,
            contradiction_note=contradiction_note,
            source_mix=source_mix,
        )


# Standalone test

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sample_queries = [
        "How do I enable row level security on a Supabase table?",
        "Why are my storage uploads failing with a permissions error?",
        "How do I track which users are online using Supabase Realtime?",
    ]

    print("Initialising pipeline...")
    try:
        pipeline = RAGPipeline()
    except EnvironmentError as e:
        print(f"\nConfiguration error: {e}")
        sys.exit(1)

    print(f"LLM: {pipeline.model}\n")

    for question in sample_queries:
        print("=" * 70)
        print(f"Q: {question}")
        print("-" * 70)

        response = pipeline.query(question)

        print(f"A: {response.answer}\n")

        print(f"Sources used ({sum(response.source_mix.values())}):")
        for r in response.final_results:
            print(f"  [{r.source_type:<15}] ce={r.ce_score:>7.3f}  {r.title!r}")

        print(f"Source mix: {response.source_mix}")

        if response.contradiction_detected:
            print(f"Contradiction: {response.contradiction_note}")
        print()
