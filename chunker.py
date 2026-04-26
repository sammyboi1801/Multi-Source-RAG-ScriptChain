"""
chunker.py - source-specific chunking strategies

Each source type gets a strategy that matches its structure:
  - Docs:   heading-based splits (## / ###) respecting code fences
  - Blogs:  heading-based splits + JSX/HTML cleanup (blogs are MDX, same format as docs)
  - Forums: question + first reply as primary chunk; remaining replies as secondary chunks

Sliding window was ruled out for blogs after inspecting the data - every post has
between 5 and 20 section headings, making word-window splits an actively bad choice.

Run standalone to verify chunk counts and inspect samples:
    python chunker.py
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    source_type: str    # documentation | blog | forum
    source_id: str
    source_url: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }


# Shared helpers

def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Splits markdown into (heading, body) pairs on ## and ### boundaries.
    Skips heading detection while inside a code fence so code samples
    that happen to contain # characters are never split incorrectly.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    in_fence = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence

        if not in_fence and re.match(r"^#{2,3}\s", line):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = re.sub(r"^#{2,3}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_heading, body))

    return sections


def _strip_html(text: str) -> str:
    """
    Removes HTML/JSX tags left behind by the MDX scraper.

    Two-pass approach:
      1. DOTALL pass - wipes known multi-line embed tags (iframe, video, source,
         div wrappers) that span 4-6 lines and can't be caught line-by-line.
      2. Line-by-line pass - removes any remaining single-line tags while
         skipping content inside code fences.
    """
    # Pass 1: multi-line JSX embeds (iframe, video players, source elements)
    # These appear as wrapper blocks around media and never contain prose text.
    multiline_tags = r"iframe|video|source|div|small"
    text = re.sub(
        rf"<({multiline_tags})\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Self-closing variants like <source ... /> and unclosed openers <iframe ...>
    text = re.sub(
        rf"<({multiline_tags})\b[^>]*?>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Pass 2: remaining single-line tags, respecting code fences
    lines = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence:
            line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Documentation - heading-based hierarchical

def chunk_docs(docs: list[dict]) -> list[Chunk]:
    """
    Splits each doc on ## / ### headings. Sections under 60 chars are skipped -
    in practice these are heading-only lines with no body content.
    """
    chunks: list[Chunk] = []

    for doc in docs:
        sections = _split_by_headings(doc.get("content", ""))

        for i, (heading, body) in enumerate(sections):
            if len(body) < 60:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc['id']}__s{i}",
                source_type="documentation",
                source_id=doc["id"],
                source_url=doc.get("url", ""),
                title=heading or doc.get("title", doc["id"]),
                content=body,
                metadata={
                    "doc_title": doc.get("title", ""),
                    "section_heading": heading,
                    "section_index": i,
                },
            ))

    return chunks


# Blogs - heading-based + HTML cleanup

def chunk_blogs(blogs: list[dict]) -> list[Chunk]:
    """
    All 25 scraped blog posts are MDX with 5–20 section headings each, so
    heading-based splitting is strictly better than sliding windows here.

    JSX/HTML remnants from the scraper (<div className>, <video>, <source>)
    are stripped before splitting so they don't pollute chunk content.

    Minimum section body: 100 chars. Narrative sections are longer than
    doc feature descriptions, so short blobs here indicate noise.
    """
    chunks: list[Chunk] = []

    for post in blogs:
        content = _strip_html(post.get("content", "").strip())
        if not content:
            continue

        sections = _split_by_headings(content)

        # If no headings found, treat the whole post as a single chunk
        if not sections:
            sections = [("", content)]

        for i, (heading, body) in enumerate(sections):
            if len(body) < 100:
                continue
            chunks.append(Chunk(
                chunk_id=f"{post['id']}__s{i}",
                source_type="blog",
                source_id=post["id"],
                source_url=post.get("url", ""),
                title=heading or post.get("title", post["id"]),
                content=body,
                metadata={
                    "blog_title": post.get("title", ""),
                    "date": post.get("date", ""),
                    "section_heading": heading,
                    "section_index": i,
                    "total_sections": len(sections),
                },
            ))

    return chunks


# Forums - thread-level + post-level

_GITHUB_TEMPLATE_HEADERS = re.compile(
    r"^#\s+(Bug report|Feature request|Question|Documentation|Other)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_forum_question(text: str) -> str:
    """
    GitHub issue templates prefix every question with a structured H1 header
    like '# Bug report'. Strip those so they don't dominate chunk embeddings.
    Also strips the standard 'describe the bug / expected behavior' template
    labels that appear as bold section markers.
    """
    text = _GITHUB_TEMPLATE_HEADERS.sub("", text)
    # Strip bold section labels like **Describe the bug**
    text = re.sub(r"^\*\*[^*]{3,60}\*\*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_forums(forums: list[dict]) -> list[Chunk]:
    """
    Each thread produces:
      - One primary chunk: cleaned question + first reply (highest signal pair)
      - One secondary chunk per additional reply, prefixed with the question
        so retrieval always knows what problem was being discussed

    No accepted-answer field exists in the scraped GitHub Issues data, so the
    first comment is used as the top reply (scraper fetches most-commented first).
    """
    chunks: list[Chunk] = []

    for thread in forums:
        raw_question = (thread.get("question") or "").strip()
        if not raw_question:
            continue

        question = _clean_forum_question(raw_question)
        replies = thread.get("comments", [])
        top_reply = replies[0].get("body", "").strip() if replies else ""

        primary_content = f"Q: {question}"
        if top_reply:
            primary_content += f"\n\nA: {top_reply}"

        chunks.append(Chunk(
            chunk_id=f"{thread['id']}__primary",
            source_type="forum",
            source_id=thread["id"],
            source_url=thread.get("url", ""),
            title=thread.get("title", ""),
            content=primary_content,
            metadata={
                "thread_title": thread.get("title", ""),
                "state": thread.get("state", ""),
                "labels": thread.get("labels", []),
                "reply_count": thread.get("comment_count", 0),
                "is_primary": True,
            },
        ))

        for j, reply in enumerate(replies[1:], start=1):
            body = reply.get("body", "").strip()
            if len(body) < 40:
                continue
            chunks.append(Chunk(
                chunk_id=f"{thread['id']}__r{j}",
                source_type="forum",
                source_id=thread["id"],
                source_url=thread.get("url", ""),
                title=thread.get("title", ""),
                content=f"Context: {question[:250]}\n\nReply: {body}",
                metadata={
                    "thread_title": thread.get("title", ""),
                    "author": reply.get("author", ""),
                    "created_at": reply.get("created_at", ""),
                    "reply_index": j,
                    "is_primary": False,
                },
            ))

    return chunks


# Standalone test

if __name__ == "__main__":
    data_root = Path("data")

    def load(path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    docs   = load(data_root / "docs/supabase_docs.json")
    blogs  = load(data_root / "blogs/supabase_blogs.json")
    forums = load(data_root / "forums/supabase_forums.json")

    doc_chunks   = chunk_docs(docs)
    blog_chunks  = chunk_blogs(blogs)
    forum_chunks = chunk_forums(forums)
    all_chunks   = doc_chunks + blog_chunks + forum_chunks

    def _avg(chunks: list[Chunk]) -> int:
        return sum(len(c.content) for c in chunks) // max(len(chunks), 1)

    print(f"{'Source':<10} {'Chunks':>6}  {'Avg chars':>10}")
    print("-" * 32)
    print(f"{'docs':<10} {len(doc_chunks):>6}  {_avg(doc_chunks):>10}")
    print(f"{'blogs':<10} {len(blog_chunks):>6}  {_avg(blog_chunks):>10}")
    print(f"{'forums':<10} {len(forum_chunks):>6}  {_avg(forum_chunks):>10}")
    print("-" * 32)
    print(f"{'total':<10} {len(all_chunks):>6}")

    # Spot-check one chunk from each source
    for label, sample in [("Doc", doc_chunks[0]), ("Blog", blog_chunks[0]), ("Forum", forum_chunks[0])]:
        print(f"\n-- {label} sample ({sample.chunk_id}) ----------")
        print(f"title : {sample.title!r}")
        print(f"meta  : {sample.metadata}")
        print(f"text  : {sample.content[:300]!r}")

    # Verify JSX render components are gone from blogs
    # (className/autoPlay/src attributes are the tell - not SQL <schema> placeholders)
    import re as _re
    jsx_remaining = sum(
        1 for c in blog_chunks
        if _re.search(r"<[a-zA-Z][^>]*(className|autoPlay|playsInline|src=.https)", c.content)
    )
    print(f"\nBlog chunks with residual JSX components: {jsx_remaining}")

    # Verify # Bug report stripped from forums
    header_remaining = sum(
        1 for c in forum_chunks
        if "# Bug report" in c.content or "# Feature request" in c.content
    )
    print(f"Forum chunks with template headers remaining: {header_remaining}")
