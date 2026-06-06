"""Long-context RAG bridge interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagChunk:
    text: str
    source: str
    score: float = 0.0


def chunk_text(text: str, chunk_tokens: int = 4096, overlap_tokens: int = 256) -> list[str]:
    words = text.split()
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must be greater than overlap_tokens")
    chunks: list[str] = []
    step = chunk_tokens - overlap_tokens
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_tokens]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def build_augmented_prompt(query: str, chunks: list[RagChunk]) -> str:
    context = "\n\n".join(f"[{chunk.source} score={chunk.score:.3f}]\n{chunk.text}" for chunk in chunks)
    return f"Use the context to answer.\n\n{context}\n\nQuestion: {query}\nAnswer:"
