"""Needle-in-haystack prompt generation for long-context validation."""

from __future__ import annotations


def build_needle_prompt(context_tokens: int, needle: str, filler: str = "VOID") -> str:
    if context_tokens < 32:
        raise ValueError("context_tokens must be at least 32")
    left = " ".join([filler] * (context_tokens // 2))
    right = " ".join([filler] * (context_tokens // 2))
    return f"{left}\nThe secret needle is: {needle}\n{right}\nWhat is the secret needle?"


def score_needle_answer(answer: str, needle: str) -> bool:
    return needle.strip().lower() in answer.strip().lower()
