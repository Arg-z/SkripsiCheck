"""Paragraph and sentence segmentation with no external language service."""

from __future__ import annotations

import re

PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def split_paragraphs(text: str, min_characters: int = 1) -> list[str]:
    """Split cleaned text on blank lines and discard tiny empty-like chunks."""

    if min_characters < 1:
        raise ValueError("min_characters must be at least 1")
    paragraphs = [" ".join(block.split()) for block in PARAGRAPH_BREAK_RE.split(text.strip())]
    return [paragraph for paragraph in paragraphs if len(paragraph) >= min_characters]


def split_sentences(text: str, min_characters: int = 1) -> list[str]:
    """Split text at common sentence boundaries for Indonesian and English."""

    if min_characters < 1:
        raise ValueError("min_characters must be at least 1")
    sentences = [sentence.strip() for sentence in SENTENCE_BREAK_RE.split(" ".join(text.split()))]
    return [sentence for sentence in sentences if len(sentence) >= min_characters]


def chunk_document(text: str) -> list[tuple[str, list[str]]]:
    """Return each paragraph together with its sentences."""

    return [(paragraph, split_sentences(paragraph)) for paragraph in split_paragraphs(text)]

