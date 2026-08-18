"""Conservative cleanup for extracted academic text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page|halaman)?\s*[-–—]?\s*(?:\d+|[ivxlcdm]+)\s*[-–—]?\s*$",
    flags=re.IGNORECASE,
)
HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n\f]+")
EXCESS_BLANKS_RE = re.compile(r"\n{3,}")


def _normalise_line(line: str) -> str:
    return HORIZONTAL_SPACE_RE.sub(" ", line).strip()


def _repeated_page_margins(pages: list[list[str]]) -> set[str]:
    candidates: Counter[str] = Counter()
    for lines in pages:
        nonempty = [line for line in lines if line]
        if not nonempty:
            continue
        candidates.update({nonempty[0], nonempty[-1]})
    return {
        line
        for line, count in candidates.items()
        if count >= 2 and len(line) <= 160 and not PAGE_NUMBER_RE.fullmatch(line)
    }


def clean_text(text: str) -> str:
    """Normalize text and remove page numbers plus simple repeated margins."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        char
        for char in normalized
        if char in "\n\t\f" or unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    )
    pages = [[_normalise_line(line) for line in page.split("\n")] for page in normalized.split("\f")]
    repeated_margins = _repeated_page_margins(pages)

    cleaned_pages: list[str] = []
    for lines in pages:
        kept = [
            line
            for line in lines
            if line not in repeated_margins and not PAGE_NUMBER_RE.fullmatch(line)
        ]
        cleaned_pages.append("\n".join(kept).strip())

    joined = "\n\n".join(page for page in cleaned_pages if page)
    return EXCESS_BLANKS_RE.sub("\n\n", joined).strip()

