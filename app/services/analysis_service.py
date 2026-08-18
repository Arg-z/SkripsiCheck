"""Document-level orchestration over extraction, retrieval, and scoring."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import SETTINGS
from app.core.chunker import split_paragraphs
from app.core.cleaner import clean_text
from app.core.extractor import DocumentExtractionError, extract_text
from app.database.repository import AnalysisRecord, Repository
from app.services.similarity_engine import CandidateRetriever, ScoredMatch, analyze_paragraph

DISCLAIMER = (
    "SkripsiCheck is a text similarity analysis tool, not an automated plagiarism "
    "verdict system. Similarity does not necessarily indicate plagiarism. Results "
    "should be reviewed manually and interpreted according to academic citation standards."
)


class AnalysisServiceError(RuntimeError):
    pass


class DocumentNotFoundError(AnalysisServiceError):
    pass


class EmptyDocumentError(AnalysisServiceError):
    pass


class AnalysisService:
    def __init__(
        self,
        repository: Repository,
        retriever_factory: Callable[[], CandidateRetriever],
    ) -> None:
        self.repository = repository
        self.retriever_factory = retriever_factory

    def analyze(
        self,
        document_id: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnalysisRecord:
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} was not found.")
        try:
            paragraphs = split_paragraphs(
                clean_text(extract_text(Path(document.stored_path)))
            )
        except DocumentExtractionError as exc:
            raise AnalysisServiceError(f"Could not read the stored document: {exc}") from exc
        if not paragraphs:
            raise EmptyDocumentError("The document produced zero text paragraphs.")

        retriever = self.retriever_factory()
        paragraph_payloads: list[dict[str, Any]] = []
        unique_contributions: dict[tuple[str, str], float] = {}
        matched_paragraphs = 0
        total_words = 0

        for number, paragraph in enumerate(paragraphs, start=1):
            word_count = len(paragraph.split())
            total_words += word_count
            result = analyze_paragraph(
                paragraph,
                retriever,
                top_k=top_k,
                min_score=min_score,
            )
            matches = [self._match_payload(match) for match in result.matches]
            if result.matches:
                matched_paragraphs += 1
                best = result.matches[0]
                key = (best.source_file.casefold(), best.chunk_id)
                contribution = word_count * best.final_score
                unique_contributions[key] = max(
                    contribution, unique_contributions.get(key, 0.0)
                )
            paragraph_payloads.append(
                {
                    "number": number,
                    "text": paragraph,
                    "word_count": word_count,
                    "candidates_retrieved": result.candidates_retrieved,
                    "matches": matches,
                }
            )

        overall = sum(unique_contributions.values()) / max(total_words, 1)
        overall = min(1.0, max(0.0, overall))
        if not math.isfinite(overall):
            raise AnalysisServiceError("Overall similarity calculation was not finite.")
        analysis_id = str(uuid4())
        report = {
            "analysis_id": analysis_id,
            "document": {"id": document.id, "filename": document.original_filename},
            "overall_similarity": overall,
            "total_paragraphs": len(paragraphs),
            "matched_paragraphs": matched_paragraphs,
            "paragraphs": paragraph_payloads,
            "methodology": {
                "overall": (
                    "Word-weighted best match per paragraph, with repeated source chunks "
                    "counted once using their strongest contribution."
                ),
                "weights": {
                    "lexical": SETTINGS.lexical_weight,
                    "semantic": SETTINGS.semantic_weight,
                    "ngram": SETTINGS.ngram_weight,
                },
            },
            "disclaimer": DISCLAIMER,
        }
        return self.repository.add_analysis(
            analysis_id=analysis_id,
            document_id=document.id,
            overall_similarity=overall,
            total_paragraphs=len(paragraphs),
            matched_paragraphs=matched_paragraphs,
            result=report,
        )

    @staticmethod
    def _match_payload(match: ScoredMatch) -> dict[str, Any]:
        payload = asdict(match)
        payload.pop("paragraph", None)
        # Preserve the source filename while avoiding internal filesystem disclosure.
        payload["source_path"] = None
        return payload
