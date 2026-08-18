"""Run a local end-to-end PHASE 3 demo using the synthetic sample documents."""

from __future__ import annotations

from pathlib import Path

from app.core.chunker import split_paragraphs
from app.core.cleaner import clean_text
from app.core.extractor import extract_text
from app.core.indexer import SourceIndexer
from app.services.similarity_engine import analyze_paragraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _progress(current: int, total: int, source_path: Path) -> None:
    print(f"[{current}/{total}] {source_path.name}")


def main() -> int:
    references = PROJECT_ROOT / "sample_documents" / "references"
    document = PROJECT_ROOT / "sample_documents" / "example.txt"
    demo_index = PROJECT_ROOT / "data" / "demo-index"

    print("Indexing synthetic sources...\n")
    indexer = SourceIndexer(index_dir=demo_index)
    result = indexer.rebuild_index(references, progress_callback=_progress)
    print(
        f"\nIndexed {result.chunks_indexed} chunks "
        f"({result.embedding_dimension}-dimensional embeddings).\n"
    )

    paragraphs = split_paragraphs(clean_text(extract_text(document)))
    for number, paragraph in enumerate(paragraphs, start=1):
        analysis = analyze_paragraph(paragraph, indexer, top_k=5, min_score=0.0)
        print(f"Paragraph {number}: {paragraph}")
        if not analysis.matches:
            print("  No candidates.\n")
            continue
        best = analysis.matches[0]
        print(f"  Matched source: {best.source_file}")
        print(f"  Matched text: {best.matched_text}")
        print(f"  Lexical similarity: {best.lexical_similarity:.1%}")
        print(f"  Semantic similarity: {best.semantic_similarity:.1%}")
        print(f"  N-gram overlap: {best.ngram_overlap:.1%}")
        print(f"  Combined score: {best.final_score:.1%}")
        print(f"  Risk: {best.risk} SIMILARITY")
        print(f"  Reason: {best.reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

