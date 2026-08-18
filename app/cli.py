"""Minimal command-line interface for PHASE 3 source indexing and search."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app import __version__
from app.core.indexer import IndexBuildResult, SourceIndexError, SourceIndexer


def _progress(current: int, total: int, source_path: Path) -> None:
    print(f"[{current}/{total}] {source_path.name}")


def _print_build_result(result: IndexBuildResult) -> None:
    print()
    print(f"Sources indexed: {result.source_count}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    if result.skipped_files:
        print(f"Sources skipped: {len(result.skipped_files)}")
        for issue in result.skipped_files:
            print(f"  - {issue.source_file}: {issue.reason}")
    print()
    print("Index saved:")
    print(result.index_path)
    print(f"Metadata: {result.metadata_path}")
    print(f"Index info: {result.index_info_path}")


def _run_index(args: argparse.Namespace, *, rebuild: bool) -> int:
    action = "Rebuilding" if rebuild else "Indexing"
    print(f"{action} sources...\n")
    indexer = SourceIndexer(index_dir=args.index_dir)
    method = indexer.rebuild_index if rebuild else indexer.build_index
    result = method(args.sources, progress_callback=_progress)
    _print_build_result(result)
    return 0


def _run_search(args: argparse.Namespace) -> int:
    indexer = SourceIndexer(index_dir=args.index_dir)
    matches = indexer.search_similar_chunks(
        args.query, top_k=args.top_k, min_score=args.min_score
    )
    if not matches:
        print("No semantic matches found.")
        return 0
    print(f"Semantic matches for: {args.query}\n")
    for position, match in enumerate(matches, start=1):
        page = f", page {match.page}" if match.page is not None else ""
        print(
            f"{position}. {match.source_file}{page} "
            f"— semantic similarity {match.semantic_score:.1%}"
        )
        print(f"   chunk_id: {match.chunk_id}")
        print(f"   {match.text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skripsicheck",
        description="Local-first semantic source indexing for SkripsiCheck.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("index", "Build a full FAISS index from a source folder."),
        ("rebuild-index", "Rebuild the complete FAISS source index."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("sources", type=Path, help="Folder containing PDF/DOCX/TXT sources.")
        command_parser.add_argument(
            "--index-dir",
            type=Path,
            default=None,
            help="Index output directory (default: SKRIPSICHECK_INDEX_DIR or data/index).",
        )
        command_parser.set_defaults(handler=_run_index, rebuild=command == "rebuild-index")

    search_parser = subparsers.add_parser(
        "search", help="Search the existing index with one paragraph or sentence."
    )
    search_parser.add_argument("query", help="Text whose nearest source chunks should be found.")
    search_parser.add_argument("--top-k", type=int, default=None, help="Maximum number of matches.")
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum semantic score (default: SKRIPSICHECK_MIN_SEMANTIC_SCORE).",
    )
    search_parser.add_argument("--index-dir", type=Path, default=None)
    search_parser.set_defaults(handler=_run_search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a PHASE 3 command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"index", "rebuild-index"}:
            return int(args.handler(args, rebuild=args.rebuild))
        return int(args.handler(args))
    except (SourceIndexError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
