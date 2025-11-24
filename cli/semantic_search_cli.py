#!/usr/bin/env python3

import argparse
import re


def semantic_chunk_sentences(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list:
    """Split `text` into sentence-based chunks.

    - Sentences are split using the regex "(?<=[.!?])\\s+" via `re.split`.
    - Each chunk contains up to `max_chunk_size` sentences.
    - Consecutive chunks overlap by `overlap` sentences (when overlap > 0).

    Returns a list of chunk strings (sentences joined with a single space).
    """
    if not isinstance(text, str):
        return []
    txt = text.strip()
    if not txt:
        return []

    # Split into sentences preserving punctuation at end of each sentence
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", txt) if s]

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")

    step = max_chunk_size - overlap
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i : i + max_chunk_size]
        if not chunk_sents:
            break
        chunks.append(" ".join(chunk_sents))
        if i + max_chunk_size >= len(sentences):
            break
        i += step

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify command: print loaded model information
    subparsers.add_parser("verify", help="Verify sentence-transformers model is loadable")
    # embed_text command: generate embedding for a single input text
    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for a single input text and print a short summary"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")
    # embedquery command: embed a query string and show first 5 dims + shape
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for a query string and print first 5 dimensions and shape"
    )
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")
    # chunk command: split text into word chunks (optional overlap)
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split input text into word-sized chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to split into chunks")
    chunk_parser.add_argument("--size", type=int, default=5, help="Number of words per chunk")
    # backward-compatible flag name used by tests
    chunk_parser.add_argument(
        "--chunk-size",
        dest="size",
        type=int,
        help=argparse.SUPPRESS,
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between consecutive chunks (must be < size)",
    )
    # semantic search command: query with embedding-based retrieval
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic embeddings"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")
    
    # semantic_chunk command: split text into chunks with optional overlap
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split input text into chunks with optional overlap (semantic chunking)"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per semantic chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between consecutive chunks (default: 0)",
    )
    # verify_embeddings command: build or load embeddings for the movie corpus
    subparsers.add_parser("verify_embeddings", help="Build or load movie embeddings and print their shape")

    args = parser.parse_args()

    match args.command:
        case "verify":
            # Import lazily so importing this module doesn't require the
            # sentence-transformers package unless the command is invoked.
            # Support two invocation styles:
            # - when running as a module from project root: `cli.lib...`
            # - when running the script directly (sys.path[0] == cli/):
            #   import from `lib...` instead.
            try:
                from cli.lib.semantic_search import verify_model
            except ImportError:
                from lib.semantic_search import verify_model

            verify_model()
        case "embed_text":
            try:
                from cli.lib.semantic_search import embed_text
            except ImportError:
                from lib.semantic_search import embed_text

            embed_text(args.text)
        case "embedquery":
            try:
                from cli.lib.semantic_search import embed_query_text
            except ImportError:
                from lib.semantic_search import embed_query_text

            embed_query_text(args.query)
        case "chunk":
            # Split text into word chunks, optionally with overlap
            text = (args.text or "").strip()
            size = args.size
            overlap = args.overlap

            if size <= 0:
                print("Error: --size must be a positive integer")
                return
            if overlap < 0:
                print("Error: --overlap must be non-negative")
                return
            if overlap >= size:
                print("Error: --overlap must be less than --size")
                return

            if not text:
                print("No text provided.")
                return

            # report chunking header (count characters in original text)
            print(f"Chunking {len(text)} characters")

            words = text.split()
            step = size - overlap
            chunks = []
            i = 0
            while i < len(words):
                chunk_words = words[i : i + size]
                chunks.append(" ".join(chunk_words))
                if i + size >= len(words):
                    break
                i += step

            # Print chunks in numbered format (one per line)
            for idx, c in enumerate(chunks, start=1):
                print(f"{idx}. {c}")
        case "verify_embeddings":
            try:
                from cli.lib.semantic_search import verify_embeddings
            except ImportError:
                from lib.semantic_search import verify_embeddings

            verify_embeddings()
        case "search":
            # Lazy imports to avoid requiring heavy deps at module import time
            try:
                from cli.lib.semantic_search import SemanticSearch, load_movies_dataset
            except ImportError:
                from lib.semantic_search import SemanticSearch, load_movies_dataset

            # load movies dataset using shared helper
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                import sys

                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)

            ss = SemanticSearch()
            # ensure embeddings exist (will build if missing)
            ss.load_or_create_embeddings(docs)

            results = ss.search(args.query, args.limit)

            # print formatted results
            if not results:
                print("No results found.")
            else:
                for rank, r in enumerate(results, start=1):
                    title = r.get("title", "<untitled>")
                    score = r.get("score", 0.0)
                    desc = (r.get("description") or "").strip()
                    # truncate description to 200 chars for CLI readability
                    if len(desc) > 200:
                        desc = desc[:200].rstrip() + "..."

                    print(f"{rank}. {title} (score: {score:.4f})")
                    if desc:
                        print(f"   {desc}\n")

        case "semantic_chunk":
            # Split text into semantic chunks using max chunk size and overlap
            text = (args.text or "").strip()
            # argparse converts --max-chunk-size to args.max_chunk_size
            max_size = int(args.max_chunk_size)
            overlap = int(args.overlap)

            try:
                chunks = semantic_chunk_sentences(text, max_size, overlap)
            except ValueError as exc:
                print(f"Error: {exc}")
                return

            print(f"Semantically chunking {len(text)} characters")
            for idx, c in enumerate(chunks, start=1):
                print(f"{idx}. {c}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
