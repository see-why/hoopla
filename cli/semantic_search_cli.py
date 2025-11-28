#!/usr/bin/env python3

import argparse
import re


def semantic_chunk_sentences(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list:
    """Split `text` into sentence-based chunks.

    - Strips leading and trailing whitespace from input text
    - Sentences are split using the regex "(?<=[.!?])\\s+" via `re.split`.
    - If only one sentence without ending punctuation, treats whole text as one sentence
    - Each chunk contains up to `max_chunk_size` sentences.
    - Consecutive chunks overlap by `overlap` sentences (when overlap > 0).
    - Strips whitespace from each sentence and only includes non-empty sentences

    Returns a list of chunk strings (sentences joined with a single space).
    """
    if not isinstance(text, str):
        return []
    
    # Strip leading and trailing whitespace from input
    txt = text.strip()
    if not txt:
        return []

    # Validate parameters before processing
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")

    # Split into sentences preserving punctuation at end of each sentence
    raw_sentences = re.split(r"(?<=[.!?])\s+", txt)
    
    # Strip whitespace from each sentence and filter out empty ones
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    # If only one sentence and it doesn't end with punctuation, treat whole text as one sentence
    if len(sentences) == 1 and sentences[0] and not sentences[0][-1] in '.!?':
        # The whole text is one sentence without ending punctuation
        sentences = [txt]
    elif len(sentences) == 0:
        # Edge case: if splitting resulted in no sentences but we have text,
        # treat the whole text as one sentence
        sentences = [txt]

    step = max_chunk_size - overlap
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i : i + max_chunk_size]
        if not chunk_sents:
            break
        # Join sentences and strip any extra whitespace
        chunk_text = " ".join(chunk_sents).strip()
        # Only add non-empty chunks
        if chunk_text:
            chunks.append(chunk_text)
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
    # embed_chunks command: build or load chunk embeddings and report
    subparsers.add_parser("embed_chunks", help="Build or load chunk embeddings and print a summary")
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
    
    # search_chunked command: query using chunked embeddings
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked semantic embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")
    
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
        case "embed_chunks":
            try:
                from cli.lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset
            except ImportError:
                from lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset

            docs, exc, movies_path = load_movies_dataset()
            if exc:
                import sys

                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)

            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(docs)

            # embeddings is a numpy array
            try:
                count = len(embeddings)
            except Exception:
                count = 0

            print(f"Generated {count} chunked embeddings")
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
                sys.exit(1)

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

        case "search_chunked":
            # Lazy imports to avoid requiring heavy deps at module import time
            try:
                from cli.lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset
            except ImportError:
                from lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset

            # load movies dataset using shared helper
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                import sys

                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)

            css = ChunkedSemanticSearch()
            # ensure chunk embeddings exist (will build if missing)
            css.load_or_create_chunk_embeddings(docs)

            results = css.search_chunks(args.query, args.limit)

            # print formatted results
            if not results:
                print("No results found.")
            else:
                for i, r in enumerate(results, start=1):
                    title = r.get("title", "<untitled>")
                    score = r.get("score", 0.0)
                    desc = (r.get("description") or "").strip()
                    # truncate description to 200 chars for CLI readability (consistent with search command)
                    if len(desc) > 200:
                        desc = desc[:200].rstrip() + "..."

                    print(f"\n{i}. {title} (score: {score:.4f})")
                    print(f"   {desc}")

        case "semantic_chunk":
            # Split text into semantic chunks using max chunk size and overlap
            text = (args.text or "").strip()
            # argparse converts --max-chunk-size to args.max_chunk_size
            max_size = int(args.max_chunk_size)
            overlap = int(args.overlap)

            if not text:
                print("No text provided.")
                return

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
