#!/usr/bin/env python3

import argparse


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
    # semantic search command: query with embedding-based retrieval
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic embeddings"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")
    # chunk command: split a long text into chunks
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into chunks preserving word boundaries"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=200, help="Maximum chunk size in characters")
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

        case "chunk":
            # Chunk by grouping N words together, where N is --chunk-size.
            text = args.text or ""
            n = args.chunk_size if getattr(args, "chunk_size", None) is not None else 200

            words = text.split()
            if n <= 0:
                print("Chunk size must be a positive integer.")
                return

            chunks = []
            for i in range(0, len(words), n):
                chunk_words = words[i : i + n]
                chunks.append(" ".join(chunk_words))

            total_chars = len(text)
            print(f"Chunking {total_chars} characters")
            if not chunks:
                print("No chunks produced.")
            else:
                for idx, c in enumerate(chunks, start=1):
                    print(f"{idx}. {c}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
