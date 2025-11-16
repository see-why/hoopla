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
                from cli.lib.semantic_search import SemanticSearch
            except ImportError:
                from lib.semantic_search import SemanticSearch

            # load movies dataset
            from pathlib import Path
            import json

            data_dir = Path(__file__).resolve().parents[1] / "data"
            movies_path = data_dir / "movies.json"
            if not movies_path.exists():
                movies_path = data_dir / "movies 2.json"

            try:
                loaded = json.loads(movies_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "movies" in loaded:
                    docs = loaded["movies"]
                elif isinstance(loaded, list):
                    docs = loaded
                else:
                    docs = []
            except (OSError, json.JSONDecodeError) as exc:
                # Provide immediate feedback to the CLI user and avoid
                # silently falling back to an empty corpus which can be
                # confusing. Print to stderr so scripts can still parse
                # normal output.
                import sys

                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                docs = []

            ss = SemanticSearch()
            # ensure embeddings exist (will build if missing)
            ss.load_or_create_embeddings(docs)

            results = ss.search(args.query, args.limit)

            # print formatted results
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
