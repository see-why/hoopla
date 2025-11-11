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
        case "verify_embeddings":
            try:
                from cli.lib.semantic_search import verify_embeddings
            except ImportError:
                from lib.semantic_search import verify_embeddings

            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
