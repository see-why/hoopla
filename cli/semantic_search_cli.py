#!/usr/bin/env python3

import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify command: print loaded model information
    verify_parser = subparsers.add_parser("verify", help="Verify sentence-transformers model is loadable")

    args = parser.parse_args()

    match args.command:
        case "verify":
            # Import lazily so importing this module doesn't require the
            # sentence-transformers package unless the command is invoked.
            from cli.lib.semantic_search import verify_model

            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
