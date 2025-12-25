#!/usr/bin/env python3
"""
Multimodal search CLI for image embedding operations.

This script provides command-line access to multimodal search capabilities,
including image embedding generation and verification using CLIP models.

Commands:
    verify_image_embedding: Load an image and generate its CLIP embedding,
                           displaying the embedding dimensions.
    image_search: Search for movies similar to a query image using CLIP embeddings.

Usage:
    python cli/multimodal_search_cli.py verify_image_embedding <image_path>
    python cli/multimodal_search_cli.py image_search <image_path>
"""

import argparse
import sys

try:
    from cli.lib.multimodal_search import verify_image_embedding, image_search_command
except ImportError:
    from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    """
    Parse CLI arguments and execute the requested multimodal search command.
    
    Supported commands: verify_image_embedding, image_search
    """
    parser = argparse.ArgumentParser(
        description="Multimodal search CLI for image embedding operations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify_image_embedding command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Load an image and generate its CLIP embedding"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to embed"
    )

    # image_search command
    search_parser = subparsers.add_parser(
        "image_search",
        help="Search for movies similar to a query image"
    )
    search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the query image file"
    )

    args = parser.parse_args()

    # Handle verify_image_embedding command
    if args.command == "verify_image_embedding":
        try:
            verify_image_embedding(args.image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error generating image embedding: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Handle image_search command
    elif args.command == "image_search":
        try:
            results = image_search_command(args.image_path)
            for result in results:
                print(f"{result['title']} (similarity: {result['similarity']:.3f})")
        except FileNotFoundError:
            print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error performing image search: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        # No command specified
        parser.print_help()


if __name__ == "__main__":
    main()
