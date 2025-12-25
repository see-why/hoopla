#!/usr/bin/env python3
"""
Multimodal search CLI for image embedding operations.

This script provides command-line access to multimodal search capabilities,
including image embedding generation and verification using CLIP models.

Commands:
    verify_image_embedding: Load an image and generate its CLIP embedding,
                           displaying the embedding dimensions.

Usage:
    python cli/multimodal_search_cli.py verify_image_embedding <image_path>
"""

import argparse
import sys

try:
    from cli.lib.multimodal_search import verify_image_embedding
except ImportError:
    from lib.multimodal_search import verify_image_embedding


def main():
    """
    Multimodal search CLI entry point.

    Purpose:
    - Provides command-line interface for image embedding operations.
    - Supports CLIP model-based image analysis and embedding generation.
    - Enables verification of image embedding functionality.

    Arguments:
    - image_path (str): Path to an image file for embedding generation.

    Behavior:
    - Parses command-line arguments to extract image path.
    - Validates that the image file exists and is readable.
    - Generates embedding using CLIP model via MultimodalSearch.
    - Displays embedding dimensions in human-readable format.
    - Exits with error code 1 on file not found or API failures.

    Output Format:
    Embedding shape: <number> dimensions

    Dependencies:
    - PIL (Pillow) for image file loading.
    - SentenceTransformer for CLIP model access.
    - MultimodalSearch from cli.lib.multimodal_search for embedding generation.

    Exit Codes:
    - 0: Successfully generated and displayed embedding information.
    - 1: Image file not found, cannot be read, or embedding generation fails.

    Notes:
    - Supported image formats: JPEG, PNG, WebP, GIF, BMP, ICO.
    - CLIP model (default: clip-ViT-B-32) generates 512-dimensional embeddings.
    - Embedding generation may take several seconds depending on model size and hardware.
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
    else:
        # No command specified
        parser.print_help()


if __name__ == "__main__":
    main()
