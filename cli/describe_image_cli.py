#!/usr/bin/env python3
"""
Multimodal image analysis and query rewriting CLI.

This script takes an image file and a text query, then uses Gemini 2.0 Flash
to analyze the image and rewrite the query to improve search results from a
movie database. The rewritten query synthesizes visual and textual information
to focus on movie-specific details such as actors, scenes, and style.

Purpose:
- Leverage multimodal AI (image + text) for intelligent query enhancement
- Improve movie search relevance by incorporating visual context
- Provide token usage metrics for API monitoring

Usage:
    python cli/describe_image_cli.py --image <path> --query <text>

Arguments:
    --image: Path to the image file (required)
    --query: Text query to rewrite based on the image (required)

Output:
    Prints the rewritten query and total tokens used by the Gemini API.
    Exits with code 1 on file not found or API errors.

Dependencies:
    - GEMINI_API_KEY environment variable must be set for API access
    - google-genai library for Gemini API interaction
    - mimetypes module for MIME type detection
"""

import argparse
import mimetypes
import os
import sys

from google.genai import types

try:
    from cli.hybrid_search_cli import get_gemini_client
except ImportError:
    from hybrid_search_cli import get_gemini_client


def main():
    """
    Multimodal query rewriting CLI entry point using image analysis.

    Purpose:
    - Analyzes an image using Gemini 2.0 Flash multimodal capabilities.
    - Rewrites a text query by synthesizing visual and textual information.
    - Improves search query specificity by incorporating visual context from the image.

    Arguments:
    - --image (str): Required. Path to an image file (JPEG, PNG, WebP, GIF).
    - --query (str): Required. Text query to rewrite based on the provided image.

    Behavior:
    - Validates that the image file exists and is readable.
    - Detects MIME type automatically using mimetypes.guess_type(); defaults to image/jpeg.
    - Reads image file in binary mode.
    - Constructs a multimodal request with system prompt, image bytes, and query text.
    - Sends request to Gemini 2.0 Flash model via google-genai client.
    - Extracts and prints the rewritten query from the response.
    - Optionally displays token usage metrics if available in response metadata.

    Output Format:
    Rewritten query: <Synthesized query incorporating visual and textual information>
    Total tokens:    <Token count (optional)>

    Dependencies:
    - GEMINI_API_KEY environment variable must be set for Gemini API access.
    - google-genai library for Gemini 2.0 Flash multimodal API.
    - mimetypes module for MIME type detection (Python standard library).
    - get_gemini_client() from hybrid_search_cli for reusable client initialization.

    Exit Codes:
    - 0: Successfully rewrote and printed the query.
    - 1: Image file not found, cannot be read, or API call fails.

    Notes:
    - Supported image formats: JPEG, PNG, WebP, GIF (determined by file extension).
    - Unknown file extensions default to image/jpeg MIME type.
    - System prompt focuses on movie-specific details for Hoopla streaming context.
    - Query rewriting synthesizes visual details (scenes, actors, style) with text input.
    """
    parser = argparse.ArgumentParser(description="Rewrite a text query based on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", required=True, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    # Validate image path
    if not os.path.isfile(args.image):
        print(f"Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Determine MIME type, default to image/jpeg
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read image bytes
    try:
        with open(args.image, "rb") as f:
            img = f.read()
    except Exception as e:
        print(f"Failed to read image file: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up Gemini client using reusable helper
    client = get_gemini_client()

    # System prompt describing the task
    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve search results from a movie database. "
        "Make sure to:\n- Synthesize visual and textual information\n- Focus on movie-specific details (actors, scenes, style, etc.)\n- Return only the rewritten query, without any additional commentary"
    )

    # Build request parts: system prompt, image bytes, and stripped text query
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=parts,
        )
        # Print rewritten query and token usage
        print(f"Rewritten query: {response.text.strip()}")
        if getattr(response, "usage_metadata", None) is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    except Exception as e:
        print(f"Error generating rewritten query: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
