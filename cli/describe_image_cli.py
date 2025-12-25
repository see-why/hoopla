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


def _detect_image_type_by_magic_bytes(file_path: str) -> str | None:
    """
    Detect image MIME type by reading file magic bytes (file signatures).
    
    This validates that the file is actually an image by checking the first few
    bytes for known image format signatures, regardless of file extension.
    
    Args:
        file_path: Path to the file to check
    
    Returns:
        MIME type string (e.g., "image/jpeg", "image/png") if detected, else None
    """
    # Magic bytes for common image formats
    # Format: (signature_bytes, mime_type)
    image_signatures = [
        (b'\xff\xd8\xff', "image/jpeg"),  # JPEG: FF D8 FF
        (b'\x89\x50\x4e\x47', "image/png"),  # PNG: 89 50 4E 47
        (b'\x47\x49\x46\x38', "image/gif"),  # GIF: 47 49 46 38
        (b'\x52\x49\x46\x46', "image/webp"),  # WebP: 52 49 46 46 (followed by WEBP)
        (b'\x42\x4d', "image/bmp"),  # BMP: 42 4D
        (b'\x00\x00\x01\x00', "image/x-icon"),  # ICO: 00 00 01 00
    ]
    
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)  # Read first 12 bytes to check signatures
            
            # Check for JPEG (requires special handling)
            if header.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            
            # Check for PNG
            if header.startswith(b'\x89\x50\x4e\x47'):
                return "image/png"
            
            # Check for GIF
            if header.startswith(b'\x47\x49\x46\x38'):
                return "image/gif"
            
            # Check for WebP (more specific check: RIFF...WEBP)
            if header.startswith(b'\x52\x49\x46\x46') and b'WEBP' in header:
                return "image/webp"
            
            # Check for BMP
            if header.startswith(b'\x42\x4d'):
                return "image/bmp"
            
            # Check for ICO
            if header.startswith(b'\x00\x00\x01\x00'):
                return "image/x-icon"
    except (IOError, OSError):
        # File read error
        return None
    
    return None


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
    - Validates file is actually an image by checking magic bytes (file signatures).
    - Detects MIME type automatically: first by magic bytes, then by file extension.
    - Raises error if MIME type cannot be determined or file is not a recognized image format.
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
    - get_gemini_client() from hybrid_search_cli for reusable client initialization (raises ValueError if API key not set).

    Exit Codes:
    - 0: Successfully rewrote and printed the query.
    - 1: Image file not found, cannot be read, file is not a valid image, API call fails, or MIME type cannot be determined.

    Notes:
    - Supported image formats: JPEG, PNG, WebP, GIF, BMP, ICO (detected by magic bytes).
    - File extension is checked if magic bytes don't identify the format.
    - Raises error if file extension is missing/unknown AND magic bytes don't match a known image format.
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

    # Read image bytes
    try:
        with open(args.image, "rb") as f:
            img = f.read()
    except Exception as e:
        print(f"Failed to read image file: {e}", file=sys.stderr)
        sys.exit(1)

    # Detect MIME type by magic bytes (more reliable than extension)
    mime = _detect_image_type_by_magic_bytes(args.image)
    
    # Fallback to extension-based detection if magic bytes didn't identify the format
    if not mime:
        mime, _ = mimetypes.guess_type(args.image)
    
    # Validate that we have a recognized image format
    if not mime:
        print(
            f"Error: Could not determine image format for '{args.image}'. "
            f"File may not be a valid image or has an unknown extension. "
            f"Supported formats: JPEG, PNG, WebP, GIF, BMP, ICO.",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Additional check: ensure the detected MIME type is an image format
    if not mime.startswith("image/"):
        print(
            f"Error: File '{args.image}' does not appear to be an image. "
            f"Detected type: {mime}. Supported formats: JPEG, PNG, WebP, GIF, BMP, ICO.",
            file=sys.stderr
        )
        sys.exit(1)

    # Set up Gemini client using reusable helper
    try:
        client = get_gemini_client()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

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
