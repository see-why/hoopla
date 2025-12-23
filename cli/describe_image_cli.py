#!/usr/bin/env python3

import argparse
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from google import genai


def main():
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
            image_bytes = f.read()
    except Exception as e:
        print(f"Failed to read image file: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up Gemini client using API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # System prompt describing the task
    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve search results from a movie database. "
        "Make sure to:\n- Synthesize visual and textual information\n- Focus on movie-specific details (actors, scenes, style, etc.)\n- Return only the rewritten query, without any additional commentary"
    )

    # Build request parts: image + original query text
    inline_data = {
        "mime_type": mime,
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            system_instruction=system_prompt,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": inline_data},
                        {"text": args.query},
                    ],
                }
            ],
        )
        # Print only the rewritten query
        print(response.text)
    except Exception as e:
        print(f"Error generating rewritten query: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
