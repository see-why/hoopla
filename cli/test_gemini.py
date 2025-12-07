#!/usr/bin/env python3
"""
Test script for Google Gemini API integration.

This script demonstrates basic usage of the Gemini API by:
1. Loading the API key from environment variables
2. Creating a Gemini client
3. Sending a test prompt to the gemini-2.0-flash-001 model
4. Displaying the response and token usage statistics
"""
import os
from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    print("API key loaded successfully")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
        )
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        exit(1)

    print(response.text)
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
