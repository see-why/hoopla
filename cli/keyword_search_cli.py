#!/usr/bin/env python3

import argparse
import json
import re
import string
from nltk.stem import PorterStemmer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # Initialize stemmer for token normalization
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            # Basic search: load movies.json and find titles containing the query
            print(f"Searching for: {args.query}")

            data_path = Path(__file__).resolve().parents[1] / "data" / "movies.json"
            results = []

            try:
                with open(data_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except FileNotFoundError:
                print(f"Movies data file not found at: {data_path}")
                return
            except json.JSONDecodeError:
                # Fallback: attempt to extract movie-like blocks and fields using regex
                try:
                    text = data_path.read_text(encoding="utf-8")

                    # Find the start of the movies array
                    arr_start = text.find("movies")
                    if arr_start == -1:
                        raise ValueError("movies array not found")

                    # Find the opening bracket for the array
                    bracket_start = text.find("[", arr_start)
                    if bracket_start == -1:
                        raise ValueError("movies array bracket not found")

                    # Scan forward to extract balanced { ... } object blocks
                    blocks = []
                    i = bracket_start + 1
                    n = len(text)
                    while i < n:
                        # Skip whitespace and commas
                        if text[i].isspace() or text[i] == ",":
                            i += 1
                            continue

                        if text[i] != "{":
                            # Stop at end of array
                            if text[i] == "]":
                                break
                            i += 1
                            continue

                        # Found an object start; scan until balanced
                        depth = 0
                        start_idx = i
                        i += 1
                        in_string = False
                        escape = False
                        while i < n:
                            ch = text[i]
                            if in_string:
                                if escape:
                                    escape = False
                                elif ch == "\\":
                                    escape = True
                                elif ch == '"':
                                    in_string = False
                            else:
                                if ch == '"':
                                    in_string = True
                                elif ch == '{':
                                    depth += 1
                                elif ch == '}':
                                    if depth == 0:
                                        # include the closing brace
                                        i += 1
                                        break
                                    depth -= 1
                            i += 1

                        blk = text[start_idx:i]
                        # strip surrounding braces for compatibility with earlier code
                        if blk.startswith("{") and blk.endswith("}"):
                            blk_inner = blk[1:-1]
                        else:
                            blk_inner = blk

                        blocks.append(blk_inner)

                    data = {"movies": []}
                    for blk in blocks:
                        # Extract id
                        m_id = re.search(r"\bid\s*:\s*(\d+)", blk)
                        try:
                            movie_id = int(m_id.group(1)) if m_id else None
                        except (ValueError, TypeError):
                            movie_id = None

                        # Extract title (non-greedy up to a closing quote)
                        m_title = re.search(r'title\s*:\s*"(.*?)"', blk, re.S)
                        title = m_title.group(1).strip() if m_title else ""

                        # Extract description if present
                        m_desc = re.search(r'description\s*:\s*"(.*?)"\s*(?:\s*,|\s*\}|$)', blk, re.S)
                        description = m_desc.group(1).strip() if m_desc else None

                        movie = {"id": movie_id, "title": title}
                        if description is not None:
                            movie["description"] = description

                        # Only add if we found a title
                        if title:
                            data["movies"].append(movie)
                except (ValueError, OSError, re.error, TypeError):
                    print(f"Failed to decode JSON from: {data_path}")
                    return

            movies = data.get("movies", []) if isinstance(data, dict) else []

            # Create a translation table that maps punctuation to spaces so
            # removing punctuation doesn't join words together (e.g. "Star-Wars"
            # -> "star wars"). Collapse consecutive spaces after translate.
            _punct_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

            # Load stop words from data/stopwords.txt (one per line) if it
            # exists. Use a set for fast membership tests.
            sw_path = Path(__file__).resolve().parents[1] / "data" / "stopwords.txt"
            stopwords = set()
            if sw_path.exists():
                try:
                    sw_text = sw_path.read_text(encoding="utf-8")
                    # splitlines keeps each stop word per line
                    stopwords = {w.strip().casefold() for w in sw_text.splitlines() if w.strip()}
                except OSError:
                    # If reading fails, continue with an empty set
                    stopwords = set()

            q_raw = args.query.strip()
            # If the query is empty after stripping whitespace, treat as no-op
            # and return no results to avoid matching every title (since
            # '' in haystack is always True).
            if not q_raw:
                print("No results found.")
                return
            q_lower = q_raw.casefold()
            # Normalize and tokenize the query: map punctuation to spaces,
            # collapse whitespace, and split into tokens.
            q_norm = " ".join(q_lower.translate(_punct_trans).split())
            q_tokens = [t for t in q_norm.split() if t and t not in stopwords]
            # Stem query tokens
            q_tokens = [stemmer.stem(t) for t in q_tokens]

            for movie in movies:
                title = (movie.get("title") or "").strip()
                title_lc = title.casefold()
                title_norm = " ".join(title_lc.translate(_punct_trans).split())
                title_tokens = [t for t in title_norm.split() if t and t not in stopwords]
                # Stem title tokens
                title_tokens = [stemmer.stem(t) for t in title_tokens]

                # Match if any query token is a substring of any title token
                matched = any(qt in tt for qt in q_tokens for tt in title_tokens)

                if matched:
                    results.append(movie)

            # Sort by id ascending and truncate to at most 5 results
            def _id_key(m):
                try:
                    return int(m.get("id", 0) or 0)
                except (ValueError, TypeError):
                    return 0

            results_sorted = sorted(results, key=_id_key)
            results_trunc = results_sorted[:5]

            # Print results as a numbered list of titles
            if results_trunc:
                for i, movie in enumerate(results_trunc, start=1):
                    title = movie.get("title", "<untitled>")
                    print(f"{i}. {title}")
            else:
                print("No results found.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()