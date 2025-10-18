#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

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
                    import re

                    # Find top-level movie object blocks between { ... } inside the movies array
                    # This is a heuristic that works for the provided dataset format.
                    blocks = re.findall(r"\{(.*?)\}(?=\s*,|\s*\])", text, re.S)
                    data = {"movies": []}
                    for blk in blocks:
                        # Extract id
                        m_id = re.search(r"\bid\s*:\s*(\d+)", blk)
                        try:
                            movie_id = int(m_id.group(1)) if m_id else None
                        except Exception:
                            movie_id = None

                        # Extract title (non-greedy up to a closing quote)
                        m_title = re.search(r'title\s*:\s*"(.*?)"', blk, re.S)
                        title = m_title.group(1).strip() if m_title else ""

                        # Extract description if present
                        m_desc = re.search(r'description\s*:\s*"(.*?)"\s*(?:,|$)', blk, re.S)
                        description = m_desc.group(1).strip() if m_desc else None

                        movie = {"id": movie_id, "title": title}
                        if description is not None:
                            movie["description"] = description

                        # Only add if we found a title
                        if title:
                            data["movies"].append(movie)
                except Exception:
                    print(f"Failed to decode JSON from: {data_path}")
                    return

            movies = data.get("movies", []) if isinstance(data, dict) else []

            q_lower = args.query.lower()
            for movie in movies:
                title = movie.get("title", "")
                if q_lower in title.lower():
                    results.append(movie)

            # Print results as a numbered list of titles
            if results:
                for i, movie in enumerate(results, start=1):
                    title = movie.get("title", "<untitled>")
                    print(f"{i}. {title}")
            else:
                print("No results found.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()