#!/usr/bin/env python3

import argparse
import json
import sys

try:
    from cli.keyword_search_cli import InvertedIndex
    from cli.lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset
    from cli.hybrid_search_cli import HybridSearch
except ImportError:
    from keyword_search_cli import InvertedIndex
    from lib.semantic_search import ChunkedSemanticSearch, load_movies_dataset
    from hybrid_search_cli import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load golden dataset
    golden_dataset_path = "data/golden_dataset.json"
    try:
        with open(golden_dataset_path, "r") as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Golden dataset not found at {golden_dataset_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse golden dataset JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load documents
    docs, exc, movies_path = load_movies_dataset()
    if exc:
        print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize hybrid search
    hs = HybridSearch(docs)
    
    # Run evaluation for each test case
    test_cases = golden_data.get("test_cases", [])
    
    for idx, test_case in enumerate(test_cases, 1):
        query = test_case.get("query", "")
        relevant_docs = test_case.get("relevant_docs", [])
        
        print(f"Test case {idx}: '{query}'")
        
        # Run RRF search with k=60 and limit from args
        results = hs.rrf_search(query, k=60, limit=limit)
        
        # Extract document titles from results
        retrieved_titles = []
        for doc_id, scores in results:
            doc = next((d for d in docs if d.get("id") == doc_id), None)
            if doc:
                retrieved_titles.append(doc.get("title", ""))
        
        print(f"  Retrieved {len(retrieved_titles)} documents")
        print(f"  Expected {len(relevant_docs)} relevant documents")
        print()


if __name__ == "__main__":
    main()

