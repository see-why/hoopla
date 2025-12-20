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
    parser.add_argument(
        "--golden-dataset",
        type=str,
        default="data/golden_dataset.json",
        help="Path to the golden dataset JSON file. Default: data/golden_dataset.json",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant parameter for Reciprocal Rank Fusion. Default: 60",
    )

    args = parser.parse_args()
    limit = args.limit
    golden_dataset_path = args.golden_dataset
    rrf_k = args.rrf_k

    # Load golden dataset
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
    hybrid_search = HybridSearch(docs)
    
    # Create document lookup dictionary for O(1) access
    docs_by_id = {doc.get("id"): doc for doc in docs if doc.get("id") is not None}
    
    # Run evaluation for each test case
    test_cases = golden_data.get("test_cases", [])
    
    # Print header with k value
    print(f"k={limit}\n")
    
    for _, test_case in enumerate(test_cases, 1):
        query = test_case.get("query", "")
        relevant_docs = test_case.get("relevant_docs", [])
        
        # Run RRF search with configurable k parameter and limit from args
        results = hybrid_search.rrf_search(query, k=rrf_k, limit=limit)
        
        # Extract document titles from results
        retrieved_titles = []
        for doc_id, scores in results:
            doc = docs_by_id.get(doc_id)
            if doc:
                retrieved_titles.append(doc.get("title", ""))
        
        # Calculate precision: how many of the retrieved documents are relevant
        relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_docs)
        precision = relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0.0
        
        # Calculate recall: how many of the relevant documents were retrieved
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0
        
        # Print results
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()

