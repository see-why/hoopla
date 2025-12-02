#!/usr/bin/env python3

import argparse
import os

try:
    from cli.keyword_search_cli import InvertedIndex
    from cli.lib.semantic_search import ChunkedSemanticSearch
except ImportError:
    from keyword_search_cli import InvertedIndex
    from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    """
    HybridSearch provides a unified interface for performing hybrid search over a collection of documents,
    combining both semantic (embedding-based) and keyword (BM25) search techniques.

    Args:
        documents (list): A list of documents to be indexed and searched. Each document should be a string
            or an object compatible with the underlying semantic and keyword search components.

    Initialization:
        - Initializes a semantic search component (`ChunkedSemanticSearch`) and loads or creates chunk embeddings
          for the provided documents.
        - Initializes a keyword search component (`InvertedIndex`). If the index does not exist, it is built and saved.

    Methods:
        - _bm25_search(query, limit): Performs a BM25 keyword search for the given query, returning up to `limit` results.
        - weighted_search(query, alpha, limit=5): (Not implemented) Intended for weighted hybrid search.
        - rrf_search(query, k, limit=10): (Not implemented) Intended for Reciprocal Rank Fusion hybrid search.

    Usage:
        hs = HybridSearch(documents)
        results = hs._bm25_search("example query", limit=5)
    """
    def __init__(self, documents):
        """
        Initialize the HybridSearch object.

        Args:
            documents (list of dict): A list of documents, where each document is a dict
                with keys 'id', 'title', and 'description'.

        Initialization steps:
            - Stores the provided documents.
            - Initializes the semantic search engine and loads or creates chunk embeddings for the documents.
            - Initializes the inverted index. If the index does not exist on disk, it is built and saved.

        Raises:
            OSError: If there are issues reading from or writing to disk during index or embedding operations.
            Exception: Propagates exceptions raised by ChunkedSemanticSearch or InvertedIndex during initialization.
        """
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build(self.documents)
            self.idx.save()
        try:
            self.idx.load()
        except FileNotFoundError:
            raise RuntimeError(f"Failed to load the index from {self.idx.index_path}. The index file is missing.")

    def _bm25_search(self, query, limit):
        """
        Perform a BM25 keyword search using the inverted index.

        Args:
            query (str): The search query string.
            limit (int): The maximum number of results to return.

        Returns:
            list of tuple: A list of (doc_id, score) tuples representing the top matching documents.
        """
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize command: apply min-max normalization to a list of scores
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    # weighted-search command: perform weighted hybrid search
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search combining BM25 and semantic search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 scores (0.0-1.0), semantic weight is (1-alpha). Default: 0.5")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return. Default: 5")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                # Don't print anything if no scores provided
                return
            
            # Min-max normalization
            min_score = min(scores)
            max_score = max(scores)
            
            if min_score == max_score:
                # All scores are the same, normalize to 1.0
                normalized = [1.0] * len(scores)
            else:
                # Apply min-max normalization: (x - min) / (max - min)
                normalized = [(score - min_score) / (max_score - min_score) for score in scores]
            
            # Print normalized scores with 4 decimal places
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            # Lazy import to load movies dataset
            try:
                from cli.lib.semantic_search import load_movies_dataset
            except ImportError:
                from lib.semantic_search import load_movies_dataset
            
            # Load documents
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                import sys
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search
            hs = HybridSearch(docs)
            
            # Perform weighted search
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            
            # Print results
            if not results:
                print("No results found.")
            else:
                for rank, (doc_id, score) in enumerate(results, start=1):
                    # Find document by id
                    doc = next((d for d in docs if d.get("id") == doc_id), None)
                    if doc:
                        title = doc.get("title", "<untitled>")
                        print(f"{rank}. [{doc_id}] {title} (score: {score:.4f})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
