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
    parser.add_subparsers(dest="command", help="Available commands")

    args = parser.parse_args()

    parser.print_help()


if __name__ == "__main__":
    main()
