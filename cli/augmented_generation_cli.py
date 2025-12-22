#!/usr/bin/env python3

import argparse
import sys

try:
    from cli.lib.semantic_search import load_movies_dataset
    from cli.hybrid_search_cli import HybridSearch, get_gemini_client
except ImportError:
    from lib.semantic_search import load_movies_dataset
    from hybrid_search_cli import HybridSearch, get_gemini_client


def main():
    """
    Retrieval Augmented Generation (RAG) CLI entry point.

    Purpose:
    - Retrieves relevant movie documents using RRF hybrid search.
    - Generates contextual answers using Gemini API based on retrieved documents.
    - Combines document retrieval with LLM generation to provide informed responses.

    Arguments:
    - query (str): The user's question or request for movie recommendations/information.
    - --k (int): RRF constant parameter controlling fusion sensitivity. Default: 60.
    - --limit (int): Number of movie documents to retrieve for RAG context. Default: 5.

    Behavior:
    - Loads the movies dataset via `load_movies_dataset()`.
    - Initializes `HybridSearch` with loaded documents.
    - Performs RRF search with configurable k and limit parameters.
    - Exits with error code 1 if no search results are found.
    - Formats retrieved documents and sends them to Gemini API with user query.
    - Generates response tailored for Hoopla movie streaming service users.

    Output Format:
    Search Results:
      - <Movie Title 1>
      - <Movie Title 2>
      ...

    RAG Response:
    <LLM-generated answer based on retrieved documents>

    Dependencies:
    - GEMINI_API_KEY environment variable must be set for LLM generation.
    - load_movies_dataset() from lib.semantic_search for movie data.
    - HybridSearch from hybrid_search_cli for RRF search functionality.
    - Gemini 2.0 Flash model via Google genai client.

    Notes:
    - Search results are limited to movie titles and descriptions.
    - LLM response is generated within context of retrieved documents only.
    - Exit code is 1 when search returns no results or API call fails; 0 on success.
    """
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--k", type=int, default=60, help="RRF constant parameter. Default: 60")
    rag_parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve and use for RAG. Default: 5")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve and summarize. Default: 5")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            k = args.k
            limit = args.limit
            
            # Load movies dataset
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search and perform RRF search
            hybrid_search = HybridSearch(docs)
            results = hybrid_search.rrf_search(query, k=k, limit=limit)
            
            # Handle case when no results are found
            if not results:
                print("No results found for the query. Unable to generate RAG response without context.", file=sys.stderr)
                sys.exit(1)
            
            # Format search results for the LLM prompt
            formatted_docs = []
            result_titles = []
            for rank, (doc_id, scores) in enumerate(results, start=1):
                doc = next((d for d in docs if d.get("id") == doc_id), None)
                if doc:
                    title = doc.get("title", "<untitled>")
                    result_titles.append(title)
                    description = doc.get("description", "")
                    # Limit description to first 500 characters for the prompt
                    if len(description) > 500:
                        description = description[:497] + "..."
                    formatted_docs.append(f"{rank}. {title}\n{description}")
            
            docs_string = "\n\n".join(formatted_docs)
            
            # Build the RAG prompt
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs_string}

Provide a comprehensive answer that addresses the query:"""
            
            try:
                client = get_gemini_client()
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=prompt
                )
                
                # Print results and response in the requested format
                print("Search Results:")
                for title in result_titles:
                    print(f"  - {title}")
                
                print(f"\nRAG Response:")
                print(response.text)
            
            except Exception as e:
                print(f"Error generating response: {e}", file=sys.stderr)
                sys.exit(1)
        
        case "summarize":
            query = args.query
            limit = args.limit
            
            # Load movies dataset
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search and perform RRF search
            hybrid_search = HybridSearch(docs)
            results = hybrid_search.rrf_search(query, k=60, limit=limit)
            
            # Handle case when no results are found
            if not results:
                print("No results found for the query. Unable to generate summary without context.", file=sys.stderr)
                sys.exit(1)
            
            # Format search results for the LLM prompt
            formatted_docs = []
            result_titles = []
            for rank, (doc_id, scores) in enumerate(results, start=1):
                doc = next((d for d in docs if d.get("id") == doc_id), None)
                if doc:
                    title = doc.get("title", "<untitled>")
                    result_titles.append(title)
                    description = doc.get("description", "")
                    # Limit description to first 500 characters for the prompt
                    if len(description) > 500:
                        description = description[:497] + "..."
                    formatted_docs.append(f"{rank}. {title}\n{description}")
            
            docs_string = "\n\n".join(formatted_docs)
            
            # Build the summarization prompt
            prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{docs_string}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            
            try:
                client = get_gemini_client()
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=prompt
                )
                
                # Print results and summary in the requested format
                print("Search Results:")
                for title in result_titles:
                    print(f"  - {title}")
                
                print(f"\nSummary:")
                print(response.text)
            
            except Exception as e:
                print(f"Error generating summary: {e}", file=sys.stderr)
                sys.exit(1)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

