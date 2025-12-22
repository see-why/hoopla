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
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            
            # Load movies dataset
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search and perform RRF search
            hybrid_search = HybridSearch(docs)
            results = hybrid_search.rrf_search(query, k=60, limit=5)
            
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
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

