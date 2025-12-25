#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import sys

try:
    from cli.keyword_search_cli import InvertedIndex
    from cli.lib.semantic_search import ChunkedSemanticSearch
except ImportError:
    from keyword_search_cli import InvertedIndex
    from lib.semantic_search import ChunkedSemanticSearch


# Constants for hybrid search configuration
# EXPANSION_FACTOR: Multiplier for initial search limit to ensure sufficient candidate documents
# for hybrid ranking. A larger factor ensures more documents are considered from both search methods
# before normalization and weighted combination, improving result quality at the cost of performance.
EXPANSION_FACTOR = 500

# MAX_EXPANDED_LIMIT: Maximum number of results to fetch from each search method, regardless of
# requested limit. This prevents excessive memory usage and processing time for large limit values.
MAX_EXPANDED_LIMIT = 10000

# DEFAULT_RERANK_MULTIPLIER: Default multiplier for initial search limit when using LLM-based reranking.
# A higher multiplier provides more candidates for reranking (improving accuracy) but increases API calls
# and processing time. Can be overridden via --rerank-multiplier argument.
DEFAULT_RERANK_MULTIPLIER = 5


def get_gemini_client():
    """
    Initialize and return a Gemini API client.
    
    Loads the API key from environment variables using dotenv and creates a Gemini client.
    Raises an error if the API key is not set.
    
    Returns:
        genai.Client: Initialized Gemini API client
    
    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set
    """
    from dotenv import load_dotenv
    from google import genai
    
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return genai.Client(api_key=api_key)


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

    @staticmethod
    def normalize(scores):
        """
        Normalize a list of scores using min-max normalization.

        Args:
            scores (list of float): The scores to normalize.

        Returns:
            list of float: The normalized scores (0.0 to 1.0 range).
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            # All scores are the same, normalize to 1.0
            return [1.0] * len(scores)
        
        # Apply min-max normalization: (x - min) / (max - min)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def weighted_search(self, query, alpha, limit=5):
        """
        Perform weighted hybrid search combining BM25 and semantic search.

        Args:
            query (str): The search query string.
            alpha (float): Weight for BM25 scores (0.0-1.0). Semantic weight is (1-alpha).
            limit (int): The maximum number of results to return.

        Returns:
            list of tuple: A list of (doc_id, scores_dict) tuples sorted by hybrid score descending,
                where scores_dict contains 'bm25', 'semantic', and 'hybrid' normalized scores.
        
        Raises:
            ValueError: If alpha is not between 0.0 and 1.0 (inclusive).
        """
        # Validate alpha parameter
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0 (inclusive), got {alpha}")
        
        # Expand the limit to fetch more candidates for better hybrid ranking.
        # We fetch many more results than needed because:
        # 1. Documents ranking high in one method may rank low in the other
        # 2. Normalization and weighted combination may reorder results significantly
        # 3. We need sufficient overlap between both methods for meaningful hybrid scores
        # Cap at MAX_EXPANDED_LIMIT to prevent excessive memory/processing for large limits.
        expanded_limit = min(limit * EXPANSION_FACTOR, MAX_EXPANDED_LIMIT)
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, expanded_limit)
        
        # Get semantic search results
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        
        # Create dictionaries for easy lookup
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}
        semantic_dict = {result["id"]: result["score"] for result in semantic_results}
        
        # Get all unique document IDs from both searches
        # Sort to ensure deterministic ordering for consistent results across runs
        all_doc_ids = sorted(set(bm25_dict.keys()) | set(semantic_dict.keys()))
        
        # Extract scores for normalization
        bm25_scores = [bm25_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        semantic_scores = [semantic_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        
        # Normalize scores
        normalized_bm25 = self.normalize(bm25_scores)
        normalized_semantic = self.normalize(semantic_scores)
        
        # Create a mapping of doc_id to normalized scores
        doc_scores = {}
        for i, doc_id in enumerate(all_doc_ids):
            bm25_norm = normalized_bm25[i]
            semantic_norm = normalized_semantic[i]
            hybrid_score = alpha * bm25_norm + (1 - alpha) * semantic_norm
            
            doc_scores[doc_id] = {
                "bm25": bm25_norm,
                "semantic": semantic_norm,
                "hybrid": hybrid_score
            }
        
        # Sort by hybrid score descending
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1]["hybrid"], reverse=True)
        
        # Return top 'limit' results as list of (doc_id, scores_dict) tuples
        return sorted_results[:limit]

    def rrf_search(self, query, k=60, limit=10):
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings from multiple search methods by summing reciprocal ranks,
        providing a rank-based fusion that doesn't require score normalization.
        
        Args:
            query (str): The search query string.
            k (int): RRF constant parameter (default 60). Higher values give less weight to top-ranked items.
            limit (int): The maximum number of results to return.
        
        Returns:
            list of tuple: A list of (doc_id, scores_dict) tuples sorted by RRF score descending,
                where scores_dict contains 'rrf', 'bm25_rank', and 'semantic_rank'.
        
        Raises:
            ValueError: If k or limit is not a positive integer.
        """
        # Validate parameters
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        
        # Expand the limit to get more candidates
        expanded_limit = min(limit * EXPANSION_FACTOR, MAX_EXPANDED_LIMIT)
        
        # Get BM25 results (returns list of (doc_id, score) tuples)
        bm25_results = self._bm25_search(query, expanded_limit)
        
        # Get semantic search results (returns list of dicts with 'id' and 'score')
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        
        # Create rank mappings (1-indexed)
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results)}
        semantic_ranks = {result["id"]: rank + 1 for rank, result in enumerate(semantic_results)}
        
        # Get all unique document IDs and sort for deterministic ordering
        all_doc_ids = sorted(set(bm25_ranks.keys()) | set(semantic_ranks.keys()))
        
        # Calculate RRF scores for each document
        doc_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            bm25_rank = bm25_ranks.get(doc_id)
            semantic_rank = semantic_ranks.get(doc_id)
            
            # Add RRF contribution from BM25 if document appears in BM25 results
            if bm25_rank is not None:
                rrf_score += 1.0 / (k + bm25_rank)
            
            # Add RRF contribution from semantic search if document appears in semantic results
            if semantic_rank is not None:
                rrf_score += 1.0 / (k + semantic_rank)
            
            doc_scores[doc_id] = {
                "rrf": rrf_score,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank
            }
        
        # Sort by RRF score descending
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1]["rrf"], reverse=True)
        
        # Return top 'limit' results
        return sorted_results[:limit]


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

    # rrf-search command: perform RRF (Reciprocal Rank Fusion) hybrid search
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform hybrid search using Reciprocal Rank Fusion (RRF)")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF constant parameter. Default: 60")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return. Default: 5")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch"],
        help="Reranking method to apply after RRF search",
    )
    rrf_search_parser.add_argument(
        "--rerank-multiplier",
        type=int,
        default=DEFAULT_RERANK_MULTIPLIER,
        help=f"Multiplier for initial search limit when reranking (default: {DEFAULT_RERANK_MULTIPLIER}). Higher values provide more candidates for reranking.",
    )

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
            import sys
            
            # Validate alpha parameter
            if not (0.0 <= args.alpha <= 1.0):
                print(f"Error: alpha must be between 0.0 and 1.0, got {args.alpha}", file=sys.stderr)
                sys.exit(1)
            
            # Validate limit parameter
            if args.limit <= 0:
                print(f"Error: limit must be a positive integer, got {args.limit}", file=sys.stderr)
                sys.exit(1)
            
            # Lazy import to load movies dataset
            try:
                from cli.lib.semantic_search import load_movies_dataset
            except ImportError:
                from lib.semantic_search import load_movies_dataset
            
            # Load documents
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search
            hs = HybridSearch(docs)
            
            # Perform weighted search
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            
            # Print results with detailed score breakdown
            if not results:
                print("No results found.")
            else:
                print(f"Top {len(results)} results for query: '{args.query}' (alpha={args.alpha}):\n")
                
                for rank, (doc_id, scores) in enumerate(results, start=1):
                    # Find document by id
                    doc = next((d for d in docs if d.get("id") == doc_id), None)
                    if doc:
                        title = doc.get("title", "<untitled>")
                        description = doc.get("description", "")
                        
                        # Truncate description to ~100 characters
                        if len(description) > 100:
                            description = description[:97] + "..."
                        
                        # Print formatted output
                        print(f"{rank}. {title}")
                        print(f"   Hybrid Score: {scores['hybrid']:.3f}")
                        print(f"   BM25: {scores['bm25']:.3f}, Semantic: {scores['semantic']:.3f}")
                        print(f"   {description}\n")
        case "rrf-search":
            import sys
            
            # Validate k parameter
            if args.k <= 0:
                print(f"Error: k must be a positive integer, got {args.k}", file=sys.stderr)
                sys.exit(1)
            
            # Validate limit parameter
            if args.limit <= 0:
                print(f"Error: limit must be a positive integer, got {args.limit}", file=sys.stderr)
                sys.exit(1)
            
            # Lazy import to load movies dataset
            try:
                from cli.lib.semantic_search import load_movies_dataset
            except ImportError:
                from lib.semantic_search import load_movies_dataset
            
            # Load documents
            docs, exc, movies_path = load_movies_dataset()
            if exc:
                print(f"Failed to load movies file {movies_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            
            # Initialize hybrid search
            hs = HybridSearch(docs)
            
            # Handle query enhancement
            query = args.query
            if args.enhance in ["spell", "rewrite", "expand"]:
                try:
                    client = get_gemini_client()
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
                
                if args.enhance == "spell":
                    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
                elif args.enhance == "rewrite":
                    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
                else:  # expand
                    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Return ONLY the additional terms (not the original query) - they will be appended automatically.

Examples:

- "scary bear movie" -> "horror grizzly terrifying film"
- "action movie with bear" -> "thriller chase fight adventure"
- "comedy with bear" -> "funny humor lighthearted"

Query: "{query}"

Expanded terms:"""
                
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-001",
                        contents=prompt
                    )
                    enhanced_query = response.text.strip()
                    
                    # Validate the enhanced query with different thresholds
                    if not enhanced_query:
                        raise ValueError("Empty enhancement response")
                    
                    # Spell correction should be similar length, rewriting can be longer, expansion can be even longer
                    if args.enhance == "spell":
                        max_length = len(query) * 3
                    elif args.enhance == "rewrite":
                        max_length = max(len(query) * 5, 200)  # At least 200 chars allowed
                    else:  # expand
                        max_length = max(len(query) * 6, 250)  # Expansions can be longer
                    
                    if len(enhanced_query) > max_length:
                        raise ValueError(f"Enhanced query too long ({len(enhanced_query)} chars)")
                    
                    # For expand, append to original query; for spell/rewrite, replace
                    if args.enhance == "expand":
                        if enhanced_query != query:
                            expanded_query = f"{query} {enhanced_query}"
                            print(f"Enhanced query ({args.enhance}): '{query}' -> '{expanded_query}'\n")
                            query = expanded_query
                    else:
                        if enhanced_query != query:
                            print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                        query = enhanced_query
                except Exception as e:
                    print(f"Warning: Query enhancement failed: {e}", file=sys.stderr)
                    print("Continuing with original query...", file=sys.stderr)
            
            # Perform RRF search
            # If reranking, gather more results initially
            search_limit = args.limit * args.rerank_multiplier if args.rerank_method else args.limit
            results = hs.rrf_search(query, args.k, search_limit)
            
            # Apply individual reranking if specified
            if args.rerank_method == "individual" and results:
                try:
                    client = get_gemini_client()
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
                
                print(f"Reranking {len(results)} results to return top {args.limit}...")
                
                async def score_document(doc_id, scores):
                    """Score a single document asynchronously."""
                    doc = next((d for d in docs if d.get("id") == doc_id), None)
                    if not doc:
                        print(f"Warning: Document with ID {doc_id} not found, skipping", file=sys.stderr)
                        return None
                    
                    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
                    
                    try:
                        # Run the synchronous API call in a thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None,
                            lambda: client.models.generate_content(
                                model="gemini-2.0-flash-001",
                                contents=prompt
                            )
                        )
                        score_text = response.text.strip()
                        
                        # Extract numeric score with robust parsing
                        # Try to find a number in the response (handles cases where LLM adds extra text)
                        llm_score = None
                        
                        # First, try direct conversion (fastest path)
                        try:
                            llm_score = float(score_text)
                        except ValueError:
                            # If that fails, try to extract the first number using regex
                            match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_text)
                            if match:
                                llm_score = float(match.group(1))
                            else:
                                # Last resort: try splitting on whitespace and converting first token
                                tokens = score_text.split()
                                for token in tokens:
                                    try:
                                        llm_score = float(token)
                                        break
                                    except ValueError:
                                        continue
                        
                        if llm_score is None:
                            print(f"Warning: Could not extract score from response '{score_text}' for {doc.get('title', 'unknown')}, skipping", file=sys.stderr)
                            return None
                        
                        # Validate score is in range
                        if 0 <= llm_score <= 10:
                            return (doc_id, {
                                **scores,
                                'llm_score': llm_score
                            })
                        else:
                            print(f"Warning: Invalid score {llm_score} for {doc.get('title', 'unknown')}, skipping", file=sys.stderr)
                            return None
                    except Exception as e:
                        print(f"Warning: Reranking failed for {doc.get('title', 'unknown')}: {e}", file=sys.stderr)
                        return None
                
                async def score_all_documents():
                    """Score all documents concurrently."""
                    tasks = [score_document(doc_id, scores) for doc_id, scores in results]
                    return await asyncio.gather(*tasks)
                
                # Run async scoring
                scored_results = asyncio.run(score_all_documents())
                
                # Filter out None results and collect valid scores
                reranked_results = [result for result in scored_results if result is not None]
                
                # Sort by LLM score (descending) and take top limit
                reranked_results.sort(key=lambda x: x[1]['llm_score'], reverse=True)
                results = reranked_results[:args.limit]
            
            # Apply batch reranking if specified
            elif args.rerank_method == "batch" and results:
                try:
                    client = get_gemini_client()
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
                
                print(f"Reranking {len(results)} results to return top {args.limit}...")
                
                async def batch_rerank():
                    """Perform batch reranking with a single LLM call."""
                    # Build document list string and track missing documents
                    doc_list_parts = []
                    missing_docs = []
                    valid_doc_ids = []
                    
                    for doc_id, scores in results:
                        doc = next((d for d in docs if d.get("id") == doc_id), None)
                        if doc:
                            doc_list_parts.append(f"{doc_id}. {doc.get('title', '')} - {doc.get('document', '')}")
                            valid_doc_ids.append(doc_id)
                        else:
                            missing_docs.append(doc_id)
                    
                    # Warn about missing documents
                    if missing_docs:
                        print(f"Warning: {len(missing_docs)} document(s) not found in dataset (IDs: {missing_docs}), excluding from reranking", file=sys.stderr)
                    
                    doc_list_str = "\n".join(doc_list_parts)
                    
                    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
                    
                    try:
                        # Run the synchronous API call in a thread pool
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None,
                            lambda: client.models.generate_content(
                                model="gemini-2.0-flash-001",
                                contents=prompt
                            )
                        )
                        response_text = response.text.strip()
                        
                        # Parse JSON response
                        try:
                            ranked_ids = json.loads(response_text)
                        except json.JSONDecodeError:
                            # Try to extract JSON from the response if it has extra text
                            # Pattern handles both empty arrays [] and arrays with elements [1, 2, 3]
                            match = re.search(r'\[\s*(\d+\s*(?:,\s*\d+\s*)*)?\s*\]', response_text)
                            if match:
                                ranked_ids = json.loads(match.group(0))
                            else:
                                print(f"Warning: Could not parse JSON from response: {response_text}", file=sys.stderr)
                                return None
                        
                        # Validate that ranked_ids is a list
                        if not isinstance(ranked_ids, list):
                            print(f"Warning: LLM returned valid JSON but not a list: {ranked_ids}", file=sys.stderr)
                            return None
                        
                        # Create a mapping of doc_id to new rank
                        rank_map = {doc_id: idx + 1 for idx, doc_id in enumerate(ranked_ids)}
                        
                        # Check if LLM missed any documents
                        ranked_set = set(ranked_ids)
                        valid_set = set(valid_doc_ids)
                        unranked_ids = valid_set - ranked_set
                        
                        if unranked_ids:
                            print(f"Warning: LLM did not rank {len(unranked_ids)} document(s) (IDs: {sorted(unranked_ids)}), placing them at the end", file=sys.stderr)
                        
                        # Add rerank position to ranked results
                        reranked_results = []
                        unranked_results = []
                        
                        for doc_id, scores in results:
                            if doc_id in rank_map:
                                reranked_results.append((doc_id, {
                                    **scores,
                                    'rerank_position': rank_map[doc_id]
                                }))
                            elif doc_id in valid_doc_ids:
                                # Document was valid but not ranked by LLM - add to end
                                unranked_results.append((doc_id, scores))
                        
                        # Sort ranked results by rerank position
                        reranked_results.sort(key=lambda x: x[1]['rerank_position'])
                        
                        # Append unranked results at the end (preserve original RRF order)
                        for doc_id, scores in unranked_results:
                            reranked_results.append((doc_id, {
                                **scores,
                                'rerank_position': None  # Indicate not ranked by LLM
                            }))
                        
                        # Return top limit results
                        return reranked_results[:args.limit]
                    
                    except Exception as e:
                        print(f"Warning: Batch reranking failed: {e}", file=sys.stderr)
                        return None
                
                # Run async batch reranking
                reranked_results = asyncio.run(batch_rerank())
                
                if reranked_results is not None:
                    results = reranked_results
            
            # Print results with rank information
            if not results:
                print("No results found.")
            else:
                # Customize header based on whether reranking was applied
                if args.rerank_method == "individual":
                    print(f"LLM Reranked Results (RRF + Individual Reranking) for '{query}' (k={args.k}):\n")
                elif args.rerank_method == "batch":
                    print(f"LLM Reranked Results (RRF + Batch Reranking) for '{query}' (k={args.k}):\n")
                else:
                    print(f"Top {len(results)} results for query: '{query}' (k={args.k}):\n")
                
                for rank, (doc_id, scores) in enumerate(results, start=1):
                    # Find document by id
                    doc = next((d for d in docs if d.get("id") == doc_id), None)
                    if doc:
                        title = doc.get("title", "<untitled>")
                        description = doc.get("description", "")
                        
                        # Truncate description to ~100 characters
                        if len(description) > 100:
                            description = description[:97] + "..."
                        
                        # Print formatted output
                        print(f"{rank}. {title}")
                        
                        # Show rerank score first if individually reranked
                        if 'llm_score' in scores:
                            print(f"   Rerank Score: {scores['llm_score']:.3f}/10")
                        # Show rerank rank if batch reranked
                        elif 'rerank_position' in scores:
                            if scores['rerank_position'] is not None:
                                print(f"   Rerank Rank: {scores['rerank_position']}")
                            else:
                                print(f"   Rerank Rank: (not ranked by LLM)")
                        
                        print(f"   RRF Score: {scores['rrf']:.3f}")
                        
                        print(f"   BM25 Rank: {scores['bm25_rank']}, Semantic Rank: {scores['semantic_rank']}")
                        print(f"   {description}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
