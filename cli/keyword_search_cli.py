#!/usr/bin/env python3

import argparse
import json
import re
import string
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import Counter

import heapq
import math
import sys
try:
    # When executed as a script, the working directory / sys.path[0]
    # will be the `cli/` directory; prefer the top-level project
    # `search_utils` module if present, otherwise fall back to
    # the local `cli.search_utils` copy. Import both tuning constants.
    from search_utils import BM25_K1, BM25_B
except Exception:
    from cli.search_utils import BM25_K1, BM25_B

# Common boolean operator words (lowercase) — keep at module scope to avoid
# reallocating this small set on every search invocation.
OPERATORS = {"and", "or", "not"}


import pickle
import os


class InvertedIndex:
    """Simple in-memory inverted index with pickle persistence.

    Attributes:
        index: dict mapping token -> set(document_id)
        docmap: dict mapping document_id -> document object
    """

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        # Term frequencies: doc_id -> Counter(token -> count)
        self.term_frequencies: dict[int, Counter] = {}
        # Document lengths: doc_id -> int (number of tokens indexed)
        self.doc_lengths: dict[int, int] = {}
        # Path for persisted doc lengths cache (set after save/load).
        # Initialize to None to avoid AttributeError if accessed early.
        self.doc_lengths_path: str | None = None

        # Normalization helpers
        self._punct_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))
        self.stemmer = PorterStemmer()

        sw_path = Path(__file__).resolve().parents[1] / "data" / "stopwords.txt"
        self.stopwords: set[str] = set()
        if sw_path.exists():
            try:
                sw_text = sw_path.read_text(encoding="utf-8")
                self.stopwords = {w.strip().casefold() for w in sw_text.splitlines() if w.strip()}
            except OSError:
                self.stopwords = set()

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and add doc_id to each token's posting set."""
        if not isinstance(text, str):
            return
        # normalize and split into tokens (remove stopwords)
        norm = " ".join(text.casefold().translate(self._punct_trans).split())
        tokens_pre_stem = [t for t in norm.split() if t and t not in self.stopwords]

        # record document length (number of tokens after stopword removal)
        # `doc_id` is expected to be an integer (the caller performs
        # coercion/validation); avoid redundant conversion and early
        # return here so errors surface at the call site.
        mid = doc_id
        token_count = len(tokens_pre_stem)
        self.doc_lengths[mid] = token_count

        # apply stemming for indexing
        tokens = [self.stemmer.stem(t) for t in tokens_pre_stem]

        for tok in tokens:
            postings = self.index.setdefault(tok, set())
            postings.add(mid)
            # update term frequencies for this document
            ctr = self.term_frequencies.setdefault(mid, Counter())
            ctr[tok] += 1

    def get_documents(self, term: str) -> list[int]:
        """Return sorted list of document ids for a token/term."""
        if not term:
            return []
        key = self.stemmer.stem(term.casefold())
        ids = self.index.get(key, set())
        return sorted(int(i) for i in ids)

    def get_tf(self, doc_id: str | int, term: str) -> int:
        """Return term frequency for a single-token term in a given document.

        doc_id may be an int or a string representing an int. The term is
        tokenized using the same normalization pipeline; it must resolve to
        exactly one token, otherwise a ValueError is raised.
        """
        try:
            mid = int(doc_id)
        except (ValueError, TypeError):
            raise ValueError("doc_id must be an integer or integer string")

        if not isinstance(term, str):
            raise ValueError("term must be a string")

        norm = " ".join(term.casefold().translate(self._punct_trans).split())
        toks = [t for t in norm.split() if t and t not in self.stopwords]
        if len(toks) != 1:
            raise ValueError("term must be a single token after normalization")

        tok = self.stemmer.stem(toks[0])
        return int(self.term_frequencies.get(mid, Counter()).get(tok, 0))

    def build(self, movies: list[dict]) -> None:
        """Build the index and docmap from a list of movie dicts."""
        for m in movies:
            # Expect an integer id; attempt to coerce
            try:
                mid = int(m.get("id") or 0)
            except (ValueError, TypeError):
                # skip documents without a valid integer id
                continue

            title = m.get("title") or ""
            description = m.get("description") or ""
            fulltext = f"{title} {description}".strip()

            # add to docmap and index
            self.docmap[mid] = m
            self.__add_document(mid, fulltext)

    def save(self, cache_dir: str | Path = None) -> None:
        """Persist index and docmap into cache/index.pkl and cache/docmap.pkl."""
        base = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "cache"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            # if we can't create the directory, raise
            raise

        idx_path = base / "index.pkl"
        docmap_path = base / "docmap.pkl"

        # Use highest protocol for efficiency
        with open(idx_path, "wb") as fh:
            pickle.dump(self.index, fh, protocol=pickle.HIGHEST_PROTOCOL)

        with open(docmap_path, "wb") as fh:
            pickle.dump(self.docmap, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Save term frequencies
        tf_path = base / "term_frequencies.pkl"
        with open(tf_path, "wb") as fh:
            pickle.dump(self.term_frequencies, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Save document lengths
        doc_lengths_path = base / "doc_lengths.pkl"
        with open(doc_lengths_path, "wb") as fh:
            pickle.dump(self.doc_lengths, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # update stored path
        self.doc_lengths_path = str(doc_lengths_path)

    def load(self, cache_dir: str | Path = None) -> None:
        """Load index and docmap from cache/index.pkl and cache/docmap.pkl.

        Raises FileNotFoundError if files are missing.
        """
        base = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "cache"
        idx_path = base / "index.pkl"
        docmap_path = base / "docmap.pkl"

        if not idx_path.exists() or not docmap_path.exists():
            raise FileNotFoundError(f"Cache files not found at {base}")

        with open(idx_path, "rb") as fh:
            self.index = pickle.load(fh)

        with open(docmap_path, "rb") as fh:
            self.docmap = pickle.load(fh)

        # Load term frequencies
        tf_path = base / "term_frequencies.pkl"
        if not tf_path.exists():
            raise FileNotFoundError(f"Cache files not found at {base}")
        with open(tf_path, "rb") as fh:
            self.term_frequencies = pickle.load(fh)

        # Load document lengths
        doc_lengths_path = base / "doc_lengths.pkl"
        if not doc_lengths_path.exists():
            raise FileNotFoundError(f"Cache files not found at {base}")
        with open(doc_lengths_path, "rb") as fh:
            self.doc_lengths = pickle.load(fh)
        self.doc_lengths_path = str(doc_lengths_path)

    def get_bm25_idf(self, term: str) -> float:
        """Compute BM25-style IDF for a single-term query.

        Uses formula: log((N - df + 0.5) / (df + 0.5) + 1)
        where N is total number of documents and df is document frequency.

        The term is normalized with the same pipeline as indexing and must
        resolve to a single token; otherwise ValueError is raised.
        """
        if not isinstance(term, str):
            raise ValueError("term must be a string")

        # Normalize like in indexing
        norm = " ".join(term.casefold().translate(self._punct_trans).split())
        toks = [t for t in norm.split() if t and t not in self.stopwords]
        if len(toks) != 1:
            raise ValueError("term must be a single token after normalization")

        tok = self.stemmer.stem(toks[0])
        df = len(self.index.get(tok, set()))
        N = len(self.docmap)

        if N <= 0:
            return 0.0
        if df > N:
            raise ValueError("Document frequency cannot exceed total documents")
        # BM25 idf variant (with +0.5 smoothing)
        return float(math.log((N - df + 0.5) / (df + 0.5) + 1))

    def __get_avg_doc_length(self) -> float:
        """Return the average document length (number of tokens).

        Returns 0.0 when there are no documents.
        """
        if not self.doc_lengths:
            return 0.0
        total = sum(self.doc_lengths.values())
        return float(total) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """Return BM25 term score component for a document-term.

        Uses BM25 TF with length normalization:
        (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl/avgdl)))
        where tf is raw term frequency, dl is document length, avgdl is average doc length.
        """
        # Ensure doc id is an integer for lookups
        try:
            mid = int(doc_id)
        except (ValueError, TypeError):
            raise ValueError("doc_id must be an integer or integer string")

        # Reuse existing get_tf to obtain the raw count (this validates term)
        try:
            tf_raw = int(self.get_tf(mid, term))
        except ValueError:
            # Propagate as-is for invalid term shapes
            raise

        if tf_raw <= 0:
            return 0.0

        if mid not in self.doc_lengths:
            raise KeyError(f"Document length for doc_id {mid} not found in doc_lengths. This indicates a data inconsistency.")
        dl = int(self.doc_lengths[mid])
        avgdl = self.__get_avg_doc_length()

        # length normalization factor; handle avgdl == 0
        if avgdl <= 0:
            denom = tf_raw + k1 * (1 - b)
        else:
            denom = tf_raw + k1 * (1 - b + b * (dl / avgdl))

        return float((tf_raw * (k1 + 1)) / denom)

    def bm25(self, doc_id: int, term: str) -> float:
        """Return the full BM25 score for a single term in a document.

        This multiplies the BM25 TF component by the BM25 IDF component.
        """
        # BM25 TF validates doc_id/term shapes and looks up doc length
        tf_comp = self.get_bm25_tf(doc_id, term)
        idf_comp = self.get_bm25_idf(term)
        return float(tf_comp * idf_comp)

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        """Perform a simple BM25 ranking over all documents for the query.

        Tokenizes the query using the same normalization pipeline used for
        indexing. Returns a list of (doc_id, score) sorted by score desc,
        limited to `limit` items.
        """
        if not isinstance(query, str):
            return []

        # Normalize query like in indexing
        q_norm = " ".join(query.casefold().translate(self._punct_trans).split())
        raw_tokens = [t for t in q_norm.split() if t and t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in raw_tokens]
        if not tokens:
            return []

        scores: dict[int, float] = {}

        # Iterate over all documents and sum term scores
        for doc_id in self.docmap.keys():
            total = 0.0
            for tok in tokens:
                # if term not in index, bm25 will return 0 via get_bm25_tf
                try:
                    total += self.bm25(doc_id, tok)
                except (ValueError, KeyError):
                    # skip documents/terms that raise due to inconsistent cache
                    continue
            if total > 0.0:
                scores[int(doc_id)] = total

        # sort by score descending, return top `limit`
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: max(0, int(limit))]
        return top

def bm25_idf_command(term: str, cache_dir: str | Path = None) -> float:
    """Load index from disk and return BM25 IDF for the given term."""
    idx = InvertedIndex()
    idx.load(cache_dir)
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B, cache_dir: str | Path = None) -> float:
    """Load index and return BM25 TF score for the given document and term.

    Accepts k1 and b tuning parameters.
    """
    idx = InvertedIndex()
    idx.load(cache_dir)
    return idx.get_bm25_tf(doc_id, term, k1, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    # build subcommand for creating an inverted index cache
    build_parser = subparsers.add_parser("build", help="Build inverted index cache")

    tf_parser = subparsers.add_parser("tf", help="Print term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to query")

    # idf subcommand to print inverse document frequency for a given term
    idf_parser = subparsers.add_parser("idf", help="Print inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to query")

    # tfidf subcommand to compute TF-IDF score for a term in a document
    tfidf_parser = subparsers.add_parser("tfidf", help="Print TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to query")

    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    # Backwards-compatible positional arguments (deprecated): allow callers
    # to pass k1 and b positionally to preserve older CLI usage. These
    # positional fallbacks are suppressed from help; prefer using flags.
    bm25_tf_parser.add_argument("k1_pos", type=float, nargs='?', default=None, help=argparse.SUPPRESS)
    bm25_tf_parser.add_argument("b_pos", type=float, nargs='?', default=None, help=argparse.SUPPRESS)

    # Preferred form: explicit flags. We set defaults to None so we can
    # detect which form the caller used and emit a deprecation warning
    # when the positional form is used.
    bm25_tf_parser.add_argument("--k1", dest="k1", type=float, default=None, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("--b", dest="b", type=float, default=None, help="Tunable BM25 b parameter")

    bm25_search_parser = subparsers.add_parser(
        "bm25search", help="Rank documents using BM25 for a query"
    )
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("--limit", type=int, default=5, help="Number of top documents to return")

    args = parser.parse_args()

    # Initialize stemmer for token normalization
    stemmer = PorterStemmer()

    match args.command:
        case "build":
            # load movies file
            data_dir = Path(__file__).resolve().parents[1] / "data"
            data_path = data_dir / "movies.json"
            # If movies.json is missing, try to find any JSON file in data
            if not data_path.exists():
                candidates = sorted(data_dir.glob("*.json"))
                if candidates:
                    # pick the first candidate and warn
                    data_path = candidates[0]
                    print(f"Warning: movies.json not found, using {data_path.name}")
                else:
                    print(f"Failed to load movies data: {data_path} does not exist")
                    return
            try:
                with open(data_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"Failed to load movies data: {e}")
                return

            movies = data.get("movies", []) if isinstance(data, dict) else []

            idx = InvertedIndex()
            idx.build(movies)
            idx.save()

            print("Built index.")

        case "tf":
            # Load cached index and print term frequency for a document-term
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return

            try:
                tf_val = idx.get_tf(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(str(tf_val))
            return
        case "idf":
            # Load cached index and compute inverse document frequency for a term
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return

            term = args.term
            # number of documents in the corpus
            N = len(idx.docmap)
            # document frequency for the normalized term
            df = len(idx.get_documents(term))
            # Use smoothed IDF formula: log((N + 1) / (df + 1)) to avoid
            # division by zero and provide a small amount of smoothing.
            idf = math.log((N + 1) / (df + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            return
        case "tfidf":
            # Load cached index and compute TF-IDF for a document-term pair
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return

            try:
                tf_val = idx.get_tf(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            N = len(idx.docmap)
            df = len(idx.get_documents(args.term))
            idf = math.log((N + 1) / (df + 1))

            tf_idf = tf_val * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            return
        case "bm25idf":
            # Use bm25_idf_command helper to compute score and print it
            try:
                bm25idf = bm25_idf_command(args.term)
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            return
        case "bm25tf":
            # Compute BM25 TF for a document-term pair (supports k1 and b)
            # Decide which k1/b to use: flags override positional fallbacks.
            k1_value = args.k1 if getattr(args, "k1", None) is not None else (
                args.k1_pos if getattr(args, "k1_pos", None) is not None else BM25_K1
            )
            b_value = args.b if getattr(args, "b", None) is not None else (
                args.b_pos if getattr(args, "b_pos", None) is not None else BM25_B
            )

            # Warn if user is using deprecated positional form so they can migrate
            if getattr(args, "k1_pos", None) is not None and getattr(args, "k1", None) is None:
                print("Warning: positional k1 argument is deprecated; use --k1 flag instead", file=sys.stderr)
            if getattr(args, "b_pos", None) is not None and getattr(args, "b", None) is None:
                print("Warning: positional b argument is deprecated; use --b flag instead", file=sys.stderr)

            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, k1_value, b_value)
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            return
        case "bm25search":
            # Rank documents with BM25 and print top-k
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return

            results = idx.bm25_search(args.query, args.limit)

            if not results:
                print("No results found.")
                return

            for rank, (did, score) in enumerate(results, start=1):
                doc = idx.docmap.get(did) or {}
                title = doc.get("title", "<untitled>")
                # Print in the requested format:
                # 1. (15) The Adventures of Mowgli - Score: 7.79
                print(f"{rank}. ({did}) {title} - Score: {score:.2f}")

            return
        case "search":
            # Use cached inverted index to answer the query
            print(f"Searching for: {args.query}")

            q_raw = args.query.strip()
            if not q_raw:
                print("No results found.")
                return

            # Prepare query tokens using same normalization pipeline
            _punct_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))
            sw_path = Path(__file__).resolve().parents[1] / "data" / "stopwords.txt"
            stopwords = set()
            if sw_path.exists():
                try:
                    sw_text = sw_path.read_text(encoding="utf-8")
                    stopwords = {w.strip().casefold() for w in sw_text.splitlines() if w.strip()}
                except OSError:
                    stopwords = set()

            q_lower = q_raw.casefold()
            q_norm = " ".join(q_lower.translate(_punct_trans).split())
            raw_tokens = [t for t in q_norm.split() if t]

            # Load the cached index
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Cached index not found. Please run: cli/keyword_search_cli.py build")
                return

            # Prepare token stream where operands are stemmed terms and
            # operators are upper-cased strings 'AND','OR','NOT'. We must
            # not remove operator words as stopwords here.
            token_stream: list[str] = []
            has_operator = False
            for t in raw_tokens:
                if t in OPERATORS:
                    # If NOT appears infix (e.g. 'bear NOT terror') many users
                    # mean 'bear AND NOT terror'. Detect that pattern and
                    # insert an implicit AND when the previous token is an
                    # operand (not another operator).
                    up = t.upper()
                    if up == "NOT" and token_stream and token_stream[-1] not in {
                        "AND",
                        "OR",
                        "NOT",
                    }:
                        token_stream.append("AND")
                    token_stream.append(up)
                    has_operator = True
                else:
                    if t in stopwords:
                        # skip stopwords in operands
                        continue
                    token_stream.append(stemmer.stem(t))

            if not token_stream:
                print("No results found.")
                return

            # If there are no operators, fall back to the previous
            # behavior: iterate tokens and collect union of postings in
            # token order until we have 5 results.
            if not has_operator:
                seen_ids: list[int] = []
                seen_set: set[int] = set()
                for qt in token_stream:
                    ids = idx.get_documents(qt)
                    for i in ids:
                        if i not in seen_set:
                            seen_set.add(i)
                            seen_ids.append(i)
                        if len(seen_ids) >= 5:
                            break
                    if len(seen_ids) >= 5:
                        break

                if not seen_ids:
                    print("No results found.")
                    return

                for rank, did in enumerate(seen_ids, start=1):
                    doc = idx.docmap.get(did) or {}
                    title = doc.get("title", "<untitled>")
                    print(f"{rank}. [{did}] {title}")

                return

            # Otherwise parse boolean expression (infix) to RPN using
            # shunting-yard and evaluate using set semantics.
            prec = {"NOT": 3, "AND": 2, "OR": 1}
            right_assoc = {"NOT"}
            output_queue: list[str] = []
            op_stack: list[str] = []

            for tok in token_stream:
                if tok in prec:
                    while op_stack:
                        top = op_stack[-1]
                        if top not in prec:
                            break
                        if (top in right_assoc and prec[top] > prec[tok]) or (
                            top not in right_assoc and prec[top] >= prec[tok]
                        ):
                            output_queue.append(op_stack.pop())
                        else:
                            break
                    op_stack.append(tok)
                else:
                    output_queue.append(tok)

            while op_stack:
                output_queue.append(op_stack.pop())

            universe = set(idx.docmap.keys())
            eval_stack: list[set[int]] = []
            try:
                for tok in output_queue:
                    if tok == "NOT":
                        a = eval_stack.pop()
                        eval_stack.append(universe - a)
                    elif tok == "AND":
                        b = eval_stack.pop()
                        a = eval_stack.pop()
                        eval_stack.append(a & b)
                    elif tok == "OR":
                        b = eval_stack.pop()
                        a = eval_stack.pop()
                        eval_stack.append(a | b)
                    else:
                        ids = set(idx.get_documents(tok))
                        eval_stack.append(ids)
            except IndexError:
                print("Malformed boolean query")
                return

            if not eval_stack:
                print("No results found.")
                return

            if len(eval_stack) != 1:
                print("Malformed boolean query")
                return

            result_ids = eval_stack.pop()

            if not result_ids:
                print("No results found.")
                return

            # Only need the 5 smallest IDs — use heapq.nsmallest to avoid
            # sorting the entire result set when it's large.
            results_sorted = heapq.nsmallest(5, (int(i) for i in result_ids))

            for rank, did in enumerate(results_sorted, start=1):
                doc = idx.docmap.get(did) or {}
                title = doc.get("title", "<untitled>")
                print(f"{rank}. [{did}] {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
