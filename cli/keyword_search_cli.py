#!/usr/bin/env python3

import argparse
import json
import re
import string
from nltk.stem import PorterStemmer
from pathlib import Path


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

        norm = " ".join(text.casefold().translate(self._punct_trans).split())
        tokens = [t for t in norm.split() if t and t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]

        for tok in tokens:
            postings = self.index.setdefault(tok, set())
            postings.add(int(doc_id))

    def get_documents(self, term: str) -> list[int]:
        """Return sorted list of document ids for a token/term."""
        if not term:
            return []
        key = self.stemmer.stem(term.casefold())
        ids = self.index.get(key, set())
        return sorted(int(i) for i in ids)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    # build subcommand for creating an inverted index cache
    build_parser = subparsers.add_parser("build", help="Build inverted index cache")

    args = parser.parse_args()

    # Initialize stemmer for token normalization
    stemmer = PorterStemmer()

    match args.command:
        case "build":
            # load movies file
            data_path = Path(__file__).resolve().parents[1] / "data" / "movies.json"
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

            # Recognize boolean operators (AND, OR, NOT) case-insensitively.
            operators = {"and", "or", "not"}

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
                if t in operators:
                    token_stream.append(t.upper())
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

            def is_op(x: str) -> bool:
                return x in prec

            for tok in token_stream:
                if is_op(tok):
                    while op_stack:
                        top = op_stack[-1]
                        if not is_op(top):
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

            universe = set(int(k) for k in idx.docmap.keys())
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

            result_ids = eval_stack.pop()

            if not result_ids:
                print("No results found.")
                return

            results_sorted = sorted(int(i) for i in result_ids)[:5]

            for rank, did in enumerate(results_sorted, start=1):
                doc = idx.docmap.get(did) or {}
                title = doc.get("title", "<untitled>")
                print(f"{rank}. [{did}] {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()