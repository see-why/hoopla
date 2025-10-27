hoopla — keyword search CLI
=================================

This repository contains a small command-line keyword search tool and a tiny test suite.

Features
--------
- Inverted index built from data/ JSON files and persisted to cache/ (index, docmap, term frequencies).
- Normalization pipeline: casefold, punctuation -> spaces, stopword filtering (data/stopwords.txt), Porter stemming (NLTK).
- Search CLI with support for simple token queries and boolean expressions (AND, OR, NOT). Boolean queries are parsed to RPN and evaluated with set semantics.
- Term-frequency (TF), inverse document frequency (IDF), and TF‑IDF utilities (CLI subcommands).

Quick start
-----------
Prerequisites
- Python 3.10+ (project uses a virtualenv `.venv` recommended)
- Install dev requirements for tests and NLTK:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
# NLTK data: the code uses PorterStemmer (no external download required)
```

Build the index cache

```bash
# Rebuild the index from data/*.json (writes cache/index.pkl, cache/docmap.pkl, cache/term_frequencies.pkl)
.venv/bin/python cli/keyword_search_cli.py build
```

CLI usage
---------
The entrypoint is `cli/keyword_search_cli.py`. Use the project's venv Python to ensure NLTK is available.

- Search (simple tokens or boolean expressions):

```bash
.venv/bin/python cli/keyword_search_cli.py search "bear AND wizard"
```

- Get term frequency (TF) for a term in a document (doc id must exist in the index):

```bash
.venv/bin/python cli/keyword_search_cli.py tf 424 bear
```

- Get inverse document frequency (IDF) for a term (smoothed):

```bash
.venv/bin/python cli/keyword_search_cli.py idf bear
# prints: Inverse document frequency of 'bear': <value>
```

IDF formula used
- idf = ln((N + 1) / (df + 1))
  - N = total number of documents in the index
  - df = number of documents containing the (normalized) term

- Compute TF‑IDF for a document-term pair:

```bash
.venv/bin/python cli/keyword_search_cli.py tfidf 424 trapper
# prints: TF-IDF score of 'trapper' in document '424': <value>
```

Implementation notes
--------------------
- The inverted index is implemented in `cli/keyword_search_cli.py` (class `InvertedIndex`). It supports `build()`, `save()`, and `load()` which persist to `cache/` by default.
- Text normalization: casefold -> translate punctuation to spaces -> split -> remove stopwords (data/stopwords.txt) -> Porter stemming (nltk.PorterStemmer).
- Boolean parsing: shunting-yard to RPN, then evaluate with set operations. A short-hand `A NOT B` is treated as `A AND NOT B` (implicit AND).
- The `search` command will print up to 5 document results. For boolean queries it computes the result set and returns the 5 smallest document ids (using a heap to avoid sorting the entire set); for non-boolean queries it preserves token ordering and unions results until 5 results are found.

Testing
-------
Run the small pytest suite (uses the project's `.venv` Python to ensure NLTK is available):

```bash
.venv/bin/python -m pytest -q
```

Notes & suggestions
-------------------
- Tests call the CLI `build` subcommand and can overwrite the repository-level `cache/`. Consider adding a `--cache-dir` option or making tests use temporary cache directories to avoid clobbering a developer cache.
- Parentheses in boolean expressions are not yet supported. They can be added by extending the tokenizer and shunting-yard logic.

License
-------
MIT
