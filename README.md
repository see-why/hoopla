hoopla — search CLI
=================================

This repository contains command-line search tools with both keyword-based and semantic (embedding-based) search capabilities, along with a test suite.

Features
--------

### Keyword Search
- Inverted index built from data/ JSON files and persisted to cache/ (index, docmap, term frequencies).
- Normalization pipeline: casefold, punctuation → spaces, stopword filtering (data/stopwords.txt), Porter stemming (NLTK).
- Search CLI with support for simple token queries and boolean expressions (AND, OR, NOT). Boolean queries are parsed to RPN and evaluated with set semantics.
- Term-frequency (TF), inverse document frequency (IDF), and TF‑IDF utilities (CLI subcommands).

### Semantic Search
- Embedding-based search using sentence-transformers (all-MiniLM-L6-v2 model).
- Full document embeddings with cosine similarity search.
- Chunked semantic search: splits documents into sentence-based chunks (4 sentences per chunk, 1 sentence overlap) for more granular matching.
- Document chunking utilities for both word-based and sentence-based splitting with configurable overlap.
- Embeddings cached to disk for fast subsequent searches.

Quick start
-----------
Prerequisites
- Python 3.10+ (project uses a virtualenv `.venv` recommended)
- Install dev requirements for tests, NLTK, and sentence-transformers:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
# NLTK data: the code uses PorterStemmer (no external download required)
# sentence-transformers: downloads all-MiniLM-L6-v2 model on first use
```

Build the keyword search index cache

```bash
# Rebuild the index from data/*.json (writes cache/index.pkl, cache/docmap.pkl, cache/term_frequencies.pkl)
.venv/bin/python cli/keyword_search_cli.py build
```

Build the semantic search embeddings cache (optional, auto-built on first search)

```bash
# Build full document embeddings (writes cache/movie_embeddings.npy)
.venv/bin/python -m cli.semantic_search_cli verify_embeddings

# Build chunked embeddings (writes cache/chunk_embeddings.npy, cache/chunk_metadata.json)
.venv/bin/python -m cli.semantic_search_cli embed_chunks
```

CLI usage
---------

### Keyword Search CLI

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

### Semantic Search CLI

The entrypoint is `cli/semantic_search_cli.py`. Use the project's venv Python to ensure sentence-transformers is available.

- Search movies using full document embeddings:

```bash
.venv/bin/python -m cli.semantic_search_cli search "romantic comedy" --limit 5
```

- Search movies using chunked embeddings (more granular matching):

```bash
.venv/bin/python -m cli.semantic_search_cli search_chunked "space adventure" --limit 5
```

- Split text into word-based chunks:

```bash
.venv/bin/python -m cli.semantic_search_cli chunk "The quick brown fox" --size 3 --overlap 1
# Output:
# 1. The quick brown
# 2. brown fox
```

- Split text into sentence-based chunks:

```bash
.venv/bin/python -m cli.semantic_search_cli semantic_chunk "First sentence. Second sentence. Third." --max-chunk-size 2 --overlap 1
# Output:
# 1. First sentence. Second sentence.
# 2. Second sentence. Third.
```

- Verify model and generate embeddings:

```bash
# Verify sentence-transformers model loads
.venv/bin/python -m cli.semantic_search_cli verify

# Generate embedding for text and show dimensions
.venv/bin/python -m cli.semantic_search_cli embed_text "sample text"

# Embed a query and show first 5 dimensions
.venv/bin/python -m cli.semantic_search_cli embedquery "movie query"
```

Implementation notes
--------------------

### Keyword Search
- The inverted index is implemented in `cli/keyword_search_cli.py` (class `InvertedIndex`). It supports `build()`, `save()`, and `load()` which persist to `cache/` by default.
- Text normalization: casefold → translate punctuation to spaces → split → remove stopwords (data/stopwords.txt) → Porter stemming (nltk.PorterStemmer).
- Boolean parsing: shunting-yard to RPN, then evaluate with set operations. A short-hand `A NOT B` is treated as `A AND NOT B` (implicit AND).
- The `search` command will print up to 5 document results. For boolean queries it computes the result set and returns the 5 smallest document ids (using a heap to avoid sorting the entire set); for non-boolean queries it preserves token ordering and unions results until 5 results are found.

### Semantic Search
- Implemented in `cli/lib/semantic_search.py` using sentence-transformers library.
- `SemanticSearch` class: handles full document embeddings with cosine similarity search.
- `ChunkedSemanticSearch` class: extends SemanticSearch to support chunked embeddings:
  - Documents split into 4-sentence chunks with 1-sentence overlap.
  - Each chunk embedded separately for granular matching.
  - Search aggregates chunk scores by document (keeps highest scoring chunk per document).
- Embeddings cached as numpy arrays in `cache/`:
  - `movie_embeddings.npy`: full document embeddings
  - `chunk_embeddings.npy`: chunked embeddings
  - `chunk_metadata.json`: mapping of chunks to documents
- Model: all-MiniLM-L6-v2 (384-dimensional embeddings, max sequence length 256 tokens).
- Lazy imports: heavy dependencies (numpy, sentence-transformers) only loaded when needed.

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
- The semantic search functionality requires ~100MB for the sentence-transformers model and generates embeddings that are cached. First run may take longer due to model download and embedding generation.
- Chunked search is more suitable for finding specific passages or concepts within documents, while full document search works better for overall document similarity.

License
-------
MIT
