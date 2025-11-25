try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None
from pathlib import Path
from typing import Any
import json
import logging
import re

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Lightweight wrapper around a SentenceTransformer model.

    Initializes the `all-MiniLM-L6-v2` model and exposes it as
    `self.model` for downstream usage.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Initialize the SentenceTransformer model instance lazily so
        # importing this module doesn't require the heavy dependency.
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except (ImportError, ModuleNotFoundError):
            # Model not available in this environment; keep `self.model`
            # set to None and defer the error until methods that require
            # the model are invoked.
            self.model = None
            logger.debug("sentence_transformers not available; model not loaded", exc_info=True)
        # embeddings storage and document bookkeeping
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, dict] = {}

    def generate_embedding(self, text: str):
        """Generate an embedding vector for `text` using the underlying model.

        Raises ValueError when `text` is empty or only whitespace.

        Returns the embedding for the single input (first element of the
        model.encode(...) result).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # SentenceTransformer.encode expects a list of inputs; we pass a
        # single-item list and return the first element from the result.
        result = self.model.encode([text])
        # result may be a list or numpy array; when passing a list with a
        # single element, the encoder typically returns an array-like where
        # the first element contains the embedding for the input.
        return result[0]

    def build_embeddings(self, documents: list[dict]):
        """Build embeddings for the provided list of document dicts.

        Each document is expected to be a dict with at least 'id', 'title',
        and 'description' keys. The method stores documents and a mapping
        from id -> document, encodes the concatenated strings for each
        document, saves the resulting embeddings to cache/movie_embeddings.npy,
        and returns the embeddings array.
        """
        # build document map and get the ordered list of accepted docs
        kept_docs = self._build_document_map(documents)
        texts = [f"{d.get('title') or ''}: {d.get('description') or ''}" for d in kept_docs]

        # encode with progress bar
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not loaded; install sentence_transformers to build embeddings")
        if np is None:
            raise RuntimeError("numpy is required to build embeddings; install numpy")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(embeddings)

        # persist to cache
        # project root is two levels up from cli/lib
        cache_dir = Path(__file__).resolve().parents[2] / "cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # If we cannot create the cache directory, continue and let
            # the subsequent save raise a clearer error (or the caller
            # can decide to ignore missing cache). Log at debug level so
            # developers can inspect the issue when needed.
            logger.debug("Could not create cache directory %s", cache_dir)
        out_path = cache_dir / "movie_embeddings.npy"
        np.save(str(out_path), self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        """Load embeddings from cache if present and valid, otherwise build.

        Ensures self.documents and self.document_map are populated.
        """
        # populate document bookkeeping consistently
        self._build_document_map(documents)

        cache_path = Path(__file__).resolve().parents[2] / "cache" / "movie_embeddings.npy"
        if cache_path.exists():
            try:
                arr = np.load(str(cache_path))
                # validate length
                if len(arr) == len(self.document_map):
                    self.embeddings = arr
                    return self.embeddings
            except Exception:
                # If loading the cached embeddings fails for any reason
                # (corrupt file, incompatible format, etc.) we fall back
                # to rebuilding the embeddings from the documents. Log
                # at debug level to aid diagnosis when debugging.
                logger.debug("Failed to load cached embeddings from %s, rebuilding", cache_path)

        # rebuild if missing or mismatched
        return self.build_embeddings(documents)

    def _build_document_map(self, documents: list[dict]) -> list[dict]:
        """Populate self.documents and self.document_map from documents.

        Returns an ordered list of documents that were accepted (skips
        documents lacking an 'id', invalid ids, or duplicate ids preserving
        the first occurrence).
        """
        self.documents = documents
        self.document_map = {}
        kept: list[dict] = []
        for d in documents:
            doc_id_raw = d.get("id")
            if doc_id_raw is None:
                continue
            try:
                doc_id = int(doc_id_raw)
            except (ValueError, TypeError):
                continue
            if doc_id in self.document_map:
                # preserve first occurrence; skip duplicates
                continue
            self.document_map[doc_id] = d
            kept.append(d)

        return kept

    def search(self, query: str, limit: int) -> list[dict]:
        """Search the loaded embeddings for the query and return top results.

        Raises ValueError if embeddings are not loaded.

        Returns a list of dicts with keys: 'score', 'title', 'description'.
        """
        # Validate query early to provide a clear error message before any
        # embedding/nearest-neighbor work is attempted.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Generate embedding for the query
        q_emb = self.generate_embedding(query)
        q_vec = np.asarray(q_emb)

        # Documents are stored in insertion order in self.document_map
        docs = list(self.document_map.values())

        n = min(len(self.embeddings), len(docs))

        # Vectorized cosine similarity computation for speed on large corpora
        # doc_vecs: (n, dim), q_vec: (dim,)
        doc_vecs = np.asarray(self.embeddings[:n], dtype=float)
        q_vec = np.asarray(q_vec, dtype=float)

        # compute norms
        q_norm = np.linalg.norm(q_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        # safe denominator to avoid division by zero; we'll zero-out scores
        # where either vector norm is zero after the division.
        denom = doc_norms * q_norm
        safe_denom = denom.copy()
        safe_denom[safe_denom == 0] = 1.0

        raw_scores = np.dot(doc_vecs, q_vec) / safe_denom
        # zero out entries where denom was zero
        raw_scores[denom == 0] = 0.0

        # Pair scores with documents
        scores = [(float(raw_scores[i]), docs[i]) for i in range(n)]

        # sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # normalize limit and slice
        try:
            limit_int = int(limit)
        except (ValueError, TypeError):
            limit_int = 5
        if limit_int < 0:
            limit_int = 0

        top = scores[:limit_int]
        results = []
        for score, doc in top:
            results.append({
                "score": score,
                "title": doc.get("title", "<untitled>"),
                "description": doc.get("description", ""),
            })

        return results


def verify_model() -> None:
    """Instantiate a SemanticSearch and print basic model info.

    Prints the model object (string representation) and the model's
    `max_seq_length` property when available.
    """
    ss = SemanticSearch()
    # Print the model representation
    print(f"Model loaded: {ss.model}")
    # Print maximum sequence length if available on the model
    max_len = getattr(ss.model, "max_seq_length", None)
    print(f"Max sequence length: {max_len}")


def embed_text(text: str) -> None:
    """Create a SemanticSearch, embed `text`, and print summary info.

    Prints the input text, the first three embedding dimensions, and the
    total embedding dimensionality.
    """
    ss = SemanticSearch()
    emb = ss.generate_embedding(text)

    # Some encoders return numpy arrays, others lists; compute length
    # portably and show first three values.
    try:
        first3 = emb[:3]
    except TypeError:
        # fallback if emb isn't sliceable
        first3 = list(emb)[:3]

    # Format first three dimensions to 3 decimal places for stable CLI output.
    # Convert to floats then format
    first3_list = [float(x) for x in first3]
    formatted_first3 = " ".join(f"{v:.3f}" for v in first3_list)

    # Determine dimensionality
    if hasattr(emb, 'shape'):
        dims = emb.shape[0]
    else:
        dims = len(emb)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {formatted_first3}")
    print(f"Dimensions: {dims}")


def embed_query_text(query: str) -> None:
    """Create a SemanticSearch, embed `query`, and print the query info.

    Prints the input query, the first five embedding dimensions, and the
    embedding shape. Uses the existing `generate_embedding` method which
    performs whitespace stripping and validation.
    """
    ss = SemanticSearch()
    emb = ss.generate_embedding(query)

    # First five dimensions (works for lists and numpy arrays)
    try:
        first5 = emb[:5]
    except Exception:
        first5 = list(emb)[:5]

    # Ensure we can report a shape: convert to numpy array if necessary
    try:
        shape = emb.shape
    except Exception:
        shape = np.asarray(emb).shape

    print(f"Query: {query}")
    print(f"First 5 dimensions: {first5}")
    print(f"Shape: {shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def load_movies_dataset(data_dir: Path | None = None):
    """Load the movies dataset from the repository `data/` directory.

    Returns a tuple: (documents_list, exception_or_None, Path_to_file).
    The caller may inspect the exception to decide whether to present a
    user-visible error (CLI) or simply log at debug level (library).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    movies_path = data_dir / "movies.json"
    if not movies_path.exists():
        movies_path = data_dir / "movies 2.json"

    try:
        loaded = json.loads(movies_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "movies" in loaded:
            docs = loaded["movies"]
        elif isinstance(loaded, list):
            docs = loaded
        else:
            docs = []
        return docs, None, movies_path
    except (OSError, json.JSONDecodeError) as exc:
        return [], exc, movies_path


def verify_embeddings() -> None:
    """Load movies and verify embeddings exist or are built, then print shape."""
    ss = SemanticSearch()
    docs, exc, movies_path = load_movies_dataset()
    if exc:
        logger.debug("Failed to load movies file %s: %s", movies_path, exc)

    embeddings = ss.load_or_create_embeddings(docs)
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings generated")
        return

    print(f"Number of docs:   {len(docs)}")
    # embeddings is a numpy array
    try:
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    except Exception:
        print(f"Embeddings shape: {len(embeddings)} vectors")


class ChunkedSemanticSearch(SemanticSearch):
    """SemanticSearch variant that stores precomputed chunk embeddings.

    Attributes:
        chunk_embeddings: numpy array of chunk embeddings (or None)
        chunk_metadata: list of metadata dicts corresponding to each chunk
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        """Build embeddings for semantic chunks from the provided documents.

        Splits each document description into sentence-based chunks (4
        sentences per chunk, 1 sentence overlap), encodes all chunks with
        the model, saves embeddings and metadata to cache, and returns the
        embeddings array.
        """
        # populate document bookkeeping and get ordered list of kept docs
        kept_docs = self._build_document_map(documents)

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        # chunking params as requested
        max_chunk_size = 4
        overlap = 1

        for doc in kept_docs:
            desc = (doc.get("description") or "").strip()
            if not desc:
                # skip docs with empty descriptions
                continue

            # split into sentences using the requested regex
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", desc) if s]

            # create chunks for this document
            step = max_chunk_size - overlap
            doc_chunks: list[str] = []
            i = 0
            while i < len(sentences):
                chunk_sents = sentences[i : i + max_chunk_size]
                if not chunk_sents:
                    break
                doc_chunks.append(" ".join(chunk_sents))
                if i + max_chunk_size >= len(sentences):
                    break
                i += step

            total_chunks = len(doc_chunks)
            for chunk_idx, chunk_text in enumerate(doc_chunks):
                all_chunks.append(chunk_text)
                # movie_idx should be the index in self.documents
                try:
                    movie_idx_in_docs = int(self.documents.index(doc))
                except ValueError:
                    movie_idx_in_docs = -1
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx_in_docs,
                        "chunk_idx": int(chunk_idx),
                        "total_chunks": int(total_chunks),
                    }
                )

        # encode all chunks (may be empty)
        if all_chunks:
            if self.model is None:
                raise RuntimeError("SentenceTransformer model not loaded; install sentence_transformers to build chunk embeddings")
            if np is None:
                raise RuntimeError("numpy is required to build chunk embeddings; install numpy")
            embeddings = self.model.encode(all_chunks, show_progress_bar=True)
            self.chunk_embeddings = np.array(embeddings)
        else:
            # empty 2D array with shape (0, embedding_dim)
            # Use a placeholder dimension that matches typical model output (384 for all-MiniLM-L6-v2)
            # This ensures consistent 2D shape even when there are no chunks
            embedding_dim = 384 if self.model is None else self.model.get_sentence_embedding_dimension()
            self.chunk_embeddings = np.zeros((0, embedding_dim)) if np is not None else None

        self.chunk_metadata = chunk_metadata

        # persist embeddings and metadata
        cache_dir = Path(__file__).resolve().parents[2] / "cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug("Could not create cache directory %s", cache_dir)

        emb_path = cache_dir / "chunk_embeddings.npy"
        np.save(str(emb_path), self.chunk_embeddings)

        meta_path = cache_dir / "chunk_metadata.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, fh, indent=2)
        except OSError:
            logger.debug("Failed to write chunk metadata to %s", meta_path)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> Any:
        """Load chunk embeddings and metadata from cache if present, otherwise build.

        Populates `self.documents` and `self.document_map` from `documents`.
        If both `cache/chunk_embeddings.npy` and `cache/chunk_metadata.json`
        exist, load them into `self.chunk_embeddings` and
        `self.chunk_metadata` and return the embeddings. Otherwise call
        `build_chunk_embeddings` to create and persist them.
        """
        # populate document bookkeeping
        self._build_document_map(documents)

        cache_dir = Path(__file__).resolve().parents[2] / "cache"
        emb_path = cache_dir / "chunk_embeddings.npy"
        meta_path = cache_dir / "chunk_metadata.json"

        if emb_path.exists() and meta_path.exists():
            try:
                if np is None:
                    raise RuntimeError("numpy is required to load chunk embeddings; install numpy")
                arr = np.load(str(emb_path))
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                chunks = meta.get("chunks", [])

                # assign loaded values
                self.chunk_embeddings = arr
                self.chunk_metadata = chunks

                return self.chunk_embeddings
            except Exception:
                logger.debug("Failed to load cached chunk embeddings or metadata; rebuilding", exc_info=True)

        # fallback: rebuild
        return self.build_chunk_embeddings(documents)
