from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json


class SemanticSearch:
    """Lightweight wrapper around a SentenceTransformer model.

    Initializes the `all-MiniLM-L6-v2` model and exposes it as
    `self.model` for downstream usage.
    """

    def __init__(self) -> None:
        # Initialize the SentenceTransformer model instance
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(embeddings)

        # persist to cache
        # project root is two levels up from cli/lib
        cache_dir = Path(__file__).resolve().parents[2] / "cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
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
                if len(arr) == len(documents):
                    self.embeddings = arr
                    return self.embeddings
            except Exception:
                # fall through to rebuild
                pass

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


def verify_embeddings() -> None:
    """Load movies and verify embeddings exist or are built, then print shape."""
    ss = SemanticSearch()
    data_dir = Path(__file__).resolve().parents[2] / "data"
    movies_path = data_dir / "movies.json"
    if not movies_path.exists():
        movies_path = data_dir / "movies 2.json"
    try:
        loaded = json.loads(movies_path.read_text(encoding="utf-8"))
        # many datasets wrap the list under a key like 'movies'
        if isinstance(loaded, dict) and "movies" in loaded:
            docs = loaded["movies"]
        elif isinstance(loaded, list):
            docs = loaded
        else:
            docs = []
    except Exception:
        docs = []

    embeddings = ss.load_or_create_embeddings(docs)
    if embeddings is None:
        print("No embeddings generated")
        return

    print(f"Number of docs:   {len(docs)}")
    # embeddings is a numpy array
    try:
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    except Exception:
        print(f"Embeddings shape: {len(embeddings)} vectors")
