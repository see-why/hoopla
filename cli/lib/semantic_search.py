from sentence_transformers import SentenceTransformer


class SemanticSearch:
    """Lightweight wrapper around a SentenceTransformer model.

    Initializes the `all-MiniLM-L6-v2` model and exposes it as
    `self.model` for downstream usage.
    """

    def __init__(self) -> None:
        # Initialize the SentenceTransformer model instance
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

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

    # Determine dimensionality
    dim = getattr(getattr(emb, "shape", None), "__getitem__", None)
    if dim is not None:
        try:
            dims = emb.shape[0]
        except Exception:
            dims = len(emb)
    else:
        dims = len(emb)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {first3}")
    print(f"Dimensions: {dims}")
