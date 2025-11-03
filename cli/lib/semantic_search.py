from sentence_transformers import SentenceTransformer


class SemanticSearch:
    """Lightweight wrapper around a SentenceTransformer model.

    Initializes the `all-MiniLM-L6-v2` model and exposes it as
    `self.model` for downstream usage.
    """

    def __init__(self) -> None:
        # Initialize the SentenceTransformer model instance
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


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
