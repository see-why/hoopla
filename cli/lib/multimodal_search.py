"""
Multimodal search library for CLIP-based image and text embeddings.

This module provides utilities for generating image embeddings using CLIP models
(Contrastive Language-Image Pre-training), enabling multimodal search capabilities
that combine image and text information for improved movie database queries.

Classes:
    MultimodalSearch: Wrapper for SentenceTransformer CLIP models with image and text encoding

Functions:
    verify_image_embedding: Utility to validate image embedding generation and inspect embedding dimensions
    image_search_command: Search movie database using an image and return top matching movies
"""

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from cli.lib.semantic_search import load_movies_dataset
except ImportError:
    try:
        from lib.semantic_search import load_movies_dataset
    except ImportError:
        from semantic_search import load_movies_dataset


class MultimodalSearch:
    """
    Multimodal search using CLIP models via SentenceTransformer.
    
    Provides image and text encoding capabilities for searching movie databases
    using image and text queries. Embeddings from images can be compared with
    pre-computed text embeddings of movie documents for similarity-based search.
    
    Attributes:
        model: SentenceTransformer instance for CLIP model operations
        documents: List of movie documents with id, title, and description fields
        texts: List of formatted text strings (title: description) for each document
        text_embeddings: Pre-computed embeddings for all text documents (numpy array)
    """
    
    def __init__(self, documents: list | None = None, model_name: str = "clip-ViT-B-32"):
        """
        Initialize MultimodalSearch with a CLIP model and optional documents.
        
        Args:
            documents: Optional list of document dicts with 'id', 'title', and 'description' keys.
                      If provided, text embeddings will be pre-computed for efficient search.
            model_name: Name of the SentenceTransformer CLIP model to load.
                       Default: "clip-ViT-B-32" (Vision Transformer Base with 32x32 patches)
                       Other options: "clip-ViT-L-14", "clip-ViT-B-32-multilingual-v1", etc.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        
        # Pre-compute text representations and embeddings if documents provided
        if self.documents:
            # Create formatted text strings: "Title: Description"
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
            
            # Generate embeddings for all texts with progress bar
            self.text_embeddings = self.model.encode(
                self.texts,
                convert_to_tensor=False,
                show_progress_bar=True
            )
        else:
            self.texts = []
            self.text_embeddings = np.array([])
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for an image file using the CLIP model.
        
        Loads an image from the specified path and generates a vector embedding
        that captures visual features in a space compatible with text embeddings.
        
        Args:
            image_path: Path to the image file (supports JPEG, PNG, WebP, GIF, etc.)
        
        Returns:
            Embedding vector as a numpy array representing the image in embedding space.
            Shape is typically (embedding_dim,) where embedding_dim depends on the model
            (e.g., 512 for clip-ViT-B-32).
        
        Raises:
            FileNotFoundError: If the image file does not exist
            PIL.UnidentifiedImageError: If the file is not a valid image format
        """
        # Load image from path using PIL with context manager to ensure proper cleanup
        with Image.open(image_path) as image:
            # Generate embedding by passing image to model
            # encode() returns a numpy array of embeddings; extract the first (only) element
            embeddings = self.model.encode([image], convert_to_tensor=False)
            return embeddings[0]
    
    def search_with_image(self, image_path: str, top_k: int = 5) -> list:
        """
        Search documents using an image query.
        
        Generates an embedding for the provided image and compares it with
        pre-computed text embeddings to find the most similar documents.
        
        Args:
            image_path: Path to the query image file
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of dicts containing search results, each with keys:
            - 'id': Document ID
            - 'title': Movie title
            - 'description': Movie description
            - 'similarity': Cosine similarity score (0-1)
            Results are sorted by similarity in descending order.
        
        Raises:
            ValueError: If documents list is empty (no text embeddings computed)
            FileNotFoundError: If the image file does not exist
        """
        if not self.documents or len(self.text_embeddings) == 0:
            raise ValueError("MultimodalSearch must be initialized with documents for search")
        
        # Generate embedding for the query image
        image_embedding = self.embed_image(image_path)
        
        # Calculate cosine similarity between image and all text embeddings
        # Reshape image_embedding to (1, embedding_dim) for similarity computation
        similarities = cosine_similarity(
            [image_embedding],
            self.text_embeddings
        )[0]  # Extract the first (only) row
        
        # Create list of results with similarity scores
        results = []
        for idx, similarity_score in enumerate(similarities):
            results.append({
                'id': self.documents[idx].get('id'),
                'title': self.documents[idx]['title'],
                'description': self.documents[idx]['description'],
                'similarity': float(similarity_score)
            })
        
        # Sort by similarity score (descending) and return top_k
        results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return results_sorted[:top_k]


def verify_image_embedding(image_path: str) -> None:
    """
    Verify image embedding generation and display embedding dimensions.
    
    This utility function loads an image, generates its embedding using a CLIP model,
    and prints the embedding dimensions. Useful for validating that image embedding
    generation is working correctly and understanding the embedding space size.
    
    Args:
        image_path: Path to the image file to embed
    
    Raises:
        FileNotFoundError: If the image file does not exist
        PIL.UnidentifiedImageError: If the file is not a valid image format
    """
    # Create MultimodalSearch instance with default CLIP model
    multimodal = MultimodalSearch()
    
    # Generate embedding for the image
    embedding = multimodal.embed_image(image_path)
    
    # Print embedding shape information
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str, top_k: int = 5) -> list:
    """
    Search movie database using an image query.
    
    Loads the movies dataset, initializes a MultimodalSearch instance with
    pre-computed text embeddings, and performs image-based search to find
    the most similar movies.
    
    Args:
        image_path: Path to the query image file
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of dicts containing top matching movies, each with:
        - 'id': Movie ID
        - 'title': Movie title
        - 'description': Movie description
        - 'similarity': Cosine similarity score (0-1)
        Results sorted by similarity in descending order.
    
    Raises:
        FileNotFoundError: If the image file does not exist
        RuntimeError: If the movies dataset cannot be loaded
    """
    # Load the movies dataset
    documents, exc, movies_path = load_movies_dataset()
    
    if exc:
        raise RuntimeError(f"Failed to load movies dataset from {movies_path}: {exc}")
    
    # Create MultimodalSearch instance with documents
    multimodal = MultimodalSearch(documents=documents)
    
    # Perform image search
    results = multimodal.search_with_image(image_path, top_k=top_k)
    
    return results

