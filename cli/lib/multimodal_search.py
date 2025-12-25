"""
Multimodal search library for CLIP-based image and text embeddings.

This module provides utilities for generating image embeddings using CLIP models
(Contrastive Language-Image Pre-training), enabling multimodal search capabilities
that combine image and text information for improved movie database queries.

Classes:
    MultimodalSearch: Wrapper for SentenceTransformer CLIP models with image encoding

Functions:
    verify_image_embedding: Utility to validate image embedding generation and inspect embedding dimensions
"""

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    """
    Multimodal search using CLIP models via SentenceTransformer.
    
    Provides image encoding capabilities for generating embeddings from image files.
    These embeddings can be used for similarity search, clustering, or multimodal
    retrieval tasks across images and text.
    
    Attributes:
        model: SentenceTransformer instance for CLIP model operations
    """
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize MultimodalSearch with a CLIP model.
        
        Args:
            model_name: Name of the SentenceTransformer CLIP model to load.
                       Default: "clip-ViT-B-32" (Vision Transformer Base with 32x32 patches)
                       Other options: "clip-ViT-L-14", "clip-ViT-B-32-multilingual-v1", etc.
        """
        self.model = SentenceTransformer(model_name)
    
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
        # Load image from path using PIL
        image = Image.open(image_path)
        
        # Generate embedding by passing image to model
        # encode() returns a list of embeddings; extract the first (only) element
        embeddings = self.model.encode([image], convert_to_tensor=False)
        return embeddings[0]


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
