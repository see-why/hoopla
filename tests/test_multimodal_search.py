"""
Tests for the multimodal_search library module.

Tests the MultimodalSearch class and verify_image_embedding function
for image embedding generation using CLIP models.
"""

import tempfile
from PIL import Image
import numpy as np

from cli.lib.multimodal_search import MultimodalSearch, verify_image_embedding


class TestMultimodalSearch:
    """Tests for MultimodalSearch class"""
    
    def test_multimodal_search_initialization(self):
        """Test that MultimodalSearch initializes with a model."""
        ms = MultimodalSearch()
        assert ms.model is not None
        assert hasattr(ms.model, 'encode')
    
    def test_multimodal_search_custom_model(self):
        """Test MultimodalSearch initialization with custom model name."""
        ms = MultimodalSearch(model_name="clip-ViT-B-32")
        assert ms.model is not None
    
    def test_embed_image(self):
        """Test image embedding generation."""
        # Create a temporary test image (simple red square)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create a small test image
            img = Image.new('RGB', (64, 64), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            ms = MultimodalSearch()
            embedding = ms.embed_image(tmp_path)
            
            # Verify embedding is a numpy array
            assert isinstance(embedding, np.ndarray)
            
            # Verify embedding has correct shape (1D vector)
            assert len(embedding.shape) == 1
            
            # Verify embedding dimension matches CLIP model output
            # clip-ViT-B-32 produces 512-dimensional embeddings
            assert embedding.shape[0] == 512
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_embed_image_file_not_found(self):
        """Test embed_image raises error for non-existent file."""
        ms = MultimodalSearch()
        try:
            ms.embed_image("/nonexistent/image.jpg")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    
    def test_verify_image_embedding(self):
        """Test verify_image_embedding function."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='blue')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # This should run without errors and print output
            verify_image_embedding(tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
