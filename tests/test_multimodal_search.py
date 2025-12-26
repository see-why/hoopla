"""
Tests for the multimodal_search library module.

Tests the MultimodalSearch class and verify_image_embedding function
for image embedding generation using CLIP models.
"""

import tempfile
from PIL import Image
import numpy as np

from cli.lib.multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command


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
    
    def test_multimodal_search_with_documents(self):
        """Test MultimodalSearch initialization with documents."""
        documents = [
            {'id': '1', 'title': 'Movie 1', 'description': 'A great film'},
            {'id': '2', 'title': 'Movie 2', 'description': 'Another great film'},
        ]
        ms = MultimodalSearch(documents=documents)
        assert ms.documents == documents
        assert len(ms.texts) == 2
        assert ms.texts[0] == 'Movie 1: A great film'
        assert ms.texts[1] == 'Movie 2: Another great film'
        assert ms.text_embeddings is not None
        assert len(ms.text_embeddings) == 2
        assert ms.text_embeddings.shape[1] == 512  # CLIP-ViT-B-32 dimension
    
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
    
    def test_search_with_image_no_documents(self):
        """Test search_with_image raises error when no documents provided."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            ms = MultimodalSearch()  # No documents
            try:
                ms.search_with_image(tmp_path)
                assert False, "Should raise ValueError"
            except ValueError as e:
                assert "documents" in str(e).lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_search_with_image(self):
        """Test image search with sample documents."""
        documents = [
            {'id': '1', 'title': 'Red Movie', 'description': 'A red colored film'},
            {'id': '2', 'title': 'Blue Movie', 'description': 'A blue colored film'},
            {'id': '3', 'title': 'Green Movie', 'description': 'A green colored film'},
            {'id': '4', 'title': 'Yellow Movie', 'description': 'A yellow colored film'},
            {'id': '5', 'title': 'Purple Movie', 'description': 'A purple colored film'},
        ]
        
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            ms = MultimodalSearch(documents=documents)
            results = ms.search_with_image(tmp_path, top_k=3)
            
            # Verify results structure
            assert len(results) == 3
            assert results[0]['similarity'] >= results[1]['similarity'] >= results[2]['similarity']
            
            for result in results:
                assert 'id' in result
                assert 'title' in result
                assert 'description' in result
                assert 'similarity' in result
                assert 0 <= result['similarity'] <= 1
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
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
    
    def test_image_search_command_success(self):
        """Test image_search_command with valid image."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='green')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # This should load the actual movies dataset and search
            results = image_search_command(tmp_path, top_k=3)
            
            # Verify results structure
            assert isinstance(results, list)
            assert len(results) <= 3  # Should return at most top_k results
            
            # Verify each result has required fields
            for result in results:
                assert 'id' in result
                assert 'title' in result
                assert 'description' in result
                assert 'similarity' in result
                assert isinstance(result['similarity'], float)
                assert 0 <= result['similarity'] <= 1
            
            # Verify results are sorted by similarity (descending)
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i]['similarity'] >= results[i + 1]['similarity']
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_image_search_command_custom_top_k(self):
        """Test image_search_command with custom top_k parameter."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (128, 128), color='orange')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Request 10 results
            results = image_search_command(tmp_path, top_k=10)
            
            # Should return at most 10 results
            assert len(results) <= 10
            assert isinstance(results, list)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_image_search_command_file_not_found(self):
        """Test image_search_command raises FileNotFoundError for non-existent image."""
        try:
            image_search_command("/nonexistent/path/to/image.jpg")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    
    def test_image_search_command_dataset_error(self, monkeypatch):
        """Test image_search_command raises RuntimeError when dataset loading fails."""
        # Create a valid test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='cyan')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Mock load_movies_dataset to simulate failure
            def mock_load_movies_dataset():
                return None, Exception("Dataset load error"), "/fake/path/movies.json"
            
            # Patch the load_movies_dataset function
            import cli.lib.multimodal_search as ms_module
            monkeypatch.setattr(ms_module, 'load_movies_dataset', mock_load_movies_dataset)
            
            # Should raise RuntimeError with dataset loading error
            try:
                image_search_command(tmp_path)
                assert False, "Should raise RuntimeError"
            except RuntimeError as e:
                assert "Failed to load movies dataset" in str(e)
                assert "Dataset load error" in str(e)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
