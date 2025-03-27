"""
Tests for the embedding service functionality.
"""
import pytest
from search_suggest.embeddings import EmbeddingService

def test_embedding_service_initialization():
    """Test that the embedding service initializes correctly with different models."""
    # Test with default model
    service = EmbeddingService()
    assert service.model is not None
    assert service.model_name == "BAAI/bge-small-en-v1.5"
    
    # Test with specific model
    service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    assert service.model is not None
    assert service.model_name == "all-MiniLM-L6-v2"

def test_create_embedding():
    """Test creating embeddings with different models."""
    service = EmbeddingService()
    
    # Test creating an embedding
    text = "Test embedding text"
    embedding = service.create_embedding(text)
    
    # Verify embedding format
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Test with a different model
    embedding2 = service.create_embedding(text, model_name="all-MiniLM-L6-v2")
    assert isinstance(embedding2, list)
    assert len(embedding2) > 0
    
    # Embeddings from different models should be different
    assert embedding != embedding2

def test_create_embeddings_batch():
    """Test creating embeddings for a batch of texts."""
    service = EmbeddingService()
    
    # Test batch embedding
    texts = ["First test text", "Second test text", "Third test text"]
    embeddings = service.create_embeddings_batch(texts)
    
    # Verify batch results
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)

def test_model_caching():
    """Test that models are properly cached."""
    service = EmbeddingService()
    
    # Use a different model
    service.create_embedding("Test", model_name="all-MiniLM-L6-v2")
    
    # Check that the model was cached
    assert "all-MiniLM-L6-v2" in service.models
    
    # The original model should still be cached
    assert service.model_name in service.models
