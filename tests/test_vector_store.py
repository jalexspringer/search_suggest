"""
Tests for the vector store functionality.
"""
import pytest
from search_suggest.vector_store import VectorStore

def test_vector_store_initialization(vector_store):
    """Test that the vector store initializes correctly."""
    assert vector_store.client is not None

def test_list_collections(vector_store):
    """Test getting collections from the vector store."""
    collections = vector_store.list_collections()
    assert isinstance(collections, list)

def test_search_functionality(vector_store, embedding_service):
    """Test search functionality with a real query."""
    # Create a test query embedding
    query = "kitchen appliances"
    query_embedding = embedding_service.create_embedding(query)
    
    # Get a collection name that exists
    collections = vector_store.list_collections()
    if not collections:
        pytest.skip("No collections available for testing search")
    
    collection_name = collections[0]["name"]
    
    # Perform search
    results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )
    
    # Verify results format
    assert isinstance(results, list)
    if results:  # If we got any results
        assert "id" in results[0]
        assert "score" in results[0]
        assert "payload" in results[0]
