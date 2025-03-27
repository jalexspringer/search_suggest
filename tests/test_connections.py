"""
Tests for connections to external services like Qdrant.
"""
import os
import pytest
from qdrant_client import QdrantClient

def test_qdrant_connection(vector_store):
    """Test connection to Qdrant vector database."""
    # The vector_store fixture already tests the connection
    # Just verify that the client is properly initialized
    assert vector_store.client is not None
    
    # Verify we can get collections list
    collections = vector_store.list_collections()
    assert isinstance(collections, list)
