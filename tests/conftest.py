"""
Shared fixtures for pytest.
"""
import os
import pytest
from dotenv import load_dotenv
from search_suggest.embeddings import EmbeddingService
from search_suggest.vector_store import VectorStore

# Load environment variables before tests
load_dotenv()

@pytest.fixture(scope="session")
def vector_store():
    """Create a vector store instance for testing."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        pytest.skip("QDRANT_URL or QDRANT_API_KEY environment variable is not set")
    
    return VectorStore(url=qdrant_url, api_key=qdrant_api_key)

@pytest.fixture(scope="session")
def embedding_service():
    """Create an embedding service instance with the default model."""
    return EmbeddingService()

@pytest.fixture(scope="session")
def embedding_services():
    """Create embedding services for different models."""
    models = [
        "all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5"
    ]
    
    services = {}
    for model_name in models:
        services[model_name] = EmbeddingService(model_name=model_name)
    
    return services
