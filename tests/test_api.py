"""
Tests for the API endpoints.
"""
import pytest
import httpx
from fastapi.testclient import TestClient
from search_suggest.api import app

@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    # Root endpoint returns HTML, not JSON
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text

def test_models_endpoint(client):
    """Test the models endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    
    # Response should be a dictionary of models
    models = response.json()
    assert isinstance(models, dict)
    assert len(models) > 0
    
    # Check model info structure
    for model_name, model_info in models.items():
        assert "dimension" in model_info
        assert "quality" in model_info

def test_collections_endpoint(client):
    """Test the collections endpoint."""
    response = client.get("/collections")
    assert response.status_code == 200
    
    # Response should be a list of collections
    collections = response.json()
    assert isinstance(collections, list)
    
    # Check collection structure if any exist
    if collections:
        assert "name" in collections[0]

def test_search_endpoint(client):
    """Test the search endpoint."""
    # Test with default parameters
    response = client.get("/search", params={"query": "kitchen appliances"})
    assert response.status_code == 200
    
    # Response should be a list of search results
    results = response.json()
    assert isinstance(results, list)
    
    # Check result structure if any exist
    if results:
        assert "id" in results[0]
        assert "score" in results[0]
        assert "full_path" in results[0]
        assert "level" in results[0]
    
    # Test with specific model
    response = client.get("/search", params={
        "query": "kitchen appliances",
        "model": "all-MiniLM-L6-v2",
        "limit": 3
    })
    assert response.status_code == 200

def test_compare_endpoint(client):
    """Test the compare endpoint."""
    # Test comparison with multiple models
    response = client.post("/compare", json={
        "query": "kitchen appliances",
        "models": ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        "limit": 3
    })
    assert response.status_code == 200
    
    # Response should be a list of comparison results
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check result structure
    for model_result in results:
        assert "model" in model_result
        assert "query_time_ms" in model_result
        assert "results" in model_result
        assert isinstance(model_result["results"], list)
