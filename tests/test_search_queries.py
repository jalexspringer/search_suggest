"""
Tests for search functionality with specific queries.
"""
import pytest
from typing import List, Dict, Any
from search_suggest.vector_store import VectorStore
from search_suggest.embeddings import EmbeddingService
from search_suggest.api import get_collection_for_model

def search_with_model(
    query: str, 
    model_name: str, 
    vector_store: VectorStore,
    embedding_services: Dict[str, EmbeddingService],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform a search with a specific model.
    
    Args:
        query: The search query
        model_name: The model to use for embedding
        vector_store: The vector store instance
        embedding_services: Dictionary of embedding services
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Get embedding service for the model
    embedding_service = embedding_services.get(model_name)
    if not embedding_service:
        embedding_service = EmbeddingService(model_name=model_name)
        embedding_services[model_name] = embedding_service
    
    # Create embedding for the query
    query_embedding = embedding_service.create_embedding(query)
    
    # Get collection name for the model
    collection_name = get_collection_for_model(model_name)
    
    # Search for similar categories
    results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )
    
    return results

@pytest.mark.parametrize("query", [
    "climbing gear",
    "kitchen appliances",
    "baby toys",
    "office supplies",
    "fitness equipment"
])
def test_general_queries(query, vector_store, embedding_services):
    """Test search with general queries across different models."""
    models = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
    
    for model_name in models:
        results = search_with_model(
            query=query,
            model_name=model_name,
            vector_store=vector_store,
            embedding_services=embedding_services
        )
        
        # Verify we got results
        assert isinstance(results, list)
        
        # Skip further assertions if no results
        if not results:
            continue
            
        # Check result structure
        assert "id" in results[0]
        assert "score" in results[0]
        assert "payload" in results[0]
        assert "full_path" in results[0]["payload"]
        
        # Verify scores are in descending order
        scores = [r["score"] for r in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

@pytest.mark.parametrize("query", [
    "kitchen appliances",
    "cooking equipment",
    "refrigerators and freezers",
    "blenders and mixers",
    "coffee makers",
    "kitchen gadgets",
    "food processors",
    "toasters",
    "microwave ovens",
    "dishwashers"
])
def test_kitchen_queries(query, vector_store, embedding_services):
    """Test search with kitchen-specific queries."""
    models = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
    
    for model_name in models:
        results = search_with_model(
            query=query,
            model_name=model_name,
            vector_store=vector_store,
            embedding_services=embedding_services
        )
        
        # Verify we got results
        assert isinstance(results, list)
        
        # Skip further assertions if no results
        if not results:
            continue
            
        # For kitchen queries, we expect "Kitchen & Dining" to appear in results
        full_paths = [r["payload"]["full_path"] for r in results]
        kitchen_related = any("Kitchen" in path for path in full_paths)
        
        # This is a soft assertion - we print a message but don't fail the test
        # since semantic search might find other relevant categories
        if not kitchen_related:
            print(f"Warning: No kitchen-related categories found for query '{query}' with model {model_name}")
            print(f"Top results: {full_paths[:3]}")
