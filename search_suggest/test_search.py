"""
Test script to verify search functionality with local embeddings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from search_suggest.embeddings import EmbeddingService
from search_suggest.vector_store import VectorStore

def test_search():
    """Test search functionality with local embeddings."""
    # Load environment variables
    load_dotenv()
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
        return False
    
    # Initialize embedding service with local model
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    
    # Test queries
    test_queries = [
        "climbing gear",
        "kitchen appliances",
        "baby toys",
        "office supplies",
        "fitness equipment"
    ]
    
    collection_name = "merchant_categories_enriched_local"
    
    print(f"Testing search with collection: {collection_name}")
    print(f"Using local embedding model: all-MiniLM-L6-v2")
    
    for query in test_queries:
        print(f"\nSearch query: '{query}'")
        
        # Create embedding for the query
        query_embedding = embedding_service.create_embedding(query)
        
        # Search for similar categories
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5
        )
        
        # Print results
        print(f"Top 5 results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['payload']['full_path']} (Score: {result['score']:.4f})")

if __name__ == "__main__":
    test_search()
