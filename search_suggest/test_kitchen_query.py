"""
Test script to compare different models for kitchen appliances queries.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from search_suggest.embeddings import EmbeddingService
from search_suggest.vector_store import VectorStore

def test_kitchen_queries():
    """Test different models on kitchen-related queries."""
    # Load environment variables
    load_dotenv()
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
        return False
    
    # Test queries focused on kitchen appliances
    test_queries = [
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
    ]
    
    # Models to test
    models = [
        {
            "name": "all-MiniLM-L6-v2",
            "collection": "merchant_categories_all-MiniLM-L6-v2_test"
        },
        {
            "name": "BAAI/bge-small-en-v1.5",
            "collection": "merchant_categories_BAAI_bge-small-en-v1.5_test"
        },
        {
            "name": "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
            "collection": "merchant_categories_msmarco"
        }
    ]
    
    # Test each model
    for model_info in models:
        model_name = model_info["name"]
        collection_name = model_info["collection"]
        
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model_name}")
        print(f"Collection: {collection_name}")
        print(f"{'=' * 80}")
        
        # Initialize embedding service with the model
        embedding_service = EmbeddingService(model_name=model_name)
        vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
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
                print(f"  {i+1}. {result['payload'].get('full_path', 'N/A')} (Score: {result['score']:.4f})")

if __name__ == "__main__":
    test_kitchen_queries()
