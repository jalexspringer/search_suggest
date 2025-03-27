"""
Script to compare different embedding models for search quality.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from tabulate import tabulate
from search_suggest.embeddings import EmbeddingService, RECOMMENDED_MODELS
from search_suggest.vector_store import VectorStore

# Test queries that should return relevant categories
TEST_QUERIES = [
    "kitchen appliances",
    "cooking equipment",
    "refrigerators and freezers",
    "blenders and mixers",
    "coffee makers",
    "climbing gear",
    "baby toys",
    "office supplies",
    "fitness equipment",
    "electronics",
    "smartphones",
    "laptops",
    "clothing",
    "shoes",
    "furniture"
]

def populate_collection_with_model(model_name, collection_suffix="_test"):
    """Populate a test collection with the specified model.
    
    Args:
        model_name: Name of the model to use
        collection_suffix: Suffix to add to the collection name
    
    Returns:
        Collection name
    """
    import subprocess
    
    collection_name = f"merchant_categories_{model_name.replace('/', '_')}{collection_suffix}"
    
    # Run the populate command with the specified model
    cmd = [
        "uv", "run", "python", "-m", "search_suggest.cli", "populate",
        "--collection", collection_name,
        "--embedding-model", model_name
    ]
    
    print(f"Populating collection {collection_name} with model {model_name}...")
    subprocess.run(cmd, check=True)
    
    return collection_name

def evaluate_model(model_name, collection_name, queries=TEST_QUERIES, top_k=5):
    """Evaluate a model on a set of test queries.
    
    Args:
        model_name: Name of the model to evaluate
        collection_name: Name of the collection to search
        queries: List of test queries
        top_k: Number of top results to consider
    
    Returns:
        Dictionary with evaluation results
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
        return None
    
    # Initialize embedding service with the specified model
    embedding_service = EmbeddingService(model_name=model_name)
    vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    
    results = []
    
    print(f"\nEvaluating model: {model_name}")
    print(f"Collection: {collection_name}")
    print(f"Embedding dimension: {embedding_service.embedding_dimension}")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Create embedding for the query
        query_embedding = embedding_service.create_embedding(query)
        
        # Search for similar categories
        search_results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Print results
        print(f"Top {top_k} results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. {result['payload'].get('full_path', 'N/A')} (Score: {result['score']:.4f})")
        
        # Store the results
        results.append({
            "query": query,
            "results": search_results
        })
    
    return results

def compare_models(model_names=None, use_existing_collections=False):
    """Compare multiple embedding models.
    
    Args:
        model_names: List of model names to compare
        use_existing_collections: Whether to use existing collections or create new ones
    """
    if model_names is None:
        model_names = list(RECOMMENDED_MODELS.keys())
    
    # Evaluate each model
    all_results = {}
    for model_name in model_names:
        if use_existing_collections:
            # Use an existing collection with the model name in it
            collection_name = f"merchant_categories_{model_name.replace('/', '_')}"
        else:
            # Create a new collection for this model
            collection_name = populate_collection_with_model(model_name)
        
        # Evaluate the model
        results = evaluate_model(model_name, collection_name)
        all_results[model_name] = results
    
    # Summarize the results
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    
    # Create a table with the model information
    table_data = []
    for model_name in model_names:
        model_info = RECOMMENDED_MODELS.get(model_name, {})
        table_data.append([
            model_name,
            model_info.get("dimension", "Unknown"),
            model_info.get("speed", "Unknown"),
            model_info.get("quality", "Unknown")
        ])
    
    print(tabulate(
        table_data,
        headers=["Model", "Dimension", "Speed", "Quality"],
        tablefmt="grid"
    ))
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2", "Alibaba-NLP/gte-small"],
        help="Models to compare"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing collections instead of creating new ones"
    )
    
    args = parser.parse_args()
    
    # Compare the models
    compare_models(args.models, args.use_existing)
