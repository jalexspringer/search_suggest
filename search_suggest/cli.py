"""
Command-line interface for search suggestions.
"""
import argparse
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tabulate import tabulate

from search_suggest.populate_db import populate_taxonomy_embeddings
from search_suggest.embeddings import RECOMMENDED_MODELS, EmbeddingService
from search_suggest.vector_store import VectorStore

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Search suggestions CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Populate command
    populate_parser = subparsers.add_parser("populate", help="Populate the database with taxonomy embeddings")
    populate_parser.add_argument(
        "--taxonomy-file", 
        default="data/taxonomy.txt", 
        help="Path to the taxonomy file"
    )
    populate_parser.add_argument(
        "--max-level", 
        type=int, 
        default=3, 
        help="Maximum level of categories to include"
    )
    populate_parser.add_argument(
        "--collection", 
        default="merchant_categories", 
        help="Name of the collection to populate"
    )
    populate_parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Name of the sentence-transformers model to use for embeddings"
    )
    
    # Generate embeddings for all models command
    generate_all_parser = subparsers.add_parser(
        "generate-all-models", 
        help="Generate embeddings for all recommended models"
    )
    generate_all_parser.add_argument(
        "--taxonomy-file", 
        default="data/taxonomy.txt", 
        help="Path to the taxonomy file"
    )
    generate_all_parser.add_argument(
        "--max-level", 
        type=int, 
        default=3, 
        help="Maximum level of categories to include"
    )
    generate_all_parser.add_argument(
        "--collection-prefix", 
        default="merchant_categories", 
        help="Prefix for collection names"
    )
    generate_all_parser.add_argument(
        "--test-suffix",
        default="_test",
        help="Suffix to append to collection names"
    )
    
    # List models command
    models_parser = subparsers.add_parser("list-models", help="List recommended embedding models")
    
    # List collections command
    collections_parser = subparsers.add_parser("list-collections", help="List collections in the vector store")
    
    # Delete collection command
    delete_parser = subparsers.add_parser("delete-collection", help="Delete a collection from the vector store")
    delete_parser.add_argument(
        "collection_name",
        help="Name of the collection to delete"
    )
    
    # Delete all collections command
    delete_all_parser = subparsers.add_parser("delete-all-collections", help="Delete all collections from the vector store")
    delete_all_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion without prompting"
    )
    
    args = parser.parse_args()
    
    if args.command == "populate":
        populate_taxonomy_embeddings(
            taxonomy_file=Path(args.taxonomy_file),
            max_level=args.max_level,
            collection_name=args.collection,
            embedding_model=args.embedding_model
        )
    elif args.command == "generate-all-models":
        generate_embeddings_for_all_models(
            taxonomy_file=Path(args.taxonomy_file),
            max_level=args.max_level,
            collection_prefix=args.collection_prefix,
            test_suffix=args.test_suffix
        )
    elif args.command == "list-models":
        print("Recommended embedding models:")
        print("-" * 80)
        for model_name, info in RECOMMENDED_MODELS.items():
            print(f"Model: {model_name}")
            print(f"  Dimension: {info['dimension']}")
            print(f"  Description: {info['description']}")
            print(f"  Speed: {info['speed']}")
            print(f"  Quality: {info['quality']}")
            print("-" * 80)
    elif args.command == "list-collections":
        # Load environment variables
        load_dotenv()
        
        # Initialize vector store
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
            return
        
        vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
        collections = vector_store.list_collections()
        
        # Format collections as a table
        table_data = []
        for collection in collections:
            table_data.append([
                collection.get("name", "N/A"),
                collection.get("vector_count", "N/A"),
                collection.get("vector_size", "N/A"),
                collection.get("created_at", "N/A")
            ])
        
        print(tabulate(
            table_data,
            headers=["Collection Name", "Vector Count", "Vector Size", "Created At"],
            tablefmt="grid"
        ))
    elif args.command == "delete-collection":
        # Load environment variables
        load_dotenv()
        
        # Initialize vector store
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
            return
        
        vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete collection '{args.collection_name}'? (y/n): ")
        if confirm.lower() != "y":
            print("Deletion cancelled.")
            return
        
        # Delete collection
        success = vector_store.delete_collection(args.collection_name)
        if success:
            print(f"✅ Successfully deleted collection '{args.collection_name}'")
        else:
            print(f"❌ Failed to delete collection '{args.collection_name}' (it might not exist)")
    elif args.command == "delete-all-collections":
        # Load environment variables
        load_dotenv()
        
        # Initialize vector store
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
            return
        
        vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
        collections = vector_store.list_collections()
        
        if not collections:
            print("No collections found.")
            return
        
        # Confirm deletion
        if not args.confirm:
            print("The following collections will be deleted:")
            for collection in collections:
                print(f"- {collection.get('name', 'N/A')}")
            confirm = input(f"Are you sure you want to delete ALL collections? (y/n): ")
            if confirm.lower() != "y":
                print("Deletion cancelled.")
                return
        
        # Delete all collections
        for collection in collections:
            collection_name = collection.get("name", "")
            if collection_name:
                success = vector_store.delete_collection(collection_name)
                if success:
                    print(f"✅ Successfully deleted collection '{collection_name}'")
                else:
                    print(f"❌ Failed to delete collection '{collection_name}'")
    else:
        parser.print_help()


def generate_embeddings_for_all_models(
    taxonomy_file: Path,
    max_level: int = 3,
    collection_prefix: str = "merchant_categories",
    test_suffix: str = "_test"
) -> None:
    """Generate embeddings for all recommended models.
    
    Args:
        taxonomy_file: Path to the taxonomy file
        max_level: Maximum level of categories to include
        collection_prefix: Prefix for collection names
        test_suffix: Suffix to append to collection names
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize vector store
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
        return
    
    vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    
    # Create a table to track performance
    performance_data = []
    
    # Process each model
    for model_name, model_info in RECOMMENDED_MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 80}")
        
        # Create collection name
        collection_name = f"{collection_prefix}_{model_name.replace('/', '_')}{test_suffix}"
        
        # Measure time
        start_time = time.time()
        
        # Generate embeddings
        try:
            populate_taxonomy_embeddings(
                taxonomy_file=taxonomy_file,
                max_level=max_level,
                collection_name=collection_name,
                embedding_model=model_name
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Add to performance data
            performance_data.append([
                model_name,
                model_info["dimension"],
                model_info["speed"],
                model_info["quality"],
                f"{elapsed_time:.2f}s",
                collection_name
            ])
            
            print(f"✅ Successfully generated embeddings for {model_name}")
            print(f"   Collection: {collection_name}")
            print(f"   Time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error generating embeddings for {model_name}: {str(e)}")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(tabulate(
        performance_data,
        headers=["Model", "Dimension", "Speed Rating", "Quality Rating", "Generation Time", "Collection"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    main()
