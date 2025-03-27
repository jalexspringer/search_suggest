"""
Command-line interface for search suggestions.
"""
import argparse
from pathlib import Path

from search_suggest.populate_db import populate_taxonomy_embeddings
from search_suggest.embeddings import RECOMMENDED_MODELS

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
    
    # List models command
    models_parser = subparsers.add_parser("list-models", help="List recommended embedding models")
    
    args = parser.parse_args()
    
    if args.command == "populate":
        populate_taxonomy_embeddings(
            taxonomy_file=Path(args.taxonomy_file),
            max_level=args.max_level,
            collection_name=args.collection,
            embedding_model=args.embedding_model
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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
