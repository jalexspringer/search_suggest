"""
Script to populate the Qdrant database with taxonomy embeddings.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv

from search_suggest.taxonomy import TaxonomyParser
from search_suggest.embeddings import EmbeddingService
from search_suggest.vector_store import VectorStore


def populate_taxonomy_embeddings(
    taxonomy_file: Path,
    max_level: int = 3,
    collection_name: str = "merchant_categories",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> None:
    """Populate the Qdrant database with taxonomy embeddings.

    Args:
        taxonomy_file: Path to the taxonomy file
        max_level: Maximum level of categories to include
        collection_name: Name of the Qdrant collection
        embedding_model: Name of the sentence-transformers model to use
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variable is not set")
    
    # Initialize embedding service with local model
    embedding_service = EmbeddingService(model_name=embedding_model)
    vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    
    # Parse taxonomy
    parser = TaxonomyParser(taxonomy_file)
    rich_categories = parser.get_rich_categories_for_embedding(max_level)
    
    # Get vector dimension from the model
    vector_dimension = embedding_service.model.get_sentence_embedding_dimension()
    
    # Create collection with the correct vector size
    vector_store.create_collection(
        collection_name=collection_name,
        vector_size=vector_dimension
    )
    
    # Process in batches
    batch_size = 32
    total_batches = (len(rich_categories) + batch_size - 1) // batch_size
    
    print(f"Processing {len(rich_categories)} categories in {total_batches} batches")
    print(f"Using enriched category text with subcategories included")
    print(f"Using local embedding model: {embedding_model} (dimension: {vector_dimension})")
    
    for i in range(0, len(rich_categories), batch_size):
        batch = rich_categories[i:i+batch_size]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        print(f"Processing batch {i//batch_size + 1}/{total_batches}")
        
        # Create embeddings
        embeddings = embedding_service.create_embeddings_batch(texts)
        
        # Prepare payloads
        payloads = []
        for id_, text in batch:
            category = parser.categories[id_]
            payloads.append({
                "id": id_,
                "name": category["name"],
                "full_path": category["full_path"],
                "level": category["level"],
                "path_parts": category["path_parts"]
            })
        
        # Upsert to Qdrant
        vector_store.upsert_vectors(
            collection_name=collection_name,
            ids=ids,
            vectors=embeddings,
            payloads=payloads
        )
        
    print(f"Successfully populated {len(rich_categories)} categories into Qdrant")


if __name__ == "__main__":
    taxonomy_file = Path(__file__).parent.parent / "data" / "taxonomy.txt"
    populate_taxonomy_embeddings(taxonomy_file)
