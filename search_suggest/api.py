"""
API for search suggestions.
"""
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Depends, HTTPException

from search_suggest.embeddings import EmbeddingService
from search_suggest.vector_store import VectorStore


# Load environment variables
load_dotenv()

app = FastAPI(title="Search Suggestions API")

# Global services
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None

def get_embedding_service(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingService:
    """Get or create the embedding service.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Embedding service
    """
    global embedding_service
    if embedding_service is None or embedding_service.model_name != model_name:
        embedding_service = EmbeddingService(model_name=model_name)
    return embedding_service

def get_vector_store() -> VectorStore:
    """Get or create the vector store.
    
    Returns:
        Vector store
    """
    global vector_store
    if vector_store is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise HTTPException(
                status_code=500, 
                detail="QDRANT_URL or QDRANT_API_KEY environment variable is not set"
            )
        vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    return vector_store

@app.get("/")
def read_root():
    """Root endpoint.
    
    Returns:
        Welcome message
    """
    return {"message": "Welcome to the Search Suggestions API"}

@app.get("/models")
def list_models():
    """List available embedding models.
    
    Returns:
        List of available models with their characteristics
    """
    return EmbeddingService.list_recommended_models()

@app.get("/search")
def search(
    query: str = Query(..., description="Search query"),
    collection: str = Query("merchant_categories_enriched_local", description="Collection name"),
    limit: int = Query(10, description="Maximum number of results to return"),
    model: str = Query("BAAI/bge-small-en-v1.5", description="Embedding model to use"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store)
) -> List[Dict]:
    """Search for categories matching the query.
    
    Args:
        query: Search query
        collection: Collection name
        limit: Maximum number of results to return
        model: Embedding model to use
        embedding_service: Embedding service
        vector_store: Vector store
        
    Returns:
        List of matching categories
    """
    # Create embedding for the query
    query_embedding = embedding_service.create_embedding(query)
    
    # Search for similar categories
    results = vector_store.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=limit
    )
    
    return results
