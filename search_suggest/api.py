"""
API for search suggestions.
"""
import os
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Depends, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from enum import Enum

from search_suggest.embeddings import EmbeddingService, RECOMMENDED_MODELS
from search_suggest.vector_store import VectorStore


# Load environment variables
load_dotenv()

app = FastAPI(title="Search Suggestions API")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global services
embedding_services: Dict[str, EmbeddingService] = {}
vector_store: Optional[VectorStore] = None

# Create an Enum for model selection in the API docs
class EmbeddingModelEnum(str, Enum):
    """Enum for embedding models."""
    # Add all recommended models as enum values
    # This will create a dropdown in the FastAPI docs
    MINI_LM = "all-MiniLM-L6-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    E5_SMALL = "intfloat/e5-small-v2"
    MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    MINI_LM_QA = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    MSMARCO = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"

# Pydantic models for request/response
class SearchResult(BaseModel):
    """Search result model."""
    id: str = Field(..., description="Unique identifier")
    score: float = Field(..., description="Similarity score")
    full_path: str = Field(..., description="Full category path")
    level: int = Field(..., description="Category level")
    
class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    dimension: int = Field(..., description="Embedding dimension")
    description: str = Field(..., description="Model description")
    speed: str = Field(..., description="Model speed")
    quality: str = Field(..., description="Model quality")

class CollectionInfo(BaseModel):
    """Collection information."""
    name: str = Field(..., description="Collection name")
    vector_count: Optional[int] = Field(None, description="Number of vectors in the collection")
    vector_size: Optional[int] = Field(None, description="Size of vectors in the collection")
    created_at: Optional[str] = Field(None, description="Creation timestamp")

class ComparisonRequest(BaseModel):
    """Comparison request model."""
    query: str = Field(..., description="Search query")
    models: List[EmbeddingModelEnum] = Field(..., description="Models to compare")
    limit: int = Field(10, description="Maximum number of results to return")

class ComparisonResult(BaseModel):
    """Comparison result model."""
    model: str = Field(..., description="Model name")
    query_time_ms: float = Field(..., description="Query time in milliseconds")
    results: List[SearchResult] = Field(..., description="Search results")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")

def get_embedding_service(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingService:
    """Get or create the embedding service.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Embedding service
    """
    global embedding_services
    if model_name not in embedding_services:
        embedding_services[model_name] = EmbeddingService(model_name=model_name)
    return embedding_services[model_name]

def get_vector_store() -> VectorStore:
    """Get or create the vector store.
    
    Returns:
        Vector store
    """
    global vector_store
    if vector_store is None:
        # Check if we should use a local Qdrant instance
        use_local_qdrant = os.getenv("USE_LOCAL_QDRANT", "false").lower() in ("true", "1", "yes")
        
        if use_local_qdrant:
            # Use local Qdrant instance (no API key needed)
            vector_store = VectorStore()
        else:
            # Use cloud Qdrant instance
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if not qdrant_url:
                raise HTTPException(
                    status_code=500, 
                    detail="QDRANT_URL environment variable is not set"
                )
            vector_store = VectorStore(url=qdrant_url, api_key=qdrant_api_key)
    return vector_store

def get_collection_for_model(model: str) -> str:
    """Determine the collection name based on the model.
    
    Args:
        model: Embedding model to use
        
    Returns:
        Collection name
    """
    # Standard format for all models with test suffix
    return f"merchant_categories_{model.replace('/', '_')}_test"

@app.get("/")
def root():
    """Root endpoint.
    
    Returns:
        HTML interface for the API
    """
    return FileResponse(static_dir / "index.html")

@app.get("/models", response_model=Dict[str, ModelInfo])
def list_models() -> Dict[str, ModelInfo]:
    """List available embedding models.
    
    Returns:
        Dictionary of model names to model information
    """
    models = EmbeddingService.list_recommended_models()
    
    # Format models to match the response model
    formatted_models = {}
    for model_name, model_info in models.items():
        formatted_models[model_name] = ModelInfo(
            name=model_name,
            dimension=model_info["dimension"],
            description=model_info.get("description", ""),
            speed=model_info.get("speed", ""),
            quality=model_info.get("quality", "")
        )
    
    return formatted_models

@app.get("/collections", response_model=List[CollectionInfo])
def list_collections(
    vector_store: VectorStore = Depends(get_vector_store)
) -> List[CollectionInfo]:
    """List available collections in the vector store.
    
    Returns:
        List of collections
    """
    collections = vector_store.list_collections()
    
    # Filter out collections we want to hide
    filtered_collections = [
        CollectionInfo(
            name=collection["name"],
            vector_count=collection.get("vector_count"),
            vector_size=collection.get("vector_size"),
            created_at=collection.get("created_at")
        )
        for collection in collections
        if collection["name"] not in ["merchant_categories", "merchant_categories_enriched"]
    ]
    
    return filtered_collections

@app.get("/search", response_model=List[SearchResult])
def search(
    query: str = Query(..., description="Search query"),
    model: EmbeddingModelEnum = Query(EmbeddingModelEnum.BGE_SMALL, description="Embedding model to use"),
    limit: int = Query(10, description="Maximum number of results to return"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store)
) -> List[SearchResult]:
    """Search for categories matching the query.
    
    Args:
        query: Search query
        model: Embedding model to use
        limit: Maximum number of results to return
        embedding_service: Embedding service
        vector_store: Vector store
        
    Returns:
        List of matching categories
    """
    import time
    start_time = time.time()
    
    # Get the string value from the Enum
    model_name = model.value
    
    # Create embedding for the query
    query_embedding = embedding_service.create_embedding(query, model_name=model_name)
    
    # Determine the collection name based on the model
    collection_name = get_collection_for_model(model_name)
    
    # Search for similar categories
    raw_results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )
    
    end_time = time.time()
    query_time_ms = (end_time - start_time) * 1000
    
    # Format results to match the response model
    results = []
    for result in raw_results:
        try:
            results.append(SearchResult(
                id=result["id"],
                score=result["score"],
                full_path=result["payload"]["full_path"],
                level=result["payload"]["level"]
            ))
        except KeyError as e:
            # Skip results that don't match the expected format
            continue
    
    return results

@app.post("/compare", response_model=List[ComparisonResult])
def compare(
    request: ComparisonRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store)
) -> List[ComparisonResult]:
    """Compare search results from multiple models.
    
    Args:
        request: Comparison request
        embedding_service: Embedding service
        vector_store: Vector store
        
    Returns:
        List of comparison results
    """
    results = []
    
    for model_enum in request.models:
        import time
        start_time = time.time()
        
        # Get the string value from the Enum
        model_name = model_enum.value
        
        # Create embedding for the query
        query_embedding = embedding_service.create_embedding(
            request.query, 
            model_name=model_name
        )
        
        # Determine the collection name based on the model
        collection_name = get_collection_for_model(model_name)
        
        # Search for similar categories
        raw_results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=request.limit or 10
        )
        
        end_time = time.time()
        query_time_ms = (end_time - start_time) * 1000
        
        # Format results to match the response model
        search_results = []
        for result in raw_results:
            try:
                search_results.append(SearchResult(
                    id=result["id"],
                    score=result["score"],
                    full_path=result["payload"]["full_path"],
                    level=result["payload"]["level"]
                ))
            except KeyError as e:
                # Skip results that don't match the expected format
                continue
        
        # Get model info if available
        model_info = None
        if model_name in EmbeddingService.list_recommended_models():
            model_info = EmbeddingService.list_recommended_models()[model_name]
        
        results.append(ComparisonResult(
            model=model_name,
            query_time_ms=query_time_ms,
            results=search_results,
            model_info=model_info
        ))
    
    return results

@app.post("/populate")
def populate_collection(
    model: str = Body(..., description="Embedding model to use"),
    collection_suffix: str = Body("", description="Optional suffix for collection name")
) -> Dict[str, Any]:
    """Populate a collection with embeddings from the specified model.
    
    This is an asynchronous operation that will run in the background.
    
    Args:
        model: Embedding model to use
        collection_suffix: Optional suffix for collection name
        
    Returns:
        Status message
    """
    import subprocess
    import threading
    
    # Validate model
    if model not in RECOMMENDED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available models: {list(RECOMMENDED_MODELS.keys())}"
        )
    
    # Create collection name
    collection_name = f"merchant_categories_{model.replace('/', '_')}{collection_suffix}"
    
    # Define function to run in background
    def run_populate():
        cmd = [
            "uv", "run", "python", "-m", "search_suggest.cli", "populate",
            "--collection", collection_name,
            "--embedding-model", model
        ]
        subprocess.run(cmd, check=True)
    
    # Start background thread
    thread = threading.Thread(target=run_populate)
    thread.daemon = True
    thread.start()
    
    return {
        "status": "started",
        "message": f"Started populating collection {collection_name} with model {model}",
        "collection": collection_name
    }
